import torch
import torch.nn.functional as F
import cv2
import json
import os
import itertools
import imageio
import numpy as np
import scipy.misc
import random
import math
from data_io import paths
from scipy import ndimage
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate
from learning.inputs.vision import standardize_image, standardize_images
from learning.datasets import image_load as iml
from learning.modules.image_resize import ImageResizer

from learning.models.visualization.viz_html_facebook_rpn import draw_boxes

from visualization import Presenter
from utils import dict_tools

TEST_EVAL_AUGMENTATION = False
OBJ_RADIUS = 0.4
MIN_BBOX_SIZE = 8

# TODO: Standardize augmentation for queries and scene


class RegionProposalOrRefinementDataset(Dataset):
    def __init__(self,
                 composed_dataset_name,
                 raw_dataset_name,
                 mode,
                 eval,
                 blur=False,
                 grain=False,
                 flip=False,
                 grayscale=False,
                 patch_size=32):
        """
        Dataset for training the region proposal network
        :param dataset_dir: directory where the composed dataset is stored
        """
        print("Indexing dataset...")
        self.mode = mode
        self._index_data(composed_dataset_name, raw_dataset_name)
        self.eval = eval
        self.blur = blur
        self.grain = grain
        self.grayscale = grayscale
        self.patch_size = patch_size
        self.runcount = 0

        self.resizer = ImageResizer()

    def _to_grayscale(self, np_image):
        return iml.image_to_grayscale(np_image)

    def _load_scene(self, sid):
        img_path = os.path.join(self.scene_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}--composed_scene.png")
        scene_img = np.asarray(imageio.imread(img_path)).astype(np.float32) * (1.0 / 255.0)
        if self.grayscale:
            scene_img = self._to_grayscale(scene_img)
        if self.blur or self.grain or TEST_EVAL_AUGMENTATION:
            scene_img = iml.augment_scene(scene_img, self.eval, self.blur, self.grain, TEST_EVAL_AUGMENTATION)
        return scene_img

    def _load_mask(self, sid, oid):
        mask_path = os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}", f"{oid}.png")
        mask = np.asarray(imageio.imread(mask_path)).astype(np.float32) * (1.0 / 255.0)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        return mask

    # TODO: Write a function that gets the correct mask based on
    def _load_mask_union(self, sid):
        """Loads all object masks for the scene, and builds a combined mask."""
        scene_mask_path = os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}")
        all_masks = [s for s in os.listdir(scene_mask_path) if s.endswith(".png")]
        mask_union = None
        for mask_fname in all_masks:
            mask_path = os.path.join(scene_mask_path, mask_fname)
            mask = np.asarray(imageio.imread(mask_path)).astype(np.float32) * (1.0 / 255.0)
            mask_union = np.maximum(mask_union, mask) if mask_union is not None else mask
        if len(mask_union.shape) > 2:
            mask_union = mask_union[:, :, 0]
        return mask_union

    def _load_mask(self, sid, oid):
        """Loads mask for the specific single object."""
        mask_path = os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}", f"{oid}.png")
        mask = np.asarray(imageio.imread(mask_path)).astype(np.float32) * (1.0 / 255.0)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        return mask

    def _load_bboxes(self, sid, scene_image, only_oid=None):
        geometry_path = os.path.join(self.bboxes_dir, f"{sid[0]}_{sid[1]}_{sid[2]}--geometry.json")
        with open(geometry_path, "r") as fp:
            geometry = json.load(fp)
        return self._get_center_based_bboxes(geometry, scene_image, only_oid=only_oid)

    def _get_center_based_bboxes(self, geometry, scene_image, only_oid=None):
        object_ids_out = []
        object_boxes_out = []
        for object_id, object_bottom_center, object_coords in zip(
            geometry["landmarkNames"], geometry["landmarkCenters"], geometry["landmarkCoordsInCam"]):
            if only_oid is not None and only_oid != object_id:
                continue
            if object_bottom_center is not None:
                bbox_scale = 1 / 6
                object_bottom_center = bbox_scale * np.asarray(object_bottom_center)
                dst_to_object = np.linalg.norm(object_coords)
                obj_radius_rad = math.atan2(OBJ_RADIUS, dst_to_object)
                obj_radius_px = math.degrees(obj_radius_rad) * scene_image.shape[1] / 84

                noisy_center = object_bottom_center# + np.random.normal([0, 0], obj_radius_px * 0.2)
                clip = lambda x, b, t:  int(min(max(x, b), t))

                min_x = clip(noisy_center[0] - obj_radius_px, 0, scene_image.shape[1])
                max_x = clip(noisy_center[0] + obj_radius_px, 0, scene_image.shape[1])
                min_y = clip(noisy_center[1] - obj_radius_px, 0, scene_image.shape[0])
                max_y = clip(noisy_center[1] + obj_radius_px, 0, scene_image.shape[0])

                object_box = [min_x, min_y, max_x, max_y]
                object_boxes_out.append(object_box)
                object_ids_out.append(object_id)
        return object_boxes_out, object_ids_out

    def _augment_query(self, query_img):
        return iml.augment_query_image(
            query_img, self.eval, self.blur, self.grain, TEST_EVAL_AUGMENTATION)

    def _eval_augmentation(self, img):
        return iml.eval_augment_query_image(img)

    def _index_data(self, dataset_name, raw_dataset_name):
        dataset_dir = paths.get_grounding_dataset_dir(dataset_name)
        raw_dataset_dir = paths.get_grounding_dataset_dir(raw_dataset_name)

        self.scene_images_dir = os.path.join(dataset_dir, "scenes")
        self.mask_images_dir = os.path.join(dataset_dir, "masks")
        self.object_images_dir = os.path.join(dataset_dir, "c_objects")
        self.bboxes_dir = raw_dataset_dir

        all_scene_image_paths = os.listdir(self.scene_images_dir)
        self.all_scene_ids = [tuple(p.split("--")[0].split("_")) for p in all_scene_image_paths]
        self.all_query_object_ids = os.listdir(self.object_images_dir)

        # Create an index of scene and object ids
        self.index = []
        # For region proposal, include all object ids
        if self.mode == "region_proposal":
            for sid in self.all_scene_ids:
                object_mask_fnames_this_scene = os.listdir(os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}"))
                objects_in_this_scene = [o.split(".")[0] for o in object_mask_fnames_this_scene]
                for oid in objects_in_this_scene:
                    self.index.append((sid, oid))

        # For region refinement, include only those ids that have visible bounding boxes
        elif self.mode == "region_refinement":
            self.bbox_log = {}
            mock_scene_image = np.zeros((96, 128, 3))
            for sid in self.all_scene_ids:
                bboxes, box_ids = self._load_bboxes(sid, mock_scene_image)
                for bbox, box_id in zip(bboxes, box_ids):
                    if bbox[2] - bbox[0] > MIN_BBOX_SIZE and bbox[3] - bbox[1] > MIN_BBOX_SIZE:
                        self.index.append((sid, box_id))
                        self.bbox_log[(sid, box_id)] = bbox

        print(f"Loaded dataset with:")
        print(f"   {len(self.index)} examples")
        print(f"   {len(self.all_scene_ids)} scenes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns a dataset of query image sets, scene images, and masks indicating query object instances in the scene
        :param idx:
        :return:
        """
        if self.mode == "region_proposal":
            return self.region_proposal_getitem(idx)
        elif self.mode == "region_refinement":
            return self.region_refinement_getitem(idx)
        else:
            raise ValueError(f"Unrecognized mode: {self.mode}")

    def region_proposal_getitem(self, idx):
        sid, oid = self.index[idx]
        scene_image = self._load_scene(sid)
        mask_union_image = self._load_mask_union(sid)
        bboxes, box_ids = self._load_bboxes(sid, scene_image)

        scene_image_t = torch.from_numpy(standardize_image(scene_image))
        mask_union_image_t = torch.from_numpy(mask_union_image)
        bboxes_t = torch.from_numpy(np.asarray(bboxes, dtype=np.float32))

        if False:
            scene_image_drawn = draw_boxes(scene_image_t, bboxes_t)
            Presenter().show_image(scene_image_drawn, "gt_boxes_DS", scale=4, waitkey=True)

        data_out = {
            "scene": scene_image_t,
            "mask_union": mask_union_image_t,
            "bboxes": bboxes_t,
            "sid": sid,
            "oid": oid
        }
        return data_out

    def region_refinement_getitem(self, idx):
        sid, oid = self.index[idx]
        scene_image = self._load_scene(sid)
        oid_mask_image = self._load_mask(sid, oid)
        bbox = self.bbox_log[(sid, oid)]
        minx, miny, maxx, maxy = bbox

        object_image_crop = scene_image[miny:maxy, minx:maxx, :]
        object_mask_crop = oid_mask_image[miny:maxy, minx:maxx]
        object_image_crop_t = torch.from_numpy(object_image_crop.transpose((2, 0, 1)))
        object_mask_crop_t = torch.from_numpy(object_mask_crop)

        # Must use the same image preprocessing pipeline as during test-time!
        object_image_crop_t_s = self.resizer.resize_to_target_size_and_normalize([object_image_crop_t])[0]
        # Add in the channel dimension before resizing, and remove it after. Don't normalize masks - they're 0/1 labels.
        object_mask_crop_t_s = self.resizer.resize_to_target_size([object_mask_crop_t.unsqueeze(0)])[0][0]
        return object_image_crop_t_s, object_mask_crop_t_s

    def join_lists(self, list_of_lists):
        return list(itertools.chain.from_iterable(list_of_lists))

    def collate_fn(self, list_of_examples):
        if self.mode == "region_proposal":
            examples_without_boxes = [dict_tools.dict_subtract(ex, ["bboxes"]) for ex in list_of_examples]
            only_bboxes = {"bboxes": [ex["bboxes"] for ex in list_of_examples]}
            batch_without_boxes = default_collate(examples_without_boxes)
            full_batch = dict_tools.dict_merge(batch_without_boxes, only_bboxes)
        else:
            full_batch = default_collate(list_of_examples)
        return full_batch


if __name__ == "__main__":
    pass

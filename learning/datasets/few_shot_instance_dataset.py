import torch
import cv2
import os
import itertools
import imageio
import numpy as np
import scipy.misc
import random
import math
from copy import deepcopy
from scipy import ndimage
from PIL import Image
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate
from learning.inputs.vision import standardize_image, standardize_images
from data_io.instructions import get_all_instructions, get_word_to_token_map
from learning.datasets import image_load as iml

from visualization import Presenter

PROFILE = False
DEBUG = False

TEST_EVAL_AUGMENTATION = False

NUM_QUERIES = 4
QUERY_SIZE = 32


class FewShotInstanceDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 eval,
                 query_scales=(32,),
                 sliding_window=False,
                 sliding_window_size=32,
                 sliding_window_stride=12,
                 window_scales=(12, 24, 48, 96),
                 blur=False,
                 grain=False,
                 grayscale=False,
                 use_all_queries=False):
        """
        Dataset for object recognizer
        :param dataset_dir: directory where the composed dataset is stored
        """
        print("Indexing dataset...")
        self.use_all_queries = use_all_queries
        self._index_data(dataset_dir)
        self.eval = eval
        self.query_scales = query_scales
        self.window_scales = window_scales
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size
        self.sliding_window_scales = window_scales
        #self.sliding_window_strides = [int(s * sliding_window_stride) for s in self.sliding_window_scales]
        self.sliding_window_stride = sliding_window_stride
        self.blur = blur
        self.grain = grain
        self.grayscale = grayscale

    def _to_grayscale(self, np_image):
        return iml.image_to_grayscale(np_image)

    def _load_scene(self, sid):
        img_path = os.path.join(self.scene_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}--composed_scene.png")
        scene_img = np.asarray(imageio.imread(img_path)).astype(np.float32) * (1.0 / 255.0)
        if self.grayscale:
            scene_img = self._to_grayscale(scene_img)
        scene_img = self._augment_scene(scene_img)
        return scene_img

    def _load_mask(self, sid, oid):
        mask_path = os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}", f"{oid}.png")
        mask = np.asarray(imageio.imread(mask_path)).astype(np.float32) * (1.0 / 255.0)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        return mask

    def _load_query(self, sid, oid):
        query_path = os.path.join(self.object_images_dir, f"{oid}", f"query_{oid}_{sid[0]}_{sid[1]}_{sid[2]}.png")
        return iml.load_query_image(query_path, self.grayscale)

    def _load_query_set(self, sid, oid):
        scenes_with_object = self.query_scenes_per_object[oid]
        layouts_with_object = [s[0] for s in scenes_with_object]
        layout = sid[0]
        if self.use_all_queries:
            allowed_layout_ids = layouts_with_object
            allowed_scene_ids = [s for s in scenes_with_object if s[0] in allowed_layout_ids]
            if len(allowed_scene_ids) >= NUM_QUERIES:
                query_scenes = random.sample(allowed_scene_ids, NUM_QUERIES)
            else:
                query_scenes = random.sample(allowed_scene_ids * NUM_QUERIES, NUM_QUERIES)
        else:
            allowed_layout_ids = set(layouts_with_object) - set([layout])
            allowed_scene_ids = [s for s in scenes_with_object if s[0] in allowed_layout_ids]
            query_scenes = random.sample(allowed_scene_ids, NUM_QUERIES)
        query_images = [self._load_query(s, oid) for s in query_scenes]
        query_images = [self._augment_query(q) for q in query_images]
        if len(self.query_scales) == 1:
            # Query images is a list of images
            query_images = [self._resize_query(q) for q in query_images]
        else:
            # Query images is a list of lists of images. For each query, we get a list of multiple scale images
            # in multiple different resolutions.
            query_images = [self._resize_multiscale_query(q) for q in query_images]
        return query_images

    def _augment_query(self, query_img):
        return iml.augment_query_image(
            query_img, self.eval, self.blur, self.grain, TEST_EVAL_AUGMENTATION)

    def _eval_augmentation(self, img):
        return iml.eval_augment_query_image(img)

    def _augment_scene(self, scene):
        #print("---")
        if not self.eval and self.blur and np.random.binomial(1, 0.5, 1) > 0.25:
            sigma = random.uniform(0.2, 2.0)
            scene = scipy.ndimage.gaussian_filter(scene, [sigma, sigma, 0])
            #print("Blur sigma: ", sigma)
        if not self.eval and self.grain and np.random.binomial(1, 0.5, 1) > 0.25:
            value_range = np.max(scene) - np.min(scene)
            sigma = random.uniform(value_range * 0.001, value_range * 0.05)
            scene = np.random.normal(scene, sigma)
            #print("grain sigma: ", sigma)

        if self.eval and TEST_EVAL_AUGMENTATION:
            scene = self._eval_augmentation(scene)
        #Presenter().show_image(scene, "scene_aug", scale=4, waitkey=True)
        return scene

    def _resize_query(self, query):
        return iml.resize_image_to_square(query, self.query_scales[0])

    def _resize_multiscale_query(self, query):
        out_queries = []
        for scale in self.query_scales:
            sized_query = self._resize_image_to_square(query, scale)
            out_queries.append(sized_query)
        return out_queries

    def _list_scenes_with_object_query(self, oid):
        object_apperances = os.listdir(os.path.join(self.object_images_dir, f"{oid}"))
        appearance_scene_ids = [tuple(o.split(".")[0].split("_")[2:5]) for o in object_apperances]
        return appearance_scene_ids

    def _index_data(self, dataset_dir):
        self.scene_images_dir = os.path.join(dataset_dir, "scenes")
        self.mask_images_dir = os.path.join(dataset_dir, "masks")
        self.object_images_dir = os.path.join(dataset_dir, "c_objects")

        all_scene_image_paths = os.listdir(self.scene_images_dir)
        self.all_scene_ids = [tuple(p.split("--")[0].split("_")) for p in all_scene_image_paths]
        self.all_query_object_ids = os.listdir(self.object_images_dir)

        # Loop through all objects and identify those that appear in more than one scene. These can be used
        self.query_scenes_per_object = {}
        for query_object_id in self.all_query_object_ids:
            appearance_scene_ids = self._list_scenes_with_object_query(query_object_id)
            appearance_layout_set = set([a[0] for a in appearance_scene_ids])
            if len(appearance_layout_set) >= (2 if self.use_all_queries else NUM_QUERIES + 1):
                if query_object_id not in self.query_scenes_per_object:
                    self.query_scenes_per_object[query_object_id] = []
                self.query_scenes_per_object[query_object_id] += appearance_scene_ids

        # Create an index of scene and object ids
        self.index = []
        for sid in self.all_scene_ids:
            object_mask_fnames_this_scene = os.listdir(os.path.join(self.mask_images_dir, f"{sid[0]}_{sid[1]}_{sid[2]}"))
            objects_in_this_scene = [o.split(".")[0] for o in object_mask_fnames_this_scene]
            valid_objects_in_this_scene = [o for o in objects_in_this_scene if o in self.query_scenes_per_object]
            for oid in valid_objects_in_this_scene:
                self.index.append((sid, oid))

        self.valid_scene_ids = list(set([i[0] for i in self.index]))
        print(f"Loaded dataset with:")
        print(f"   {len(self.index)} examples")
        print(f"   {len(self.valid_scene_ids)} scenes")
        print(f"   {len(self.query_scenes_per_object)} objects")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns a dataset of query image sets, scene images, and masks indicating query object instances in the scene
        :param idx:
        :return:
        """
        if self.sliding_window:
            return self.sliding_window_getitem(idx)
        else:
            return self.mask_getitem(idx)

    def mask_getitem(self, idx, return_extra_data=False):
        sid, oid = self.index[idx]
        scene_image = self._load_scene(sid)
        mask_image = self._load_mask(sid, oid)
        query_images = self._load_query_set(sid, oid)

        scene_image_t = torch.from_numpy(standardize_image(scene_image))
        mask_image_t = torch.from_numpy(mask_image)

        def standardize_query_images(list_of_images):
            return tuple([torch.from_numpy(standardize_image(q)) for q in list_of_images])

        if len(self.query_scales) == 1:
            query_images_t = standardize_query_images(query_images)
            query_images_t = torch.stack(query_images_t, dim=0)
        else:
            query_images_by_scale = list(zip(*query_images))
            query_images_t = [standardize_query_images(img_lst) for img_lst in query_images_by_scale]
            query_images_t = [torch.stack(q, dim=0) for q in query_images_t]
        extra_data = {
            "raw_scene": scene_image,
            "raw_mask": mask_image
        }
        data_out = {
            "scene": scene_image_t,
            "mask": mask_image_t,
            "query": query_images_t,
            "sid": sid,
            "oid": oid
        }
        if return_extra_data:
            return data_out, extra_data
        else:
            return data_out

    def sliding_window_getitem(self, idx):
        data_out, extra_data = self.mask_getitem(idx, True)
        scene_image = extra_data["raw_scene"]
        mask_image = extra_data["raw_mask"]
        window_images, window_masks, window_labels, window_scales = self.extract_window_images_and_labels(
            scene_image, mask_image, self.sliding_window_scales, self.sliding_window_stride, self.sliding_window_size)

        data_out["scene_windows"] = window_images
        data_out["mask_windows"] = window_masks
        data_out["window_labels"] = window_labels
        data_out["window_scales"] = window_scales
        return data_out

    @classmethod
    def extract_window_images_and_labels(cls, scene_image, mask_image, all_scales, stride, window_size):
        scene_windows = []
        mask_windows = []
        window_labels = []
        for scale in all_scales:
            scale_scene_windows = []
            scale_mask_windows = []
            scale_labels = []
            num_windows_h = int((scene_image.shape[0] - scale + stride) / stride)
            num_windows_w = int((scene_image.shape[1] - scale + stride) / stride)
            for t in range(0, scene_image.shape[0] - scale + 1, stride):
                b = t + scale
                for l in range(0, scene_image.shape[1] - scale + 1, stride):
                    r = l + scale
                    scene_crop = scene_image[t:b, l:r, :]
                    mask_crop = mask_image[t:b, l:r]
                    mask_label = int(mask_crop.sum() > 0.3 * mask_crop.shape[0] * mask_crop.shape[1])

                    scene_crop = cls._resize_image_to_square(scene_crop, window_size)
                    mask_crop = cls._resize_image_to_square(mask_crop, window_size)
                    scene_crop_t = torch.from_numpy(standardize_image(scene_crop))
                    mask_crop_t = torch.from_numpy(mask_crop)

                    scale_scene_windows.append(scene_crop_t)
                    scale_mask_windows.append(mask_crop_t)
                    scale_labels.append(mask_label)

            scale_scene_windows_t = torch.stack(scale_scene_windows, dim=0).view((num_windows_h, num_windows_w, 3, window_size, window_size))
            scale_mask_windows_t = torch.stack(scale_mask_windows, dim=0).view((num_windows_h, num_windows_w, 1, window_size, window_size))
            scale_mask_labels_t = torch.tensor(scale_labels).view((num_windows_h, num_windows_w))
            scene_windows.append(scale_scene_windows_t)
            mask_windows.append(scale_mask_windows_t)
            window_labels.append(scale_mask_labels_t)

        return scene_windows, mask_windows, window_labels, all_scales

    @classmethod
    def compute_window_back_masks(cls, scene_height, scene_width, all_scales, stride):
        back_masks = []
        for scale in all_scales:
            num_windows_h = int((scene_height - scale + stride) / stride)
            num_windows_w = int((scene_width - scale + stride) / stride)
            back_mask = np.zeros((num_windows_h, num_windows_w, scene_height, scene_width)).astype(np.bool)
            for h, t in enumerate(range(0, scene_height - scale + 1, stride)):
                b = t + scale
                for w, l in enumerate(range(0, scene_width - scale + 1, stride)):
                    r = l + scale
                    back_mask[h, w, t:b, l:r] = True
            back_masks.append(back_mask)
        return back_masks

    def join_lists(self, list_of_lists):
        return list(itertools.chain.from_iterable(list_of_lists))

    def collate_fn(self, list_of_examples):
        return default_collate(list_of_examples)


if __name__ == "__main__":
    test_dir = "/media/clic/shelf_space/grounding_data_out/composed/"
    dataset = FewShotInstanceDataset(test_dir)
    all_options = list(range(len(dataset)))
    random.shuffle(all_options)
    for i in all_options:
        example = dataset[i]
        p = Presenter()
        oid = example["oid"]
        sid = example["sid"]
        print(sid, oid)
        p.show_image(example["scene"], "scene", scale=4, waitkey=False)
        p.show_image(example["mask"], "mask", scale=4, waitkey=False)
        p.show_image(example["query"][0], "query", scale=4, waitkey=True)

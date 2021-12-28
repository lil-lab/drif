import os
import json
import random
import torch
from data_io import paths

from learning.datasets import image_load as iml
from learning.modules.image_resize import ImageResizer

IMAGES_PER_OBJECT = 5
STRINGS_PER_OBJECT = 5

AUGMENT = False


class StaticObjectDatabase():
    # TODO: Extend this to load a fixed novel object dataset and not a dynamically generated one
    def __init__(self, object_database_name, side_length):
        print("Using STATIC novel object dataset: ", object_database_name)
        object_database_dir = paths.get_object_database_dir(object_database_name)
        self.object_image_dir = os.path.join(object_database_dir, "c_objects")
        self.object_reference_path = os.path.join(object_database_dir, "object_references.json")
        with open(self.object_reference_path, "r") as fp:
            self.object_reference_dict = json.load(fp)
        self.query_img_side_length = side_length
        self.image_resizer = ImageResizer()
        self.data = None

    def get_landmark_list(self):
        landmark_list = os.listdir(self.object_image_dir)
        return landmark_list

    def build(self, device=None):
        landmarks_present = self.get_landmark_list()
        landmark_query_images = [self._load_landmark_query_images(lm_name) for lm_name in landmarks_present]
        landmark_strings = [self._load_landmark_reference_strings(lm_name) for lm_name in landmarks_present]
        # Convert numpy to torch
        landmark_query_images_t = [[torch.from_numpy(img).permute((2, 0, 1)) for img in images] for images in
                                   landmark_query_images]
        # Resize to square and normalize
        landmark_query_images_t = torch.stack(
            [self.image_resizer.resize_to_target_size_and_normalize(images) for images in landmark_query_images_t],
            dim=0)

        if device:
            landmark_query_images_t = landmark_query_images_t.to(device)

        object_database = {
            "object_references": landmark_strings,
            "object_images": landmark_query_images_t,
            "object_names": landmarks_present
        }
        return object_database

    def build_for_segment(self, seg_data):
        del seg_data # static database does not depend on segment
        if not self.data:
            self.data = self.build()
        return self.data

    def build_for_env(self, env_id, device=None):
        del env_id # static database does not depend on env_id
        if not self.data:
            self.data = self.build(device)
        return self.data

    def _load_object_query_image(self, path):
        query_img = iml.load_query_image(path)
        if AUGMENT:
            query_img = iml.augment_query_image(
                query_img, eval=True, grain=False, blur=False, test_eval_augmentation=True, rotate=False, flip=False)
        return query_img

    def _load_landmark_query_images(self, lm_name):
        lm_img_dir = os.path.join(self.object_image_dir, lm_name)
        image_filenames = os.listdir(lm_img_dir)
        sample_filenames = random.sample(image_filenames, IMAGES_PER_OBJECT)
        sample_paths = [os.path.join(lm_img_dir, f) for f in sample_filenames]
        images = [self._load_object_query_image(p) for p in sample_paths]
        return images

    def _load_landmark_reference_strings(self, lm_name):
        if lm_name in self.object_reference_dict:
            reference_strings = random.sample(self.object_reference_dict[lm_name], STRINGS_PER_OBJECT)
            return reference_strings
        raise ValueError(f"No object reference strings provided for landmark: {lm_name}")

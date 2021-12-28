import os
import json
import random
import torch
from data_io.env import load_and_convert_env_config
from data_io import paths

from learning.datasets import image_load as iml
from learning.modules.image_resize import ImageResizer

IMAGES_PER_OBJECT = 5
STRINGS_PER_OBJECT = 5


class DynamicObjectDatabase():
    # TODO: Extend this to load a fixed novel object dataset and not a dynamically generated one
    def __init__(self, object_database_name, side_length, text_only=False):
        print("Using novel object dataset: ", object_database_name)
        novel_object_data_dir = paths.get_object_database_dir(object_database_name)
        self.object_image_dir = os.path.join(novel_object_data_dir, "c_objects")
        self.object_reference_path = os.path.join(novel_object_data_dir, "object_references.json")
        with open(self.object_reference_path, "r") as fp:
            self.object_reference_dict = json.load(fp)
        self.query_img_side_length = side_length
        self.image_resizer = ImageResizer()
        self.text_only = text_only

    def build_for_segment(self, seg_data):
        # For each object in the environment, load a random set of images, and a random set of descriptions
        if len(seg_data) == 0:
          env_id = -1
        else:
          md = seg_data[0]["metadata"] if "metadata" in seg_data[0] else seg_data[0]
          env_id = md["env_id"]
        return self.build_for_env(env_id)

    def build_for_env(self, env_id, device=None):
        if env_id == -1:
          env_id = 0
        landmarks_present = self.get_list_of_landmarks_in_env(env_id)

        landmark_strings = [self._load_landmark_reference_strings(lm_name) for lm_name in landmarks_present]

        if not self.text_only:
            landmark_query_images = [self._load_landmark_query_images(lm_name) for lm_name in landmarks_present]
            # Convert numpy to torch
            landmark_query_images_t = [[torch.from_numpy(img).permute((2, 0, 1)) for img in images] for images in
                                    landmark_query_images]
            # Resize to square and normalize
            landmark_query_images_t = torch.stack(
                [self.image_resizer.resize_to_target_size_and_normalize(images) for images in landmark_query_images_t],
                dim=0)
            if device:
                landmark_query_images_t = landmark_query_images_t.to(device)
        else:
            # Create a dummy tensor that will pass through the collating mechanism
            landmark_query_images_t = torch.zeros([7], device=device)

        object_database = {
            "object_references": landmark_strings,
            "object_images": landmark_query_images_t,
            "object_names": landmarks_present
        }
        return object_database

    def get_list_of_landmarks_in_env(self, env_id):
        conf_json = load_and_convert_env_config(env_id)
        landmarks_present = conf_json["landmarkName"]
        return landmarks_present

    def _load_object_query_image(self, path):
        query_img = iml.load_query_image(path)
        return query_img

    def _load_landmark_query_images(self, lm_name):
        lm_img_dir = os.path.join(self.object_image_dir, lm_name)
        image_filenames = os.listdir(lm_img_dir)
        numimgs = len(image_filenames)
        assert IMAGES_PER_OBJECT <= len(image_filenames), (
            f"Not enough images for object: {lm_name}. Got: {numimgs}, Need: {IMAGES_PER_OBJECT}")
        sample_filenames = random.sample(image_filenames, IMAGES_PER_OBJECT)
        sample_paths = [os.path.join(lm_img_dir, f) for f in sample_filenames]
        images = [self._load_object_query_image(p) for p in sample_paths]
        return images

    def _load_landmark_reference_strings(self, lm_name):
        if lm_name in self.object_reference_dict:
            reference_strings = random.sample(self.object_reference_dict[lm_name], STRINGS_PER_OBJECT)
            return reference_strings
        raise ValueError(f"No object reference strings provided for landmark: {lm_name}")

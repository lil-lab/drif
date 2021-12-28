import torch
import os
import itertools
import random
import json
import numpy as np
import fcntl
import dill
from time import sleep
from data_io import paths
from data_io.paths import get_tmp_dir_for_run
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate
from learning.inputs.vision import standardize_image
from learning.datasets import image_load as iml
from learning.modules.image_resize import ImageResizer

from visualization import Presenter

PROFILE = False
DEBUG = False

TEST_EVAL_AUGMENTATION = False

OTHER_OBJECT_OBSERVATIONS = 5
NUM_OTHER_OBJECTS = 1

# Consider every possible pairing of two objects when deciding dataset length
FULL_LENGTH_TEST_2WAY_ONLY = False


class ObjectMultiMatchingDatasetMulti(Dataset):
    def __init__(self,
                 dataset_name,
                 eval,
                 run_name,
                 query_scale=32,
                 grayscale=False,
                 hard_negative_mining=False):
        """
        Dataset for object recognizer
        :param dataset_name: directory where the composed dataset is stored
        """
        print("Indexing dataset...")
        self.vector_dim = 32
        self.min_dist = 0.0
        self.max_dist = float("inf")
        self.dataset_name = dataset_name
        self.n_repeats = 1
        self._index_data()
        self.eval = eval
        self.query_scale = query_scale
        self.grayscale = grayscale
        self.hard_negative_mining = hard_negative_mining

        self.run_name = run_name

        self.image_resizer = ImageResizer()

    def _load_query_image(self, path):
        return iml.load_query_image(path, self.grayscale)

    def set_distance_filter_limits(self, min_dist, max_dist):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self._index_data()

    def set_repeats(self, repeats):
        self.n_repeats = repeats

    def _distance_filter(self, object_id_dir, object_id_files):
        if self.min_dist <= 0 and self.max_dist >= float("inf"):
            return object_id_files
        keep_object_id_files = []
        for obj_id_file in object_id_files:
            meta_file = obj_id_file.replace(".png", ".json")
            with open(os.path.join(object_id_dir, meta_file), "r") as fp:
                md = json.load(fp)
            dst_to_obj = md["dst_to_obj"]
            if self.min_dist < dst_to_obj < self.max_dist:
                keep_object_id_files.append(obj_id_file)
        print(f"Filtered out {len(object_id_files) - len(keep_object_id_files)}/{len(object_id_files)}"
              f"files based on object distance limits {self.min_dist}m to {self.max_dist}m")
        return keep_object_id_files

    def _index_data(self):
        dataset_dir = paths.get_grounding_dataset_dir(self.dataset_name)
        self.object_images_dir = os.path.join(dataset_dir, "c_objects")
        self.all_query_object_ids = os.listdir(self.object_images_dir)

        # Loop through all objects and count scenes that each object appears in
        self.all_object_data = {}
        for query_object_id in self.all_query_object_ids:
            object_id_dir = os.path.join(self.object_images_dir, query_object_id)
            object_id_files = [f for f in os.listdir(object_id_dir) if f.endswith(".png")]
            object_id_files = self._distance_filter(object_id_dir, object_id_files)
            object_sids = [o.split(".png")[0].split("_")[-3:] for o in object_id_files]
            object_data = {
                "filenames": object_id_files,
                "scene_ids": object_sids
            }
            self.all_object_data[query_object_id] = object_data
        # Filter only those objects that have enough observations
        self.all_object_data = {k: v for k, v in self.all_object_data.items() if len(v["scene_ids"]) > OTHER_OBJECT_OBSERVATIONS}
        self.all_object_ids = list(sorted(self.all_object_data.keys()))
        self.all_object_parings = list(itertools.permutations(self.all_object_ids, 2))
        num_objects = len(self.all_object_ids)
        total_images = sum([len(v["scene_ids"]) for k, v in self.all_object_data.items()])

        print(f"Loaded matching dataset with:")
        print(f"   {len(self.all_object_ids)} objects")
        print(f"   {total_images} images")
        print(f"   {len(self.all_object_parings)} object pairs")

    def _sample_other_objects(self, object_id):
        other_objects = [o for o in self.all_object_ids if o != object_id]
        return random.sample(other_objects, NUM_OTHER_OBJECTS)

    def __len__(self):
        return self.reallen() * self.n_repeats

    def reallen(self):
        if FULL_LENGTH_TEST_2WAY_ONLY and self.eval:
            return len(self.all_object_parings)
        else:
            return len(self.all_object_ids)

    def __getitem__(self, idx):
        """
        Returns
        Returns a triple (a, b, c), where a and b are observations of the sme object, while c is a stack of observations
        a N different objects.
        :param idx:
        :return:
        """
        idx = idx % self.reallen()
        if self.eval and FULL_LENGTH_TEST_2WAY_ONLY:
            object_id, other_object_id = self.all_object_parings[idx]
            other_object_ids = [other_object_id]
        else:
            object_id = self.all_object_ids[idx]
            other_object_ids = self._sample_other_objects(object_id)

        # Pick two random image of the selected object
        obj_img_a, path_a = self._load_random_image_for_id(object_id)
        obj_imgs_b, paths_b = self._load_random_images_for_id(object_id, n=OTHER_OBJECT_OBSERVATIONS, exclude_path=path_a)

        # Sample a random different object
        all_obj_imgs_c = [self._load_random_images_for_id(other_id, n=OTHER_OBJECT_OBSERVATIONS)[0] for other_id in other_object_ids]

        # First convert to PyTorch
        obj_img_a = torch.from_numpy(obj_img_a).permute((2, 0, 1))
        obj_imgs_b = [torch.from_numpy(img).permute((2, 0, 1)) for img in obj_imgs_b]
        all_obj_imgs_c = [[torch.from_numpy(img).permute((2, 0, 1)) for img in obj_imgs_c] for obj_imgs_c in all_obj_imgs_c]

        obj_a_t = self.image_resizer.resize_to_target_size_and_normalize([obj_img_a])[0]
        obj_b_t = self.image_resizer.resize_to_target_size_and_normalize(obj_imgs_b)
        obj_c_t = [self.image_resizer.resize_to_target_size_and_normalize(obj_imgs_c) for obj_imgs_c in all_obj_imgs_c]
        obj_c_t = torch.stack(obj_c_t, dim=0)

        return obj_a_t, obj_b_t, obj_c_t, object_id, other_object_ids

    def _load_random_image_for_id(self, object_id, exclude_path=None):
        object_filenames = self.all_object_data[object_id]["filenames"]
        object_paths = [os.path.join(self.object_images_dir, object_id, fname) for fname in object_filenames]
        if exclude_path is not None:
            object_paths = list(set(object_paths).difference(set(exclude_path)))
        picked_path = random.sample(object_paths, 1)[0]
        image = self._load_query_image(picked_path)
        return image, picked_path

    def _load_random_images_for_id(self, object_id, n, exclude_path=None):
        object_filenames = self.all_object_data[object_id]["filenames"]
        object_paths = [os.path.join(self.object_images_dir, object_id, fname) for fname in object_filenames]
        if exclude_path is not None:
            object_paths = list(set(object_paths).difference(set(exclude_path)))
        picked_paths = random.sample(object_paths, n)
        images = [self._load_query_image(p) for p in picked_paths]
        return images, picked_paths

    def join_lists(self, list_of_lists):
        return list(itertools.chain.from_iterable(list_of_lists))

    def collate_fn(self, list_of_examples):
        obj_a, obj_b, obj_c, id_a, id_c = default_collate(list_of_examples)
        # For some reason the d
        id_c = [ex[4] for ex in list_of_examples]
        return obj_a, obj_b, obj_c, id_a, id_c


if __name__ == "__main__":
    test_dir = "/media/clic/BigStore/grounding_data_both_sim_3_huge/composed/"
    dataset = ObjectMultiMatchingDatasetMulti(test_dir, eval=False)
    all_options = list(range(len(dataset)))
    random.shuffle(all_options)
    for i in all_options:
        obj_a, obj_b, obj_c, id_ab, id_c = dataset[i]
        p = Presenter()
        p.show_image(obj_a, "obj_a", scale=4, waitkey=False)
        p.show_image(obj_b, "obj_b", scale=4, waitkey=False)
        p.show_image(obj_c, "obj_c", scale=4, waitkey=True)

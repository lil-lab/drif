import torch
import os
import itertools
import random
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

# Consider every possible pairing of two objects when deciding dataset length
FULL_LENGTH_TEST = True


class ObjectMultiMatchingDataset(Dataset):
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
        self._index_data(dataset_name)
        self.eval = eval
        self.query_scale = query_scale
        self.grayscale = grayscale
        self.hard_negative_mining = hard_negative_mining

        #self.confusion_matrix_path = os.path.join(get_tmp_dir_for_run(run_name), "matching_confusion.npy"
        self.prediction_cache_path = os.path.join(get_tmp_dir_for_run(run_name), "prediction_cache.npy")
        self.fpw = open(self.prediction_cache_path, "ab")
        self.run_name = run_name

        self.cache_updates = 0
        self.store_cache_every_n = 50
        self.getitem_count = 0
        self.load_cache_every_n = 2000

        self.image_resizer = ImageResizer()

        # Store the matrix initially
        self._store_cache()

    def _load_query_image(self, path):
        return iml.load_query_image(path, self.grayscale)

    def _index_data(self, dataset_name):
        dataset_dir = paths.get_grounding_dataset_dir(dataset_name)
        self.object_images_dir = os.path.join(dataset_dir, "c_objects")
        self.all_query_object_ids = os.listdir(self.object_images_dir)

        # Loop through all objects and count scenes that each object appears in
        self.all_object_data = {}
        for query_object_id in self.all_query_object_ids:
            object_id_dir = os.path.join(self.object_images_dir, query_object_id)
            object_id_files = os.listdir(object_id_dir)
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

        self.vector_cache = np.ones((num_objects, self.vector_dim))
        self.sampling_matrix = np.ones((num_objects, num_objects))

        print(f"Loaded matching dataset with:")
        print(f"   {len(self.all_object_ids)} objects")
        print(f"   {total_images} images")
        print(f"   {len(self.all_object_parings)} object pairs")

    def _load_cache(self):
        fcntl.lockf(self.fpw, fcntl.LOCK_EX)
        with open(self.prediction_cache_path, "rb") as fp:
            self.vector_cache = dill.load(fp)
        fcntl.lockf(self.fpw, fcntl.LOCK_UN)
        self._update_sampling_matrix()

    def _update_sampling_matrix(self):
        a = self.vector_cache[np.newaxis, :, :]
        b = self.vector_cache[:, np.newaxis, :]
        self.sampling_matrix = 1 / (((a - b) ** 2).sum(2) + 1e-9)
        print("Updated sampling matrix")

    def _store_cache(self):
        fcntl.lockf(self.fpw, fcntl.LOCK_EX)
        with open(self.prediction_cache_path, "wb") as fp:
            dill.dump(self.vector_cache, fp)
        fcntl.lockf(self.fpw, fcntl.LOCK_UN)

    def _sample_other_object(self, object_id):
        if self.hard_negative_mining and not self.eval:
            object_idx = self.all_object_ids.index(object_id)
            other_obj_pdf = self.sampling_matrix[object_idx]
            other_obj_pdf[object_idx] = 0.0
            other_obj_prob = other_obj_pdf / other_obj_pdf.sum()
            other_obj_cdf = np.cumsum(other_obj_prob)
            randnum = np.random.uniform(0, 1, 1)
            shifted_cdf_squared = (other_obj_cdf - randnum) ** 2
            other_obj_index = int(np.argmin(shifted_cdf_squared))
            return self.all_object_ids[other_obj_index]
        else:
            return random.sample(set(self.all_object_ids).difference(set(object_id)), 1)[0]

    def log_predictions(self, vectors_a, vectors_b, vectors_c, ids_ab, ids_c):
        for id_a, vector_a in zip(ids_ab, vectors_a):
            idx = self.all_object_ids.index(id_a)
            self.vector_cache[idx] = vector_a.detach().cpu().numpy()
        for id_b, vector_b in zip(ids_ab, vectors_b):
            idx = self.all_object_ids.index(id_b)
            self.vector_cache[idx] = vector_b.detach().cpu().numpy()
        for id_c, vector_c in zip(ids_c, vectors_c):
            idx = self.all_object_ids.index(id_c)
            self.vector_cache[idx] = vector_c.detach().cpu().numpy()
        if self.cache_updates % self.store_cache_every_n == 0:
            self._store_cache()
        self.cache_updates += 1

    def __len__(self):
        if FULL_LENGTH_TEST and self.eval:
            return len(self.all_object_parings)
        else:
            return len(self.all_object_ids)

    def __getitem__(self, idx):
        """
        Returns a triple (a, b, c), where a and b are observations of the sme object, while c is an observation of
        a different object.
        :param idx:
        :return:
        """
        self.getitem_count += 1
        if self.hard_negative_mining:
            if self.getitem_count % self.load_cache_every_n == 0:
                self._load_cache()

        if self.eval and FULL_LENGTH_TEST:
            object_id, other_object_id = self.all_object_parings[idx]
        else:
            object_id = self.all_object_ids[idx]
            other_object_id = self._sample_other_object(object_id)

        # Pick two random image of the selected object
        obj_img_a, path_a = self._load_random_image_for_id(object_id)
        obj_imgs_b, paths_b = self._load_random_images_for_id(object_id, n=OTHER_OBJECT_OBSERVATIONS, exclude_path=path_a)

        # Sample a random different object
        obj_imgs_c, paths_c = self._load_random_images_for_id(other_object_id, n=OTHER_OBJECT_OBSERVATIONS)

        # First convert to PyTorch
        obj_img_a = torch.from_numpy(obj_img_a).permute((2, 0, 1))
        obj_imgs_b = [torch.from_numpy(img).permute((2, 0, 1)) for img in obj_imgs_b]
        obj_imgs_c = [torch.from_numpy(img).permute((2, 0, 1)) for img in obj_imgs_c]

        obj_a_t = self.image_resizer.resize_to_target_size_and_normalize([obj_img_a])[0]
        obj_b_t = self.image_resizer.resize_to_target_size_and_normalize(obj_imgs_b)
        obj_c_t = self.image_resizer.resize_to_target_size_and_normalize(obj_imgs_c)

        return obj_a_t, obj_b_t, obj_c_t, object_id, other_object_id

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
        return default_collate(list_of_examples)


if __name__ == "__main__":
    test_dir = "/media/clic/BigStore/grounding_data_both_sim_3_huge/composed/"
    dataset = ObjectMultiMatchingDataset(test_dir, eval=False)
    all_options = list(range(len(dataset)))
    random.shuffle(all_options)
    for i in all_options:
        obj_a, obj_b, obj_c, id_ab, id_c = dataset[i]
        p = Presenter()
        p.show_image(obj_a, "obj_a", scale=4, waitkey=False)
        p.show_image(obj_b, "obj_b", scale=4, waitkey=False)
        p.show_image(obj_c, "obj_c", scale=4, waitkey=True)

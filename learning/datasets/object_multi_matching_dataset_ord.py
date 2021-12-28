import torch
import os
import itertools
import random
import numpy as np
import fcntl
import dill

from data_io import paths
from data_io.instructions import get_restricted_env_id_lists
from torch.utils.data import Dataset
# This used to be not protected, inside dataloader module
from torch.utils.data._utils.collate import default_collate
from learning.datasets import image_load as iml
from learning.datasets.static_object_database import StaticObjectDatabase
from learning.datasets.dynamic_object_database import DynamicObjectDatabase
from learning.modules.image_resize import ImageResizer

from visualization import Presenter

PROFILE = False
DEBUG = False

TEST_EVAL_AUGMENTATION = False

OTHER_OBJECT_OBSERVATIONS = 5

# Consider every possible pairing of two objects when deciding dataset length
FULL_LENGTH_TEST = True


class ObjectMultiMatchingDatasetOrd(Dataset):
    def __init__(self,
                 dataset_name,
                 eval,
                 run_name,
                 query_scale=32,
                 static_database=False):
        """
        Dataset for object recognizer
        :param dataset_name: directory where the composed dataset is stored
        """
        print("Indexing dataset...")
        self.static_database = static_database
        self.vector_dim = 32
        self.eval = eval
        self.query_scale = query_scale

        if self.static_database:
            self.object_database = StaticObjectDatabase(dataset_name, self.query_scale)
        else:
            self.object_database = DynamicObjectDatabase(dataset_name, self.query_scale)

        self.run_name = run_name
        self.image_resizer = ImageResizer()
        self.train_envs, _, _ = get_restricted_env_id_lists()

    def set_object_distance_filter_limits(self, min_distance, max_distance):
        ...

    def __len__(self):
        return len(self.train_envs)

    def __getitem__(self, idx):
        """
        Returns a triple (a, b, c), where a and b are observations of the sme object, while c is an observation of
        a different object.
        :param idx:
        :return:
        """
        env_id = self.train_envs[idx]
        database = self.object_database.build_for_env(env_id, device="cpu")

        # num_obj x img_per_obj x channels x 32 x 32
        object_images = database["object_images"]
        object_ids = database["object_names"]

        return object_images, object_ids

    def collate_fn(self, list_of_examples):
        return default_collate(list_of_examples)

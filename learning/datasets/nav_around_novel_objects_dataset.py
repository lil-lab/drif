import torch
from torch.utils.data import Dataset
from learning.inputs.sequence import instruction_sequence_batch_to_tensor
from learning.datasets.segment_dataset_simple import SegmentDataset
from learning.datasets.dynamic_object_database import DynamicObjectDatabase
import parameters.parameter_server as P

PROFILE = False
DEBUG = False


class NavAroundNovelObjectsDataset(Dataset):
    def __init__(self,
                 model,
                 object_database_name,
                 query_img_side_length,
                 data=None,
                 env_list=None,
                 dataset_names=["simulator"],
                 dataset_prefix="supervised",
                 domain="sim",
                 max_traj_length=None,
                 aux_provider_names=[],
                 segment_level=False,
                 index_by_env=False):
        """
        Dataset for the replay memory
        :param model: The FewShot_Stage1_Bidomain model instance that will be using this data
        :param data: if data is pre-loaded in memory, this is the training data
        :param env_list: if data is to be loaded by the dataset, this is the list of environments for which to include data
        :param dataset_names: list of datasets from which to load data
        :param dataset_prefix: name of the dataset. Default: supervised will use data collected with collect_supervised_data
        :param max_traj_length: truncate trajectories to this long
        :param cuda:
        :param aux_provider_names:
        """
        self.model = model
        self.segment_dataset = SegmentDataset(data,
                                              env_list,
                                              dataset_names,
                                              dataset_prefix,
                                              domain,
                                              max_traj_length,
                                              aux_provider_names,
                                              segment_level,
                                              index_by_env=index_by_env)

        self.object_database = DynamicObjectDatabase(object_database_name, query_img_side_length)

    def set_size_cap(self, size_cap):
        self.segment_dataset.set_size_cap(size_cap)

    def __len__(self):
        return len(self.segment_dataset)

    def __getitem__(self, idx):
        # Sometimes in background workers, the parameters won't be carried along during spawn
        if P.get_current_parameters() is None:
            P.initialize_experiment()
        seg_data = self.segment_dataset.get_seg_data(idx)
        example = self.segment_dataset.build_sample(idx, seg_data, blur=True)
        if example is None or len(seg_data) == 0:
            return None
        object_database = self.object_database.build_for_segment(seg_data)
        example["object_database"] = object_database
        return example

    def split_into_segments(self, env_list, dname):
        return self.segment_dataset.split_into_segments(env_list, dname)

    def filter_segment_availability(self, dname, envs, segs):
        return self.segment_dataset.filter_segment_availability(dname, envs, segs)

    def set_word2token(self, token2term, word2token):
        self.segment_dataset.set_word2token(token2term, word2token)

    def add_batch_dims(self, batch):
        for item in batch:
            for k, v in item.items():
                if hasattr(v, "unsqueeze"):
                    item[k] = v.unsqueeze(0)
                else:
                    item[k] = [v]
        return batch

    def collate_fn(self, list_of_samples):
        if None in list_of_samples:
            return None

        data_batch = [self.segment_dataset.collate_fn([example]) for example in list_of_samples]
        processed_batch = [self.model.preprocess_batch(b) for b in data_batch]

        return processed_batch

from collections import namedtuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import parameters.parameter_server as P

from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map, load_instruction
from data_io.train_data import load_single_env_from_dataset, filter_env_list_has_data
from env_config.definitions.landmarks import get_landmark_names
from learning.inputs.sequence import pad_segment_with_nones, \
    sequence_list_to_tensor, none_padded_seq_to_tensor, instruction_sequence_batch_to_tensor
from learning.inputs.vision import standardize_images, standardize_depth_images
from learning.datasets.aux_data_providers import resolve_data_provider, get_aux_label_names, get_stackable_label_names

from utils.dict_tools import dict_zip, dict_map
from utils.simple_profiler import SimpleProfiler

Sample = namedtuple("Sample", ("instruction", "state", "action", "reward", "done", "metadata"))

PROFILE = False


class SegmentDataset(Dataset):
    def __init__(self,
                 data=None,
                 env_list=None,
                 dataset_name="supervised",
                 max_traj_length=None,
                 aux_provider_names=[],
                 segment_level=False,
                 cache=True):
        """
        Dataset for the replay memory
        :param data: if data is pre-loaded in memory, this is the training data
        :param env_list: if data is to be loaded by the dataset, this is the list of environments for which to include data
        :param dataset_name: name of the dataset. Default: supervised will use data collected with collect_supervised_data
        :param max_traj_length: truncate trajectories to this long
        :param cuda:
        :param aux_provider_names:
        """

        # If data is already loaded in memory, use it
        self.data = data
        self.prof = SimpleProfiler(torch_sync=False, print=PROFILE)
        # Otherwise use the env list
        if env_list is not None:
            self.env_list = filter_env_list_has_data(env_list, dataset_name)

        self.dataset_name = dataset_name

        self.max_traj_length = max_traj_length
        train_instr, dev_instr, test_instr, corpus = get_all_instructions()
        # TODO: This shouldn't have access to all instructions. We should really make distinct train, dev, test modes
        self.all_instr = {**train_instr, **dev_instr, **test_instr}

        self.segment_level = segment_level
        if self.data is None:
            if segment_level:
                self.env_list, self.seg_list = self.split_into_segments(self.env_list)
            else:
                self.seg_list = [0 for env in self.env_list]

        self.token2word, self.word2token = get_word_to_token_map(corpus)
        self.aux_provider_names = aux_provider_names
        self.aux_label_names = get_aux_label_names(aux_provider_names)
        self.stackable_names = get_stackable_label_names(aux_provider_names)
        self.do_cache = cache
        self.data_cache = {}

    def load_env_data(self, env_id):
        if self.do_cache:
            if env_id not in self.data_cache:
                self.data_cache[env_id] = load_single_env_from_dataset(env_id, self.dataset_name)
            return self.data_cache[env_id]
        else:
            return load_single_env_from_dataset(env_id, self.dataset_name)

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return len(self.env_list)

    def __getitem__(self, idx):
        self.prof.tick("out")
        # If data is already loaded, use it
        if self.data is not None:
            seg_data = self.data[idx]
            if type(seg_data) is int:
                raise NotImplementedError("Mixing dynamically loaded envs with training data is no longer supported.")
                seg_data = load_single_env_from_dataset(seg_data, "supervised")
        else:
            env_id = self.env_list[idx]
            env_data = self.load_env_data(env_id)
            if self.segment_level:
                seg_idx = self.seg_list[idx]
                seg_data = []
                for sample in env_data:
                    if sample["metadata"]["seg_idx"] == seg_idx:
                        seg_data.append(sample)
            else:
                seg_data = env_data

        if len(seg_data) == 0:
            return None

        # Convert to tensors, replacing Nones with zero's
        images_in = [seg_data[i]["state"].image if i < len(seg_data) else None for i in range(len(seg_data))]
        states = [seg_data[i]["state"].state if i < len(seg_data) else None for i in range(len(seg_data))]

        images_np = standardize_images(images_in)
        images = none_padded_seq_to_tensor(images_np)

        depth_images_np = standardize_depth_images(images_in)
        depth_images = none_padded_seq_to_tensor(depth_images_np)

        states = none_padded_seq_to_tensor(states)

        actions = none_padded_seq_to_tensor([s["ref_action"] for s in seg_data])
        stops = [1.0 if s["done"] else 0.0 for s in seg_data]

        # e.g. [1 1 1 1 1 1 0 0 0 0 .. 0] for segment with 6 samples
        mask = [1.0 if s["ref_action"] is not None else 0.0 for s in seg_data]

        stops = torch.FloatTensor(stops)
        mask = torch.FloatTensor(mask)

        # This is a list, converted to tensor in collate_fn
        #if INSTRUCTIONS_FROM_FILE:
        #    tok_instructions = [tokenize_instruction(load_instruction(md["env_id"], md["set_idx"], md["seg_idx"]), self.word2token) if s["md"] is not None else None for s in seg_data]
        #else:
        tok_instructions = [tokenize_instruction(s["instruction"], self.word2token) if s["instruction"] is not None else None for s in seg_data]

        md = [seg_data[i]["metadata"] for i in range(len(seg_data))]
        flag = md[0]["flag"] if "flag" in md[0] else None

        data = {
            "instr": tok_instructions,
            "images": images,
            "depth_images": depth_images,
            "states": states,
            "actions": actions,
            "stops": stops,
            "masks": mask,
            "flags": flag,
            "md": md
        }

        self.prof.tick("getitem_core")
        for aux_provider_name in self.aux_provider_names:
            aux_datas = resolve_data_provider(aux_provider_name)(seg_data, data)
            for d in aux_datas:
                data[d[0]] = d[1]
            self.prof.tick("getitem_" + aux_provider_name)

        return data

    def split_into_segments(self, env_list):
        envs = []
        segs = []
        for env_id in env_list:
            # 0th instr set
            instruction_set = self.all_instr[env_id][0]["instructions"]
            for seg_idx in range(len(instruction_set)):
                envs.append(env_id)
                segs.append(seg_idx)
        return envs, segs

    def set_word2token(self, token2term, word2token):
        self.token2term = token2term
        self.word2token = word2token

    def stack_tensors(self, one):
        if one is None:
            return None
        one = torch.stack(one, dim=0)
        one = Variable(one)
        return one

    def collate_fn(self, list_of_samples):
        self.prof.tick("out")
        if None in list_of_samples:
            return None

        data_batch = dict_zip(list_of_samples)

        data_t = dict_map(data_batch, self.stack_tensors,
                          ["images", "depth_images", "states", "actions", "stops", "masks"] + self.stackable_names)

        instructions_t, instruction_lengths = instruction_sequence_batch_to_tensor(data_batch["instr"])

        data_t["instr"] = instructions_t
        data_t["instr_len"] = instruction_lengths
        data_t["cam_pos"] = data_t["states"][:, 9:12].clone()
        data_t["cam_rot"] = data_t["states"][:, 12:16].clone()

        self.prof.tick("collate")
        self.prof.loop()
        self.prof.print_stats(5)
        return data_t
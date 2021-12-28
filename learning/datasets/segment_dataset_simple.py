from collections import namedtuple

import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.utils.data import Dataset

from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map, get_instruction_segment
from data_io.train_data import load_single_env_from_dataset, load_single_env_metadata_from_dataset, filter_env_list_has_data
from learning.inputs.sequence import none_padded_seq_to_tensor, instruction_sequence_batch_to_tensor
from learning.inputs.vision import standardize_images
from learning.datasets.aux_data_providers import resolve_data_provider, get_aux_label_names, get_stackable_label_names
from learning.datasets import image_load as iml

from utils.dict_tools import dict_zip, dict_map
from utils.simple_profiler import SimpleProfiler

import parameters.parameter_server as P

PROFILE = False
DEBUG = False


class SegmentDataset(Dataset):
    def __init__(self,
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
        :param data: if data is pre-loaded in memory, this is the training data
        :param env_list: if data is to be loaded by the dataset, this is the list of environments for which to include data
        :param dataset_names: list of datasets from which to load data
        :param dataset_prefix: name of the dataset. Default: supervised will use data collected with collect_supervised_data
        :param max_traj_length: truncate trajectories to this long
        :param index_by_env: if true, dataset length is the number of envs, and when retrieving one, a random segment will be drawn.
        :param cuda:
        :param aux_provider_names:
        """

        del data
        self.prof = SimpleProfiler(torch_sync=False, print=PROFILE)
        self.min_seg_len = P.get_current_parameters()["Data"].get("min_seg_len", 3)
        self.dataset_prefix = dataset_prefix
        self.dataset_names = dataset_names
        self.domain = domain
        self.index_by_env = index_by_env
        self.size_cap = None

        self.env_restrictions = P.get_current_parameters()["Data"].get("dataset_env_restrictions")
        if self.env_restrictions:
            self.dataset_restricted_envs = {dname:P.get_current_parameters()["Data"]["EnvRestrictionGroups"][self.env_restrictions[dname]] for dname in dataset_names if dname in self.env_restrictions}
            print(f"Using restricted envs: {list(self.dataset_restricted_envs.keys())}")
        else:
            self.dataset_restricted_envs = {}

        self.max_traj_length = max_traj_length
        train_instr, dev_instr, test_instr, corpus = get_all_instructions()
        # TODO: This shouldn't have access to all instructions. We should really make distinct train, dev, test modes
        self.all_instr = {**train_instr, **dev_instr, **test_instr}

        train_instr_full, dev_instr_full, test_instr_full, corpus = get_all_instructions(ignore_min_augment_len=True)
        self.all_instr_full = {**train_instr_full, **dev_instr_full, **test_instr_full}

        self.segment_level = segment_level
        self.sample_ids = []

        assert env_list is not None
        for i, dataset_name in enumerate(self.dataset_names):
            dataset_env_list = filter_env_list_has_data(dataset_name, env_list, dataset_prefix)
            if self.segment_level:
                dataset_env_list, dataset_seg_list = self.split_into_segments(dataset_env_list, dataset_name)
            else:
                dataset_seg_list = [0 for _ in dataset_env_list]
            for env, seg in zip(dataset_env_list, dataset_seg_list):
                self.sample_ids.append((dataset_name, env, seg))

        self.token2word, self.word2token = get_word_to_token_map(corpus)
        self.aux_provider_names = aux_provider_names
        self.aux_label_names = get_aux_label_names(aux_provider_names)
        self.stackable_names = get_stackable_label_names(aux_provider_names)

        self.traj_len = P.get_current_parameters()["Setup"]["trajectory_length"]

    def load_env_data(self, dataset_name, env_id):
        return load_single_env_from_dataset(dataset_name, env_id, self.dataset_prefix)

    def set_size_cap(self, size_cap):
        self.size_cap = size_cap

    def __len__(self):
        datalen = len(self.sample_ids)
        if self.size_cap:
            datalen = min(self.size_cap, datalen)
        return datalen

    def get_seg_data(self, idx):
        # If data is already loaded, use it
        dataset_name, env_id, seg_idx = self.sample_ids[idx]
        env_data = self.load_env_data(dataset_name, env_id)

        if not self.segment_level:
            seg_data = env_data
        else:
            seg_data = []
            segs_in_data = set()
            for sample in env_data:
                if "metadata" not in sample:
                    sample["metadata"] = sample
                sample["metadata"]["domain"] = self.domain
                segs_in_data.add(sample["metadata"]["seg_idx"])

            # Keep the segments for which we have instructions
            segs_in_data_and_instructions = set()
            for _seg_idx in segs_in_data:
                if get_instruction_segment(env_id, 0, _seg_idx, all_instr=self.all_instr_full) is not None:
                    segs_in_data_and_instructions.add(_seg_idx)

            if seg_idx not in segs_in_data_and_instructions:
                if DEBUG: print(f"Segment {env_id}::{seg_idx} not in (data)and(instructions)")
                # We specificially draw random segments from each environment
                if seg_idx == "random":
                    seg_idx = random.choice(list(segs_in_data_and_instructions)) if len(segs_in_data_and_instructions) > 0 else -1
                # If there's a single segment in this entire dataset, just return that segment even if it's not a match.
                elif len(segs_in_data) == 1:
                    seg_data = env_data
                # Otherwise return a random segment instead
                elif len(segs_in_data_and_instructions) > 0:
                    # print(f"seg idx: {seg_idx} for env: {env_id} missing. Drawing randomly.")
                    seg_idx = random.choice(list(segs_in_data_and_instructions))
                elif dataset_name == "real" and len(segs_in_data) > 0:
                    seg_idx = random.choice(list(segs_in_data))
                else:
                    seg_idx = -1

            if len(seg_data) == 0:
                for sample in env_data:
                    if sample["metadata"]["seg_idx"] == seg_idx:
                        seg_data.append(sample)
        return seg_data

    def build_sample(self, idx, seg_data, blur=False):
        dataset_name, env_id, seg_idx = self.sample_ids[idx]
        # I get a lot of Nones here in RL training because the dataset index is created based on different data than available!
        # TODO: in RL training, treat entire environment as a single segment and don't distinguish.
        # How? Check above
        if len(seg_data) < self.min_seg_len:
            print(f"   None reason: len:{len(seg_data)} in {dataset_name}, env:{env_id}, seg:{seg_idx}")
            return None

        if len(seg_data) > self.traj_len:
            seg_data = seg_data[:self.traj_len]

        seg_idx = seg_data[0]["metadata"]["seg_idx"]
        set_idx = seg_data[0]["metadata"]["set_idx"]
        env_id = seg_data[0]["metadata"]["env_id"]
        instr = get_instruction_segment(env_id, set_idx, seg_idx, all_instr=self.all_instr)
        if instr is None and dataset_name != "real":
            # print(f"{dataset_name} Seg {env_id}:{set_idx}:{seg_idx} not present in instruction data")
            return None

        instr = get_instruction_segment(env_id, set_idx, seg_idx, all_instr=self.all_instr_full)
        if instr is None:
            print(f"{dataset_name} Seg {env_id}:{set_idx}:{seg_idx} not present in FULL instruction data. WTF?")
            return None

        # Convert to tensors, replacing Nones with zero's
        images_in = [seg_data[i]["state"].image if i < len(seg_data) else None for i in range(len(seg_data))]
        states = [seg_data[i]["state"].state if i < len(seg_data) else None for i in range(len(seg_data))]

        # Blur scene images if requested
        if blur:
            for i in range(len(images_in)):
                if images_in[i] is None:
                    continue
                images_in[i] = iml.eval_augment_query_image(images_in[i])

        images_np = standardize_images(images_in)
        images = none_padded_seq_to_tensor(images_np)

        states = none_padded_seq_to_tensor(states)

        actions = [s["ref_action"] for s in seg_data]
        actions = none_padded_seq_to_tensor(actions)
        stops = [1.0 if s["done"] else 0.0 for s in seg_data]

        # e.g. [1 1 1 1 1 1 0 0 0 0 .. 0] for segment with 6 samples
        mask = [1.0 if s["ref_action"] is not None else 0.0 for s in seg_data]

        stops = torch.FloatTensor(stops)
        mask = torch.FloatTensor(mask)

        # This is a list, converted to tensor in collate_fn
        tok_instructions = [
            tokenize_instruction(s["instruction"], self.word2token) if s["instruction"] is not None else None for s in
            seg_data]
        nl_instructions = [s["instruction"] if s["instruction"] is not None else None for s in seg_data]

        md = [seg_data[i]["metadata"] for i in range(len(seg_data))]
        flag = md[0]["flag"] if "flag" in md[0] else None

        data = {
            "instr_nl": nl_instructions,
            "instr": tok_instructions,
            # "tok_chunks": tok_chunks,
            # "chunk_lengths": chunk_lengths,
            "images": images,
            # "depth_images": depth_images,
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

    def __getitem__(self, idx):
        # Sometimes in background workers, the parameters won't be carried along during spawn
        if P.get_current_parameters() is None:
            P.initialize_experiment()
        self.prof.tick("out")
        seg_data = self.get_seg_data(idx)
        example = self.build_sample(idx, seg_data)
        return example

    def split_into_segments(self, env_list, dname):
        envs = []
        segs = []
        sk = 0
        skipenv = 0
        for env_id in env_list:
            # If we only allow certain envs from a dataset, and this env is not allowed, skip it
            # (Intended use is to train Stage1 with limited real-world data and compare)
            if dname in self.dataset_restricted_envs:
                if env_id not in self.dataset_restricted_envs[dname]:
                    skipenv += 1
                    continue
            # 0th instr set
            instruction_set = self.all_instr[env_id][0]["instructions"]
            for seg in instruction_set:
                seg_idx = seg["seg_idx"]
                if DEBUG: print(f"For env {env_id} including segment: {seg_idx}")
                envs.append(env_id)
                segs.append(seg_idx)
        print(f"Skipped {sk} segments due to merge_len constraints from dataset: {dname}")
        print(f"Skipped {skipenv} environments due to restriction on dataset: {dname}")
        print(f"  kept {len(segs)} segments")

        if self.index_by_env:
            envs = list(sorted(set(envs)))
            segs = ["random" for _ in envs]
            print(f"Switched to indexing by {len(envs)} environments")

        #envs, segs = self.filter_segment_availability(dname, envs, segs)
        return envs, segs

    def filter_segment_availability(self, dname, envs, segs):
        data_env = None
        envs_out, segs_out = [], []
        # TODO: When saving envs, also save metadata for which segments are present
        for env_id, seg_id in zip(envs, segs):
                md = load_single_env_metadata_from_dataset(dname, env_id, self.dataset_prefix)
                if md is None or seg_id in md["seg_ids"]:
                    envs_out.append(env_id)
                    segs_out.append(seg_id)
                else:
                    print(f"Env {env_id} doesn't have seg {seg_id}")
        return envs_out, segs_out

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
                            ["images", "states", "actions", "stops", "masks"] + self.stackable_names)

        try:
            instructions_t, instruction_lengths = instruction_sequence_batch_to_tensor(data_batch["instr"])
        except Exception as e:
            print("ding")

        data_t["instr"] = instructions_t
        data_t["instr_len"] = instruction_lengths
        #data_t["cam_pos"] = data_t["states"][:, 9:12].clone()
        #data_t["cam_rot"] = data_t["states"][:, 12:16].clone()

        self.prof.tick("collate")
        self.prof.loop()
        self.prof.print_stats(5)
        return data_t

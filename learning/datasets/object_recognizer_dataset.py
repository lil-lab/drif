import cv2
import math
import numpy as np
import torch
import random
import itertools
from torch.autograd import Variable
from torch.utils.data import Dataset
from env_config.definitions.landmarks import get_landmark_name_to_index, get_landmark_index_to_name
from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map, load_landmark_noun_chunks_by_name
from data_io.tokenization import bert_tokenize_instruction
from data_io.env import load_env_config
from learning.inputs.sequence import text_sequence_batch_to_tensor, instruction_sequence_batch_to_tensor, text_batch_to_tensor
from learning.inputs.partial_2d_distribution import Partial2DDistribution
from learning.datasets.segment_dataset_simple import SegmentDataset

from utils.dict_tools import dict_zip, dict_map

PROFILE = False
DEBUG = False


class ObjectRecognizerDataset(Dataset):
    def __init__(self,
                 env_list=None,
                 dataset_names=["simulator"],
                 dataset_prefix="supervised",
                 domain="sim",
                 max_traj_length=None,
                 aux_provider_names=[],
                 segment_level=False,
                 cache=False,
                 eval=False):
        """
        Dataset for object recognizer
        :param env_list: if data is to be loaded by the dataset, this is the list of environments for which to include data
        :param dataset_names: list of datasets from which to load data
        :param dataset_prefix: name of the dataset. Default: supervised will use data collected with collect_supervised_data
        :param max_traj_length: truncate trajectories to this long
        :param cuda:
        :param aux_provider_names:
        """
        self.segment_dataset = SegmentDataset(
            env_list=env_list,
            dataset_names=dataset_names,
            dataset_prefix=dataset_prefix,
            domain=domain,
            max_traj_length=max_traj_length,
            aux_provider_names=aux_provider_names,
            segment_level=segment_level,
            cache=cache
        )

        # These are extracted from training data
        self.landmark_chunks = load_landmark_noun_chunks_by_name()
        self.lm_name_to_idx = get_landmark_name_to_index(add_empty=True)
        self.lm_idx_to_name = get_landmark_index_to_name(add_empty=True)
        self.all_landmark_indices = list(self.lm_idx_to_name.keys())

        train_instr_full, dev_instr_full, test_instr_full, corpus = get_all_instructions(ignore_min_augment_len=True)
        self.all_instr_full = {**train_instr_full, **dev_instr_full, **test_instr_full}
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        # In eval mode we only use chunks from instructions, not random chunks!
        self.eval_mode = eval

    def __len__(self):
        return len(self.segment_dataset)

    def __getitem__(self, idx):
        """
        Returns a dataset that iterates over every image in the observation, over every landmark in the image.
        Each landmark is paired with query string and a mask indicating its location.
        :param idx:
        :return:
        """
        data = self.segment_dataset[idx]
        if data is None:
            return None

        data_out = {
            "images": [],
            "states": [],
            "tok_chunks": [],
            "obj_masks": [],
            "lm_indices": [],
            "lm_pos_fpv": [],
            "lm_pos_map": [],
            "md": []
        }

        seq_len = len(data["images"])
        env_config = load_env_config(data["md"][0]["env_id"])
        env_landmark_names = env_config["landmarkName"]
        env_landmark_indices = [self.lm_name_to_idx[n] for n in env_landmark_names]

        for i in range(seq_len):
            lm_indices = data["lm_indices"][i]

            if lm_indices is None:
                lm_indices = list()
            else:
                lm_indices = [l.item() for l in lm_indices]

            for lm_idx in env_landmark_indices:
                lm_name = self.lm_idx_to_name[lm_idx]
                if lm_name not in self.landmark_chunks:# or lm_name == "0Null":
                    continue
                lm_noun_chunks = self.landmark_chunks[lm_name]

                # If this landmark is visible in frame, and it's not the null landmark
                if lm_idx in lm_indices and lm_name != "0Null":
                    l = lm_indices.index(lm_idx)
                    lm_pos_map = data["lm_pos_map"][i][l]
                    lm_pos_fpv = data["lm_pos_fpv"][i][l]
                # If the landmark is absent or refers to the null landmark
                else:
                    # Skip some of these examples to re-balance the data
                    flip = random.uniform(0, 1)
                    if flip > 0.2:
                        continue
                    lm_pos_map = None
                    lm_pos_fpv = None

                random_chunk = random.sample(lm_noun_chunks, 1)[0]
                tokenized_chunk = tokenize_instruction(random_chunk, self.word2token)
                #tokenized_chunk = bert_tokenize_instruction(random_chunk)

                mask = self.generate_faux_object_mask(data["images"][i], data["states"][i], lm_pos_map, lm_pos_fpv)
                outer_prob = torch.tensor([1 - mask.sum().item()])
                obj_mask = Partial2DDistribution(mask, outer_prob)

                data_out["images"].append(data["images"][i])
                data_out["states"].append(data["states"][i])
                data_out["lm_indices"].append(data["lm_indices"][i])
                data_out["lm_pos_fpv"].append(data["lm_pos_fpv"][i])
                data_out["lm_pos_map"].append(data["lm_pos_map"][i])
                data_out["obj_masks"].append(obj_mask)
                data_out["tok_chunks"].append(tokenized_chunk)
                data_out["md"].append(data["md"][i])

        if len(data_out["images"]) == 0:
            return None
        return data_out

    def generate_faux_object_mask(self, image, drone_state, lm_pos_map, lm_pos_fpv):
        height = image.shape[1]
        width = image.shape[2]
        mask = np.zeros((height, width), dtype=np.float32)
        drone_pos = drone_state[9:12]
        drone_pos_2d = drone_pos[:2]

        if lm_pos_map is not None and lm_pos_fpv is not None:
            dst_to_lm = torch.sqrt(torch.sum((drone_pos_2d - lm_pos_map) ** 2)).item()

            # TODO: grab from params
            object_radius_m = 0.1
            image_fov = 84
            image_half_fov = image_fov / 2
            obj_size_rad = math.atan2(object_radius_m, dst_to_lm)
            obj_size_deg = math.degrees(obj_size_rad)
            obj_size_frac = obj_size_deg / image_half_fov
            obj_radius_px = obj_size_frac * width

            cv2.circle(mask, (int(lm_pos_fpv[1].item()), int(lm_pos_fpv[0].item())), int(obj_radius_px), (1.0, 1.0, 1.0), -1)

            if False:
                bw_image = image.numpy().transpose((1, 2, 0))
                bw_image = np.mean(bw_image, keepdims=True, axis=2)
                bw_image = np.tile(bw_image, (1, 1, 3))
                bw_image -= np.min(bw_image)
                bw_image /= (np.max(bw_image) + 1e-9)
                bw_image *= 0.5
                bw_image[:, :, 0] += (mask * 0.5)

                cv2.imshow("landmark_loc", bw_image)
                cv2.waitKey(0)

        # Make this a probability distribution
        mask /= (np.sum(mask) + 1e-9)
        mask_t = torch.from_numpy(mask[np.newaxis, :, :])
        return mask_t

    def stack_tensors(self, one):
        if one is None:
            return None
        one = torch.stack(one, dim=0)
        return one

    def join_lists(self, list_of_lists):
        return list(itertools.chain.from_iterable(list_of_lists))

    def collate_fn(self, list_of_samples):
        # None indicates some sort of error - skip these items
        if list_of_samples is None or None in list_of_samples or len(list_of_samples) == 0:
            return None
        # If there were no observed landmarks during this execution, skip this too.
        for sample in list_of_samples:
            if len(sample) == 0:
                return None

        data_batch = dict_zip(list_of_samples)
        data_batch = dict_map(data_batch, self.join_lists)
        data_t = dict_map(data_batch, self.stack_tensors, ["images", "states"])

        data_t["obj_masks"] = Partial2DDistribution.stack(data_t["obj_masks"])
        chunks_t, chunk_lengths = text_batch_to_tensor(data_batch["tok_chunks"])

        data_t["tok_chunks"] = chunks_t
        data_t["chunk_len"] = chunk_lengths
        data_t["cam_pos"] = data_t["states"][:, 9:12].clone()
        data_t["cam_rot"] = data_t["states"][:, 12:16].clone()
        return data_t

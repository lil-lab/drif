import random
import torch
from utils.dict_tools import dict_zip
from data_io.tokenization import bert_tokenize_instruction
from torch.utils.data import Dataset
from env_config.definitions.landmarks import get_landmark_name_to_index


class AlignmentDataset(Dataset):
    def __init__(self, alignment_data):
        self.data = alignment_data
        self.lm_name_to_idx = get_landmark_name_to_index(add_empty=True)

    def shuffle(self):
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        landmark_indices = []
        tokenized_chunks = []
        raw_chunks = []
        for chunk in example["chunks"]:
            tokenized_chunk = bert_tokenize_instruction(chunk)
            tokenized_chunks.append(tokenized_chunk)
            raw_chunks.append(chunk)
        for landmark in example["landmarks"]:
            landmark_idx = self.lm_name_to_idx[landmark]
            landmark_indices.append(landmark_idx)
        # Always add the 0Null landmark to make sure that there's somewhere to ground non-informative chunks!
        landmark_indices.append(0)
        return {"landmarks": landmark_indices, "chunks": tokenized_chunks, "raw_chunks": raw_chunks}

    def collate_fn(self, list_of_samples):
        batch = dict_zip(list_of_samples)
        # TODO: support batches
        chunks = batch["chunks"][0]
        landmarks = batch["landmarks"][0]
        if len(chunks) == 0 or len(landmarks) == 0:
            return None
        chunk_lengths = [len(chunk) for chunk in chunks]
        max_chunk_length = max(chunk_lengths)

        # Pad all the chunks to the maximum length
        padded_chunks = [chunk + [0]*(max_chunk_length-len(chunk)) for chunk in chunks]
        chunks_t = torch.tensor(padded_chunks)
        landmarks_t = torch.tensor(landmarks)
        chunk_lenghts_t = torch.tensor(chunk_lengths)
        return {"landmarks": landmarks_t, "chunks": chunks_t, "chunk_lengths": chunk_lenghts_t}
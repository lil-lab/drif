import sys
import torch
import torch.optim as optim

from data_io.paths import get_logging_dir
from data_io.parsing import load_chunk_landmark_sets
from data_io.model_io import save_pytorch_model, load_pytorch_model
from env_config.definitions.landmarks import get_landmark_name_to_index, get_landmark_index_to_name

from utils.logging_summary_writer import LoggingSummaryWriter
from learning.alignment.chunk_landmark_model import ChunkLandmarkModel
from learning.alignment.chunk_landmark_generative_model import ChunkLandmarkModelG
from learning.alignment.alignment_dataset import AlignmentDataset

import parameters.parameter_server as P

PRINT_THIS_MANY = 10


def investigate_ml_ibm2_alignments():
    P.initialize_experiment()
    run_name = P.get_current_parameters()["Setup"]["run_name"]
    chunk_landmark_sets = load_chunk_landmark_sets(split="dev")

    dataset = AlignmentDataset(chunk_landmark_sets)
    dataset.shuffle()

    model_fname = f"ibm_model_2_train_object_recognizer_cond_lr.001_minp.001_FIX"
    model = ChunkLandmarkModelG(num_landmarks=64,
                                word_embedding_size=32,
                                embed_size=64,
                                lstm_layers=1,
                                dropout=0.0)
    model.eval()
    load_pytorch_model(model, model_fname)

    lm_name_to_idx = get_landmark_name_to_index(add_empty=True)
    lm_idx_to_name = get_landmark_index_to_name(add_empty=True)

    ord = 0
    batch_count = 0
    for example in dataset:
        batch = dataset.collate_fn([example])
        if batch is None:
            continue
        batch_count += 1
        ord += 1

        raw_chunks = example['raw_chunks']
        landmarks = batch["landmarks"]
        chunks = batch["chunks"]
        chunk_lengths = batch["chunk_lengths"]

        top5_landmarks, top5_probs = model.get_most_likely_landmark_groundings(chunks, chunk_lengths)

        for raw_chunk, lmidx in zip(raw_chunks, top5_landmarks):
            print(f"Chunk: \"{raw_chunk}\" top-5 landmarks: ")
            strout = " ".join([lm_idx_to_name[idx.item()] for idx in lmidx])
            print(strout)
            print('-------------------------------------------------------')
        if ord > PRINT_THIS_MANY:
            break


if __name__ == "__main__":
    investigate_ml_ibm2_alignments()
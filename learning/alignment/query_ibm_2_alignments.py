import sys
import torch
import torch.optim as optim

from data_io.paths import get_logging_dir
from data_io.parsing import load_chunk_landmark_sets
from data_io.tokenization import bert_tokenize_instruction
from data_io.model_io import save_pytorch_model, load_pytorch_model
from env_config.definitions.landmarks import get_landmark_name_to_index, get_landmark_index_to_name

from utils.logging_summary_writer import LoggingSummaryWriter
from learning.alignment.chunk_landmark_model import ChunkLandmarkModel
from learning.alignment.chunk_landmark_generative_model import ChunkLandmarkModelG
from learning.alignment.alignment_dataset import AlignmentDataset

import parameters.parameter_server as P

PRINT_THIS_MANY = 10


def query_ml_ibm2_alignments():
    P.initialize_experiment()
    run_name = P.get_current_parameters()["Setup"]["run_name"]
    chunk_landmark_sets = load_chunk_landmark_sets()
    writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}")

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

    while True:
        chunk = input("Input noun chunk >>")

        tok_chunk = bert_tokenize_instruction(chunk)
        tok_chunk_len = len(tok_chunk)

        tok_chunks = torch.tensor([tok_chunk])
        tok_chunk_lengths = torch.tensor([tok_chunk_len])

        top5_landmarks, probs = model.get_most_likely_landmark_groundings(tok_chunks, tok_chunk_lengths, n=64)

        entropy = -torch.sum(probs * torch.log(probs))

        lmsout = " ".join([lm_idx_to_name[idx.item()] for idx in top5_landmarks[0]])
        probsout = " ".join(["{:.4f}".format(prob.item()) for prob in probs[0]])
        print(f"Top5 Landmarks: {lmsout}")
        print(f" Probabilities: {probsout}")
        print(f"       Entropy: {entropy.item()}")


if __name__ == "__main__":
    query_ml_ibm2_alignments()
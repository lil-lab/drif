import torch

from data_io.parsing import load_chunk_landmark_sets
from data_io.instructions import save_chunk_landmark_alignments
from data_io.model_io import load_pytorch_model
from env_config.definitions.landmarks import get_landmark_index_to_name, NUM_LANDMARKS

from learning.alignment.chunk_landmark_generative_model import ChunkLandmarkModelG
from learning.alignment.alignment_dataset import AlignmentDataset

import parameters.parameter_server as P

ENTROPY_THRESHOLD = 2.5
NULL_PROB_THRESHOLD = 0.05
LANDMARK_PROB_THRESHOLD = 0.15

NUM_EXAMPLES = -1


def investigate_ml_ibm2_alignments():
    P.initialize_experiment()
    run_name = P.get_current_parameters()["Setup"]["run_name"]
    datasplit = "dev"
    #datasplit = "train"
    chunk_landmark_sets = load_chunk_landmark_sets(split=datasplit)

    dataset = AlignmentDataset(chunk_landmark_sets)
    dataset.shuffle()

    model_fname = f"alignments/ibm_model_2_data_collect_jan20_oracle_empty_real"
    model = ChunkLandmarkModelG(num_landmarks=NUM_LANDMARKS,
                                word_embedding_size=32,
                                embed_size=64,
                                lstm_layers=1,
                                dropout=0.0)
    model.eval()
    load_pytorch_model(model, model_fname)

    lm_idx_to_name = get_landmark_index_to_name(add_empty=True)

    noun_chunk_landmark_alignments = {}

    ord = 0
    batch_count = 0
    for example in dataset:
        batch = dataset.collate_fn([example])
        if batch is None:
            continue
        batch_count += 1
        ord += 1
        print(f"Processing example: {ord} / {len(dataset)}")

        raw_chunks = example['raw_chunks']
        chunks = batch["chunks"]
        chunk_lengths = batch["chunk_lengths"]

        landmark_indices, probs = model.get_most_likely_landmark_groundings(chunks, chunk_lengths, n=NUM_LANDMARKS)

        for chunk_idx in range(len(chunks)):
            # Zero index corresponds to Null landmark
            chunk_landmark_indices = [l.item() for l in landmark_indices[chunk_idx]]
            chunk_landmark_probs = probs[chunk_idx]
            entropy = -torch.sum(chunk_landmark_probs * torch.log(chunk_landmark_probs))
            null_ord = chunk_landmark_indices.index(0)
            null_prob = chunk_landmark_probs[null_ord].item()

            # If conditions are met, emit a noun-landmark alignment
            if entropy < ENTROPY_THRESHOLD:
                if null_prob < NULL_PROB_THRESHOLD:
                    chunk_str = raw_chunks[chunk_idx]
                    if chunk_str not in noun_chunk_landmark_alignments:
                        noun_chunk_landmark_alignments[chunk_str] = {"landmarks": []}
                    for i, lm_idx in enumerate(chunk_landmark_indices):
                        if chunk_landmark_probs[i].item() > LANDMARK_PROB_THRESHOLD:
                            lm_name = lm_idx_to_name[lm_idx]
                            if lm_name not in noun_chunk_landmark_alignments[chunk_str]["landmarks"]:
                                noun_chunk_landmark_alignments[chunk_str]["landmarks"].append(lm_name)
                                print(f"               {chunk_str} : {lm_name}")
                        else:
                            break
                elif null_prob > 0.5:
                    chunk_str = raw_chunks[chunk_idx]
                    if chunk_str not in noun_chunk_landmark_alignments:
                        noun_chunk_landmark_alignments[chunk_str] = {"landmarks": []}
                    lm_name = lm_idx_to_name[0]
                    if lm_name not in noun_chunk_landmark_alignments[chunk_str]["landmarks"]:
                        noun_chunk_landmark_alignments[chunk_str]["landmarks"].append(lm_name)

        if ord > NUM_EXAMPLES > 0:
            break

    save_chunk_landmark_alignments(noun_chunk_landmark_alignments, split=datasplit)


if __name__ == "__main__":
    investigate_ml_ibm2_alignments()

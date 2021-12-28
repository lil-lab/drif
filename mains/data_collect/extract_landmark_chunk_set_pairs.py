import spacy
import numpy as np

from env_config.definitions.landmarks import get_landmark_names, get_landmark_name_to_index
from data_io.instructions import \
    get_all_instructions, load_english_vocabulary, clean_instruction, \
    split_instruction, get_word_to_token_map, save_chunk_landmark_alignments, words_to_terms, load_noun_chunk_corpus_and_frequencies
from data_io.parsing import remap_noun_chunk, save_chunk_landmark_sets
from data_io.env import load_path, load_env_config
from data_io.tokenization import bert_tokenize_instruction

import parameters.parameter_server as P

CLOSE_LM_THRES = 300
MIN_CHUNK_OCCURENCES = 1
MUTUAL_INFO_THRESHOLD = 0.0009
MAX_CHUNK_PROB = 0.025

UNK_TERM = "NA"


def close_landmark_names(env_config, path, start_idx, end_idx):
    close_landmarks = []
    for step in path[start_idx:end_idx]:
        for i, lm_name in enumerate(env_config["landmarkName"]):
            lm_x = env_config["xPos"][i]
            lm_y = env_config["zPos"][i]
            lm_pos = np.asarray([lm_x, lm_y])
            dst = np.linalg.norm(lm_pos - step)
            if dst < CLOSE_LM_THRES and lm_name not in close_landmarks:
                close_landmarks.append(lm_name)
    return close_landmarks


def extract_chunk_landmark_sets(all_landmark_names, train_instructions):
    # the clustered corpus is a dictionary of lists, where the keys are valid english words and the values are
    # lists of words found in the corpus that are assumed to be misspellings of the key valid words

    # We make the distinction that a word is any word in an instruction
    # Terms are words in the english vocabulary. Multiple words (misspellings) can map to a single term.
    #nlp = spacy.load("en_core_web_lg")

    num_landmarks = len(all_landmark_names)
    landmark_name_to_idx = get_landmark_name_to_index()

    landmark_chunk_sets = []

    bert_token_corpus = set()

    # Count landmark and word occurences and co-occurences
    env_ord = 0
    for env_id, instruction_sets in train_instructions.items():
        env_ord += 1
        print(f"Extracting from env: {env_ord}/{len(train_instructions)}")

        path = load_path(env_id)
        env_config = load_env_config(env_id)
        for instruction_set in instruction_sets:
            for instruction_segment in instruction_set["instructions"]:
                instruction_str = instruction_segment["instruction"]
                start_idx = instruction_segment["start_idx"]
                end_idx = instruction_segment["end_idx"]

                # Get all present chunks and augment with versions that remove "the", "a" etc.
                base_present_chunks = set(instruction_segment["noun_chunks"])
                aug_present_chunks = set([remap_noun_chunk(c) for c in base_present_chunks])
                present_chunks = base_present_chunks.union(aug_present_chunks)
                present_landmarks = close_landmark_names(env_config, path, start_idx, end_idx)
                present_lm_indices = [landmark_name_to_idx[lm] for lm in present_landmarks]

                bert_tokenized_chunks = []
                for chunk in present_chunks:
                    bert_tok = bert_tokenize_instruction(chunk)
                    bert_token_corpus = bert_token_corpus.union(set(bert_tok))
                    bert_tokenized_chunks.append(bert_tok)

                example = {
                    "landmarks": list(present_landmarks),
                    "chunks": list(present_chunks),
                    "bert_tokenized_chunks_raw": bert_tokenized_chunks,
                    "env_id": env_id,
                    "set_idx": 0,
                    "seg_idx": instruction_segment["seg_idx"]
                }
                landmark_chunk_sets.append(example)

    return landmark_chunk_sets


def extract_landmark_chunk_set_pairs():
    P.initialize_experiment()
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    _, word2token = get_word_to_token_map(corpus, use_thesaurus=False)
    all_landmark_names = get_landmark_names()

    chunk_landmark_sets = extract_chunk_landmark_sets(all_landmark_names, train_instr)
    save_chunk_landmark_sets(chunk_landmark_sets, split="train")

    chunk_landmark_sets_dev = extract_chunk_landmark_sets(all_landmark_names, dev_instr)
    save_chunk_landmark_sets(chunk_landmark_sets_dev, split="dev")

    # chunk_landmark_sets is a list of examples.
    # Each example is a pair of two sets. The pair is represented by a dict with two keys, "chunks" and "landmarks".
    # The first set is the landmark names close to the trajectory in that example.
    # The second set is the noun chunks extracted from the instruction from that example.


if __name__ == "__main__":
    extract_landmark_chunk_set_pairs()

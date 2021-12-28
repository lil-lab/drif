import spacy
import numpy as np

from env_config.definitions.landmarks import get_landmark_names, get_landmark_name_to_index
from data_io.instructions import \
    get_all_instructions, load_english_vocabulary, clean_instruction, \
    split_instruction, get_word_to_token_map, save_chunk_landmark_alignments, words_to_terms, load_noun_chunk_corpus_and_frequencies
from data_io.parsing import remap_noun_chunk
from data_io.env import load_path, load_env_config

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


def align_chunks(all_chunks, all_landmark_names, train_instructions):
    # the clustered corpus is a dictionary of lists, where the keys are valid english words and the values are
    # lists of words found in the corpus that are assumed to be misspellings of the key valid words

    # We make the distinction that a word is any word in an instruction
    # Terms are words in the english vocabulary. Multiple words (misspellings) can map to a single term.

    num_chunks = len(all_chunks)
    num_landmarks = len(all_landmark_names)

    # Calculate the mutual information between each cluster and each landmark
    # Number of times each term appears in an instruction
    chunk_occurences = np.zeros(num_chunks)
    # Number of times each landmark appears near a segment path
    landmark_occurences = np.zeros(num_landmarks)
    # The number of times each term and landmark combination appears in the instruction and near the path
    chunk_landmark_cooccurences = np.zeros((num_chunks, num_landmarks))
    # The number of total segments that were considered
    total_occurences = 0

    landmark_indices = get_landmark_name_to_index()

    chunk2id = {c: i for i, c in enumerate(all_chunks)}

    # Count landmark and word occurences and co-occurences
    env_ord = 0
    for env_id, instruction_sets in train_instructions.items():
        env_ord += 1
        print(f"Collecting stats from env: {env_ord}/{len(train_instructions)}")

        path = load_path(env_id)
        env_config = load_env_config(env_id)
        for instruction_set in instruction_sets[0]["instructions"]:
            instruction_str = instruction_set["instruction"]
            start_idx = instruction_set["start_idx"]
            end_idx = instruction_set["end_idx"]

            #doc = nlp(instruction_str)
            #present_chunks = [str(chunk) for chunk in doc.noun_chunks]
            present_chunks = instruction_set["noun_chunks"]
            present_chunks = [remap_noun_chunk(c) for c in present_chunks]

            present_landmarks = close_landmark_names(env_config, path, start_idx, end_idx)
            present_lm_indices = [landmark_indices[lm] for lm in present_landmarks]
            for chunk in present_chunks:
                chunk_id = chunk2id[chunk]
                chunk_occurences[chunk_id] += 1
            for lm_idx in present_lm_indices:
                landmark_occurences[lm_idx] += 1
                for chunk in present_chunks:
                    chunk_id = chunk2id[chunk]
                    chunk_landmark_cooccurences[chunk_id, lm_idx] += 1
            total_occurences += 1

    chunk_prob = np.expand_dims(chunk_occurences / total_occurences, 1).repeat(num_landmarks, 1)
    landmark_prob = np.expand_dims(landmark_occurences / total_occurences, 0).repeat(num_chunks, 0)
    chunk_and_landmark_prob = chunk_landmark_cooccurences / total_occurences

    # term_and_landmark_prob has dimensions 0: terms, 1: landmarks
    mutual_info_factor = chunk_and_landmark_prob / (landmark_prob * chunk_prob + 1e-27)
    #mutual_info_factor = term_and_landmark_prob / ((1 / num_landmarks) * term_prob + 1e-9)
    mutual_info = chunk_and_landmark_prob * np.log(mutual_info_factor + 1e-27)

    # The above line is the correct formula for mutual information. For our case, below formula might be better?
    # The mutual information is higher for common words than uncommon ones. We might prefer the opposite effect.
    # On the other hand, uncommon words are more likely to spuriously correlate with landmarks, which will cause a
    # less reliable corpus.
    #mutual_info = np.log(mutual_info_factor + 1e-27)

    # Ground each term and produce the alignments
    chunk_alignments = {}
    common_chunks = []
    rare_chunks = []

    for i in range(num_chunks):
        grounded_lm_indices = [idx for idx in range(num_landmarks) if mutual_info[i][idx] > MUTUAL_INFO_THRESHOLD]

        grounded_lm_names = [all_landmark_names[idx] for idx in grounded_lm_indices]
        mutual_infos = np.asarray([mutual_info[i][idx] for idx in grounded_lm_indices])

        args = list(np.argsort(mutual_infos))
        grounded_lm_names = list(reversed([grounded_lm_names[idx] for idx in args]))
        mutual_infos = list(reversed([mutual_infos[idx] for idx in args]))

        remark = "ok"

        # If the word is too common to be referring to a landmark, ignore ita
        this_chunk_prob = chunk_prob[i][0]
        if this_chunk_prob > MAX_CHUNK_PROB:
            common_chunks.append(all_chunks[i])
            grounded_lm_names = []
            mutual_infos = []
            remark = "chunk too frequent"

        if chunk_occurences[i] < MIN_CHUNK_OCCURENCES:
            rare_chunks.append(all_chunks[i])
            grounded_lm_names = []
            mutual_infos = []
            remark = "chunk too rare"

        chunk_alignments[all_chunks[i]] = \
            {
                "landmarks": grounded_lm_names,
                "mutual_info": mutual_infos,
                "chunk_prob": this_chunk_prob,
                "remark": remark
            }

    for k in chunk_alignments.keys():
        if len(chunk_alignments[k]["landmarks"]) > 0:
            print(k, chunk_alignments[k])

    print("Ignored groundings for these common words: " + str(common_chunks))
    return chunk_alignments


def generate_thesaurus():
    P.initialize_experiment()
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    _, word2token = get_word_to_token_map(corpus, use_thesaurus=False)

    all_noun_chunks, chunk_frequencies = load_noun_chunk_corpus_and_frequencies()
    all_landmark_names = get_landmark_names()

    chunk_alignments = align_chunks(all_noun_chunks, all_landmark_names, train_instr)

    print("-------------------------------------------------------------------")
    print(" Chunks for each landmark:")
    print("-------------------------------------------------------------------")
    lm_to_chunks = {}
    for chunk, results in chunk_alignments.items():
        for landmark in results["landmarks"]:
            if landmark not in lm_to_chunks:
                lm_to_chunks[landmark] = []
            lm_to_chunks[landmark].append(chunk)

    for landmark, chunks in lm_to_chunks.items():
        print(f"{landmark} : {chunks}")

    save_chunk_landmark_alignments(chunk_alignments)


if __name__ == "__main__":
    generate_thesaurus()
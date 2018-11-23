import numpy as np

from env_config.definitions.landmarks import get_landmark_names, get_landmark_name_to_index
from data_io.instructions import \
    get_all_instructions, load_english_vocabulary, clean_instruction, \
    split_instruction, get_word_to_token_map, save_thesaurus, words_to_terms
from data_io.env import load_path, load_env_config

CLOSE_LM_THRES = 300
MIN_TERM_OCCURENCES = 10
MUTUAL_INFO_THRESHOLD = 0.008
MAX_TERM_PROB = 0.1

UNK_TERM = "NA"

"""
This file is used to generate the alignments between landmarks and words in the corpus.
Also, multiple words are mapped to the same "term". All likely misspellings of the term map to it.
This way "gorilla" and "gorila" both map to "gorilla".
The alignment is then found between the term "gorilla" and the corresponding landmark.
"""


# Computes the levenshtein edit distance between two words.
# Words with very small edit distance are most likely typos
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_adjacency_mat(corpus):
    vocab_size = len(corpus)
    adjacency_mat = np.zeros((vocab_size, vocab_size))
    for i, word1 in enumerate(corpus):
        for j, word2 in enumerate(corpus):
            adjacency_mat[i][j] = levenshtein(word1, word2)
    return adjacency_mat


def cluster_corpus(corpus, train_instructions, max_edit_distance=3):
    english = load_english_vocabulary()
    terms = {}
    potential_misspellings = []

    # Count the number of times each word occurs in the corpus
    word_counts = {}
    for word in corpus:
        word_counts[word] = 0
    for env_id, instruction_sets in train_instructions.items():
        for instruction_set in instruction_sets[0]["instructions"]:
            instruction_str = instruction_set["instruction"]
            instr_split = split_instruction(clean_instruction(instruction_str))

            for word in instr_split:
                word_counts[word] += 1

    for word in corpus:
        if word in english:
            terms[word] = [word]
        else:
            potential_misspellings.append(word)

    terms[UNK_TERM] = []

    # Find the closest english word by edit distance
    edit_dists = np.zeros((len(terms)))
    term_list = sorted(list(terms.keys()))

    unresolved_words = []

    for i, misspelled_word in enumerate(potential_misspellings):

        # Words that contain a numbers should be assumed not to be misspellings
        if any(char.isdigit() for char in misspelled_word):
            unresolved_words.append(misspelled_word)
            continue

        # For other words, see if they might be misspellings of one of the terms
        for j, term in enumerate(term_list):
            edit_dists[j] = levenshtein(misspelled_word, term)
        closest = int(np.argmin(edit_dists))
        min_dist = edit_dists[closest]

        # If the misspelled word is likely a misspelling of the closest term, add it in, except to the "NA" term
        if min_dist <= max_edit_distance and term_list[closest] != UNK_TERM:
            terms[term_list[closest]].append(misspelled_word)
        # Otherwise add it to the list of unresolved words that are too different from every term
        else:
            unresolved_words.append(misspelled_word)

    rejected_words = []

    # Handle words that are not misspellings
    for unresolved_word in unresolved_words:
        # If the word is not a misspelling and is also very infrequent, reject it
        if word_counts[unresolved_word] < MIN_TERM_OCCURENCES:
            rejected_words.append(unresolved_word)
        # Otherwise create a term for this word
        else:
            terms[unresolved_word] = [unresolved_word]

    # For each rejected word, add it to the unknown term
    for rejected_word in rejected_words:
        terms[UNK_TERM].append(rejected_word)

    print("After clustering words, found " + str(len(rejected_words)) + " rare ones that have been rejected:")
    print(rejected_words)
    print("...")

    return terms, rejected_words


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


def ground_terms(word2id, clustered_corpus, landmark_names, train_instructions):
    # the clustered corpus is a dictionary of lists, where the keys are valid english words and the values are
    # lists of words found in the corpus that are assumed to be misspellings of the key valid words

    # We make the distinction that a word is any word in an instruction
    # Terms are words in the english vocabulary. Multiple words (misspellings) can map to a single term.

    num_terms = len(clustered_corpus)
    vocab_size = len(word2id)
    num_landmarks = len(landmark_names)

    # This is gonna be the new word2id, once we start using the thesaurus
    term2id = {}
    id2term = {}
    for i, term in enumerate(sorted(clustered_corpus.keys())):
        term2id[term] = i
        id2term[i] = term

    # Calculate the mutual information between each cluster and each landmark
    # Number of times each term appears in an instruction
    term_occurences = np.zeros(num_terms)
    # Number of times each landmark appears near a segment path
    landmark_occurences = np.zeros(num_landmarks)
    # The number of times each term and landmark combination appears in the instruction and near the path
    term_landmark_cooccurences = np.zeros((num_terms, num_landmarks))
    # The number of total segments that were considered
    total_occurences = 0

    landmark_indices = get_landmark_name_to_index()

    # Inverse the clusters so that we can efficiently map each word in each instruction to it's cluster core
    word2term = {}
    for real_word, misspellings in clustered_corpus.items():
        for misspelling in misspellings:
            word2term[misspelling] = real_word

    # Count landmark and word occurences and co-occurences
    for env_id, instruction_sets in train_instructions.items():
        path = load_path(env_id)
        env_config = load_env_config(env_id)
        for instruction_set in instruction_sets[0]["instructions"]:
            instruction_str = instruction_set["instruction"]
            start_idx = instruction_set["start_idx"]
            end_idx = instruction_set["end_idx"]

            present_landmarks = close_landmark_names(env_config, path, start_idx, end_idx)
            present_lm_indices = [landmark_indices[lm] for lm in present_landmarks]

            mentioned_words = split_instruction(clean_instruction(instruction_str))
            mentioned_terms = words_to_terms(mentioned_words, word2term)

            for term in mentioned_terms:
                term_id = term2id[term]
                term_occurences[term_id] += 1

            for lm_idx in present_lm_indices:
                landmark_occurences[lm_idx] += 1
                for term in mentioned_terms:
                    term_id = term2id[term]
                    term_landmark_cooccurences[term_id][lm_idx] += 1

            total_occurences += 1

    term_prob = np.expand_dims(term_occurences / total_occurences, 1).repeat(num_landmarks, 1)
    landmark_prob = np.expand_dims(landmark_occurences / total_occurences, 0).repeat(num_terms, 0)
    term_and_landmark_prob = term_landmark_cooccurences / total_occurences

    # term_and_landmark_prob has dimensions 0: terms, 1: landmarks
    mutual_info_factor = term_and_landmark_prob / (landmark_prob * term_prob + 1e-27)
    #mutual_info_factor = term_and_landmark_prob / ((1 / num_landmarks) * term_prob + 1e-9)
    mutual_info = term_and_landmark_prob * np.log(mutual_info_factor + 1e-27)

    # The above line is the correct formula for mutual information. For our case, below formula might be better?
    # The mutual information is higher for common words than uncommon ones. We might prefer the opposite effect.
    # On the other hand, uncommon words are more likely to spuriously correlate with landmarks, which will cause a
    # less reliable corpus.
    #mutual_info = np.log(mutual_info_factor + 1e-27)

    # Ground each term and produce the thesaurus
    term_meanings = {}

    common_words = []

    for i in range(num_terms):
        grounded_lm_indices = [idx for idx in range(num_landmarks) if mutual_info[i][idx] > MUTUAL_INFO_THRESHOLD]

        grounded_lm_names = [landmark_names[idx] for idx in grounded_lm_indices]
        mutual_infos = np.asarray([mutual_info[i][idx] for idx in grounded_lm_indices])

        args = list(np.argsort(mutual_infos))
        grounded_lm_names = list(reversed([grounded_lm_names[idx] for idx in args]))
        mutual_infos = list(reversed([mutual_infos[idx] for idx in args]))

        # If the word is too common to be referring to a landmark, ignore ita
        this_term_prob = term_prob[i][0]
        if this_term_prob > MAX_TERM_PROB:
            common_words.append(id2term[i])
            grounded_lm_names = []
            mutual_infos = []

        term_meanings[id2term[i]] = \
            {
                "landmarks": grounded_lm_names,
                "mutual_info": mutual_infos,
                "term_prob": this_term_prob
            }

    for k in term_meanings.keys():
        if len(term_meanings[k]["landmarks"]) > 0:
            print(k, term_meanings[k])

    print ("Ignored groundings for these common words: " + str(common_words))

    return term_meanings, word2term


# Function used only for debugging to see whether a word is present in a dataset
# Some words are only found in test data, but don't actually appear in train data
def findword(word, dataset):
    for env, sets in dataset.items():
        for set in sets[0]["instructions"]:
            if word in set["instruction"]:
                return True
    return False


def generate_thesaurus():
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    _, word2token = get_word_to_token_map(corpus, use_thesaurus=False)

    terms, rejected_words = cluster_corpus(corpus, train_instr)

    landmark_names = get_landmark_names()

    term_groundings, word2term = ground_terms(word2token, terms, landmark_names, train_instr)

    thesaurus = {
        "term2word": terms,
        "word2term": word2term,
        "term_groundings": term_groundings,
        "rejected_words": rejected_words
    }

    save_thesaurus(thesaurus)


if __name__ == "__main__":
    generate_thesaurus()
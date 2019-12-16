from env_config.definitions.nlp_templates import N_LANDMARKS
from data_io.instructions import \
    get_all_instructions, get_word_to_token_map, save_landmark_alignments

from env_config.generation.generate_template_curves import SAMPLING_MODE

from data_io.env import load_config_metadata

import parameters.parameter_server as P

UNK_TERM = "NA"


# Function used only for debugging to see whether a word is present in a dataset
# Some words are only found in test data, but don't actually appear in train data
def findword(word, dataset):
    for env, sets in dataset.items():
        for set in sets[0]["instructions"]:
            if word in set["instruction"]:
                return True
    return False


def get_identity_term_mapping(corpus):
    term2word = {}
    word2term = {}
    for word in corpus:
        term2word[word] = word
        word2term[word] = word
    return term2word, word2term


def referent_list_to_word_list(ref_list):
    all_words = []
    for ref in ref_list:
        ref_split = ref.split(" ")
        for word in ref_split:
            all_words.append(word)
    return all_words


def get_template_term_groundings(corpus, word2term):
    term2lm = {}

    lm_names = load_config_metadata()["all_landmark_names"]

    for lmname in lm_names:
        referents = N_LANDMARKS[lmname]
        if SAMPLING_MODE == "consistent":
            referents = [referents[0]]
        wordlist = referent_list_to_word_list(referents)
        for word in wordlist:
            # This can happen if we generate with a subset of landmarks
            if word not in word2term:
                continue
            term = word2term[word]
            if term not in term2lm:
                term2lm[term] = {"landmarks": []}
            if lmname not in term2lm[term]:
                term2lm[term]["landmarks"].append(lmname)

    for word in corpus:
        term = word2term[word]
        if term not in term2lm:
            term2lm[term] = {"landmarks": []}

    return term2lm


def generate_thesaurus():
    P.initialize_experiment()

    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    _, word2token = get_word_to_token_map(corpus, use_thesaurus=False)

    term2word, word2term = get_identity_term_mapping(corpus)

    term2landmark = get_template_term_groundings(corpus, word2term)

    thesaurus = {
        "term2word": term2word,
        "word2term": word2term,
        "term_groundings": term2landmark,
        "rejected_words": []
    }

    save_landmark_alignments(thesaurus)


if __name__ == "__main__":
    generate_thesaurus()
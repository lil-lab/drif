from data_io import spacy_singleton
import random
import json
from collections import defaultdict

from data_io.paths import get_noun_chunk_corpus_path, get_noun_chunk_frequencies_path
from data_io.models import load_model
from data_io.parsing import remap_noun_chunk

from env_config.definitions.landmarks import get_landmark_names
from data_io.instructions import get_all_instructions, get_word_to_token_map, save_landmark_alignments

import parameters.parameter_server as P

CHUNKER = "spacy"
MIN_FREQUENCY = 1


# Supervised learning parameters
def generate_and_save_chunk_corpus():
    P.initialize_experiment()

    nlp = spacy_singleton.load("en_core_web_lg")
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()

    train_instr = list(train_instr.items())
    random.shuffle(train_instr)

    chunk_frequencies = defaultdict(float)

    tot_chunks = 0
    i = 0
    for env_id, instr_sets in train_instr:
        i += 1
        print(f"{i}/{len(train_instr)}")
        for instr_set in instr_sets:
            if len(instr_set) == 0:
                continue
            for instr_seg in instr_set["instructions"]:
                instr_str = instr_seg["instruction"]
                if CHUNKER == "spacy":
                    doc = nlp(instr_str)
                    chunk_strings = [str(chunk) for chunk in doc.noun_chunks]
                    chunk_strings = [remap_noun_chunk(c) for c in chunk_strings]
                    for chkstr in chunk_strings:
                        chunk_frequencies[chkstr] += 1
                        tot_chunks += 1

    noun_chunk_corpus = list(sorted(chunk_frequencies.keys()))
    noun_chunk_corpus = [n for n in noun_chunk_corpus if chunk_frequencies[n] >= MIN_FREQUENCY]
    chunk_frequency_list = [chunk_frequencies[c] for c in noun_chunk_corpus]

    with open(get_noun_chunk_corpus_path(), "w") as fp:
        json.dump(noun_chunk_corpus, fp)

    with open(get_noun_chunk_frequencies_path(), "w") as fp:
        json.dump(chunk_frequency_list, fp)


if __name__ == "__main__":
    generate_and_save_chunk_corpus()
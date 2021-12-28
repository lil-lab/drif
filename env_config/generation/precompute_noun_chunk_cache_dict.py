import json

from data_io.paths import get_noun_chunk_instruction_cache_path
from data_io import spacy_singleton

from data_io.instructions import get_all_instructions

import parameters.parameter_server as P

MIN_FREQUENCY = 1


# Supervised learning parameters
def generate_noun_chunk_cache():
    P.initialize_experiment()

    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    all_instr = {**train_instr, **dev_instr, **test_instr}

    all_instr = list(all_instr.items())

    chunk_dict = {}

    i = 0
    for env_id, instr_sets in all_instr:
        i += 1
        print(f"{i}/{len(all_instr)}")
        for instr_set in instr_sets:
            if len(instr_set) == 0:
                continue
            for instr_seg in instr_set["instructions"]:
                instr_str = instr_seg["instruction"]
                chunks = spacy_singleton.get_noun_chunks(instr_str)
                chunk_strings = [str(chunk) for chunk in chunks]
                chunk_dict[instr_str] = chunk_strings

    with open(get_noun_chunk_instruction_cache_path(), "w") as fp:
        json.dump(chunk_dict, fp)


if __name__ == "__main__":
    generate_noun_chunk_cache()
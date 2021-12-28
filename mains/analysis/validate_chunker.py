import spacy
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from data_io.models import load_model
from data_io.instructions import get_all_instructions

out_dir = "~/analysis_results/"

import parameters.parameter_server as P

CHUNKER = "spacy"
COUNT = 20
#COUNT = False


def highlight_chunks_html(text, chunks):
    highlight_start_tag = '<span style="background-color: #44AAFF">'
    highlight_end_tag = '</span>'

    # Collect chunks
    tagged_chunks = []
    prev_end = 0
    for i, chunk in enumerate(chunks):
        start = text[prev_end:].find(chunk) + prev_end
        if start > prev_end:
            tagged_chunks.append((text[prev_end:start], False))
        tagged_chunks.append((text[start:start+len(chunk)], True))
        prev_end = start + len(chunk)
    if prev_end < len(text):
        tagged_chunks.append((text[prev_end:], False))

    print(tagged_chunks)

    # Build HTML string
    output = "<p>"
    for chunk in tagged_chunks:
        if chunk[1]:
            output += highlight_start_tag + chunk[0] + highlight_end_tag
        else:
            output += chunk[0]
    output += "</p>"
    output += "<p></p>"
    return output


# Supervised learning parameters
def print_chunker_results():
    P.initialize_experiment()

    nlp = spacy.load("en_core_web_lg")

    setup = P.get_current_parameters()["Setup"]
    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")

    print("Loading data")
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()

    pick_instr = dev_instr
    #pick_instr = train_instr

    pick_instr = list(pick_instr.items())
    random.shuffle(pick_instr)

    chunk_frequencies = defaultdict(float)

    html_doc = "<!DOCTYPE html><html><head>"

    i = 0
    tot_chunks = 0
    for env_id, instr_sets in pick_instr:
        instr_set = instr_sets[0]["instructions"]
        if len(instr_set) == 0:
            continue
        instr_seg = random.sample(instr_set, 1)[0]
        #for instr_seg in instr_set:
        if True:
            instr_str = instr_seg["instruction"]
            if CHUNKER == "spacy":
                doc = nlp(instr_str)
                print("<<<<<<<<< INSTRUCTION >>>>>>>>>>")
                print(instr_str)
                print("................................")
                chunk_strings = [str(chunk) for chunk in doc.noun_chunks]
                html_line = highlight_chunks_html(instr_str, chunk_strings)
                html_doc += html_line
                for chkstr in chunk_strings:
                    chunk_frequencies[chkstr] += 1
                    tot_chunks += 1
        i += 1
        if COUNT and i > COUNT:
            break
    html_doc += "</html></head>"
    os.makedirs(os.path.expanduser(out_dir), exist_ok=True)
    with open(os.path.expanduser(os.path.join(out_dir, "chunking.html")), "w") as fp:
        fp.write(html_doc)

    dist = np.asarray(list(reversed(sorted(chunk_frequencies.values()))))
    print(dist)
    sns.lineplot(range(len(dist)), dist)
    plt.show()

    # How many noun chunks occur only once
    num_ones = 1
    for count in chunk_frequencies.values():
        if count == 1:
            num_ones += 1
    print(f"{num_ones} out of {len(chunk_frequencies)} chunks appear only once ({(100.0 * num_ones) / len(chunk_frequencies)}%)")


if __name__ == "__main__":
    print_chunker_results()

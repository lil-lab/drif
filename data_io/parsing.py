import json
from data_io.paths import get_chunk_landmark_set_path
from data_io.tokenization import OBJ_REF_TOK

def remap_noun_chunk(chunk):
    removable_prefixes = ["the ", "a "]
    removable_suffixes = [" stop"]
    for prefix in removable_prefixes:
        if chunk.startswith(prefix):
            chunk = chunk[len(prefix):]
    for suffix in removable_suffixes:
        if chunk.startswith(suffix):
            chunk = chunk[:len(chunk) - len(suffix)]
    return chunk


def save_chunk_landmark_sets(chunk_landmark_sets, split="train"):
    with open(get_chunk_landmark_set_path(split), "w") as fp:
        json.dump(chunk_landmark_sets, fp, indent=4)


def load_chunk_landmark_sets(split="train"):
    with open(get_chunk_landmark_set_path(split), "r") as fp:
        chklmsts = json.load(fp)
    return chklmsts
"""
Spacy models can take up a significant amount of host memory.
This module ensures that only a single spacy model of a type gets loaded at once per process.
"""

import spacy
import thinc.extra.load_nlp
from data_io.paths import get_noun_chunk_instruction_cache_path
import os
import json

_model_dict = {}

_chunk_cache = {}

# TODO: Implement a variant that uses Ray to share a single model


def load(spacy_model_name, **kwargs):
    global _model_dict
    if spacy_model_name not in _model_dict or len(thinc.extra.load_nlp.VECTORS) == 0:
        print("Loading Spacy")
        _model_dict[spacy_model_name] = spacy.load(spacy_model_name, **kwargs)
    return _model_dict[spacy_model_name]


def get_noun_chunks(text_str):
    global _chunk_cache
    if len(_chunk_cache) == 0:
        cache_path = get_noun_chunk_instruction_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, "r") as fp:
                print("Loading chunk cache")
                _chunk_cache = json.load(fp)
    # If cache contains pre-extracted chunks for this text / instruction string, return those
    if text_str in _chunk_cache:
        return _chunk_cache[text_str]
    # Otherwise load (if not yet loaded) a spacy model and extract the chunks
    else:
        nlp = load("en_core_web_lg")
        doc = nlp(text_str, disable=["ner"])
        noun_chunk_list = list([str(c) for c in doc.noun_chunks])
        return noun_chunk_list

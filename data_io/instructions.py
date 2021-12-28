import json
import os
import string
from collections import defaultdict, OrderedDict
from itertools import chain

import spacy
import torch
import numpy as np
import math

from data_io import spacy_singleton
from data_io.helpers import load_json, save_json
from env_config.definitions.landmarks import get_landmark_name_to_index, get_landmark_names
from env_config.definitions.nlp_templates import has_ambiguous_noun_phrase
from data_io.env import load_path, load_env_config, load_template
import parameters.parameter_server as P
from data_io.paths import get_config_dir, get_instruction_cache_dir, get_english_vocab_path, get_thesaurus_path, \
    get_env_config_dir, get_noun_chunk_corpus_path, get_noun_chunk_frequencies_path, get_chunk_landmark_alignment_path, \
    get_unaligned_chunk_list_path, get_corpus_dir, get_wordset_path, get_wordfreq_path

# TODO: Get rid of these:
UNITY_UNITS = True

cache = None

loaded_alignments = None
loaded_corpus = None
loaded_size = None


def load_train_dev_test_annotations(anno_set):
    corpus_dir = get_corpus_dir()
    if anno_set == "8000":
        train_data = load_json(os.path.join(corpus_dir, "train_annotations_8000.json"))
        dev_data = load_json(os.path.join(corpus_dir, "dev_annotations_8000.json"))
        test_data = load_json(os.path.join(corpus_dir, "test_annotations_8000.json"))
        return train_data, dev_data, test_data
    if anno_set == "7000":
        train_data = load_json(os.path.join(corpus_dir, "train_annotations_7000.json"))
        dev_data = load_json(os.path.join(corpus_dir, "dev_annotations_7000.json"))
        test_data = load_json(os.path.join(corpus_dir, "test_annotations_7000.json"))
        return train_data, dev_data, test_data
    elif anno_set == "6000":
        train_data = load_json(os.path.join(corpus_dir, "train_annotations_6000.json"))
        dev_data = load_json(os.path.join(corpus_dir, "dev_annotations_6000.json"))
        test_data = load_json(os.path.join(corpus_dir, "test_annotations_6000.json"))
        return train_data, dev_data, test_data
    elif anno_set == "4000":
        data = load_json(os.path.join(corpus_dir, "annotation_results.json"))
        return data["train"], data["dev"], data["test"]
    else:
        raise ValueError(f"Unknown annotation set: {anno_set}. Expected one of: 4000, 6000, 7000, 8000")


def augment_dataset(dataset, merge_len=2, min_merge_len=1):
    print("ding")
    total_added = 0
    merge_range_start = max(2, min_merge_len)

    for env, sets in dataset.items():
        for set_idx in range(len(sets)):
            set = sets[set_idx]
            all_segs = set["instructions"]
            num_segs = len(set["instructions"])
            for i in range(num_segs):
                set["instructions"][i]["int_points"] = []
                set["instructions"][i]["merge_len"] = 1
                set["instructions"][i]["seg_idx"] = i

            augseg_id = num_segs
            #set["instructions"] is a list of dicts, where each dict corresponds to an instruction segment
            # each instruction segment dict has keys:
            #   instruction: str
            #   end_idx, start_idx - path end/start indices, integers
            #   start_pos, end_pos - list, 2D coordinate
            #   start_yaw, end_yaw - float, radians
            new_segs = []
            for this_merge_len in range(merge_range_start, merge_len+1):
                for i in range(num_segs - this_merge_len + 1):
                    aug_segs = [all_segs[j] for j in range(i, i + this_merge_len)]
                    aug_seg_d = {}
                    aug_seg_d["instruction"] = " ".join([seg["instruction"] for seg in aug_segs])
                    aug_seg_d["end_idx"] = aug_segs[-1]["end_idx"]
                    aug_seg_d["start_idx"] = aug_segs[0]["start_idx"]
                    aug_seg_d["start_pos"] = aug_segs[0]["start_pos"]
                    aug_seg_d["start_yaw"] = aug_segs[0]["start_yaw"]
                    aug_seg_d["end_pos"] = aug_segs[-1]["end_pos"]
                    aug_seg_d["end_yaw"] = aug_segs[-1]["end_yaw"]
                    aug_seg_d["int_points"] = [aug_segs[i]["end_pos"] for i in range(this_merge_len - 1)]
                    aug_seg_d["merge_len"] = this_merge_len
                    aug_seg_d["seg_idx"] = augseg_id
                    total_added += 1
                    augseg_id += 1
                    new_segs.append(aug_seg_d)

            if min_merge_len > 1:
                dataset[env][set_idx]["instructions"] = new_segs
            else:
                dataset[env][set_idx]["instructions"] += new_segs
    print(f"Augmented dataset with {total_added} new merged segments")
    return dataset


def add_parse_results(dataset):
    print("Adding parsing results to instruction data")
    nlp = spacy_singleton.load("en_core_web_lg")
    for env, sets in dataset.items():
        for set_idx in range(len(sets)):
            set = sets[set_idx]
            num_segs = len(set["instructions"])
            for i in range(num_segs):
                instr_str = set["instructions"][i]["instruction"]
                doc = nlp(instr_str)
                chunk_strings = [str(chunk) for chunk in doc.noun_chunks]
                set["instructions"][i]["noun_chunks"] = chunk_strings
    return dataset


def get_all_instructions(max_size=0, do_prune_ambiguous=False, ignore_min_augment_len=False):
    #print("max_size:", max_size)

    # If instructions already loaded in memory, return them
    global cache
    global loaded_corpus
    global loaded_size

    if ignore_min_augment_len:
        min_augment_len = 1
    else:
        min_augment_len = P.get_current_parameters()["Setup"].get("min_augment_len", 1)
    max_augment_len = P.get_current_parameters()["Setup"].get("augment_len", 1)
    corpus_name = P.get_current_parameters()["Setup"].get("corpus_name")

    train_key = f"train-{min_augment_len}-{max_augment_len}"
    dev_key = f"dev-{min_augment_len}-{max_augment_len}"
    test_key = f"test-{min_augment_len}-{max_augment_len}"

    if cache is not None and train_key in cache:# loaded_size == max_size:
        train_instructions = cache[train_key]
        dev_instructions = cache[dev_key]
        test_instructions = cache[test_key]
        wordset = loaded_corpus

    # Otherwise see if they've been pre-build in tmp files
    else:
        # Cache
        cache_dir = get_instruction_cache_dir()

        train_file = os.path.join(cache_dir, f"train_{min_augment_len}-{max_augment_len}.json")
        dev_file = os.path.join(cache_dir, f"dev_{min_augment_len}-{max_augment_len}.json")
        test_file = os.path.join(cache_dir, f"test_{min_augment_len}-{max_augment_len}.json")
        wordset_file = get_wordset_path()
        wordfreq_file = get_wordfreq_path()

        corpus_already_exists = False
        if os.path.isfile(wordset_file):
            with open(wordset_file, "r") as f:
                wordset = list(json.load(f))
                #print("corpus: ", len(corpus))
            corpus_already_exists = True

        # If they have been saved in tmp files, load them
        if os.path.isfile(train_file):
            train_instructions = load_instruction_data_from_json(train_file)
            dev_instructions = load_instruction_data_from_json(dev_file)
            test_instructions = load_instruction_data_from_json(test_file)
            assert corpus_already_exists, "Insruction data exists but corpus is gone!"

        # Otherwise rebuild instruction data from annotations
        else:
            print(f"REBUILDING INSTRUCTION DATA FOR SEGMENT LENGTHS: {min_augment_len} to {max_augment_len}!")
            print(f"USING OLD CORPUS: {corpus_already_exists}")
            os.makedirs(cache_dir, exist_ok=True)

            all_instructions, new_wordset = defaultdict(list), set()

            train_an, dev_an, test_an = load_train_dev_test_annotations(corpus_name)

            print("Loaded JSON Data")

            print("Parsing dataset")
            print("    train...")
            train_instructions, new_wordset, word_freq = parse_dataset(train_an, new_wordset)
            print("    dev...")
            dev_instructions, new_wordset, _ = parse_dataset(dev_an, new_wordset)
            print("    test...")
            test_instructions, new_wordset, _ = parse_dataset(test_an, new_wordset)

            print("Augmenting maybe?")
            train_instructions = augment_dataset(train_instructions, merge_len=max_augment_len, min_merge_len=min_augment_len)
            dev_instructions = augment_dataset(dev_instructions, merge_len=max_augment_len, min_merge_len=min_augment_len)
            test_instructions = augment_dataset(test_instructions, merge_len=max_augment_len, min_merge_len=min_augment_len)

            train_instructions = add_parse_results(train_instructions)
            dev_instructions = add_parse_results(dev_instructions)
            test_instructions = add_parse_results(test_instructions)

            save_json(train_instructions, train_file)
            save_json(dev_instructions, dev_file)
            save_json(test_instructions, test_file)

            out_corpus_path = get_wordset_path(corpus_name)

            if not os.path.exists(out_corpus_path):
                wordset = new_wordset
                save_json(list(wordset), wordset_file)
                save_json(word_freq, wordfreq_file)
            else:
                print("Warning! Regenerated pomdp, but kept the old corpus!")

            print("Saved instructions for quicker loading!")

    # Clip datasets to the provided size
    if max_size is not None and max_size > 0:
        num_train = int(math.ceil(max_size*0.7))
        num_dev = int(math.ceil(max_size*0.15))
        num_test = int(math.ceil(max_size*0.15))

        train_instructions = slice_list_tail(train_instructions, num_train)
        dev_instructions = slice_list_tail(dev_instructions, num_dev)
        test_instructions = slice_list_tail(test_instructions, num_test)

    if do_prune_ambiguous:
        train_instructions = prune_ambiguous(train_instructions)
        dev_instructions = prune_ambiguous(dev_instructions)
        test_instructions = prune_ambiguous(test_instructions)

    #print("Corpus: ", len(corpus))
    #print("Loaded: ", len(train_instructions), len(dev_instructions), len(test_instructions))
    if cache is None:
        cache = {}

    cache[train_key] = train_instructions
    cache[dev_key] = dev_instructions
    cache[test_key] = test_instructions
    loaded_corpus = wordset
    loaded_size = max_size

    return train_instructions, dev_instructions, test_instructions, wordset


def get_instruction_segment(env_id, set_idx, seg_idx, ignore_min_augment_len=True, all_instr=None):
    if all_instr is None:
        train_instr, dev_instr, test_instr, corpus = get_all_instructions(ignore_min_augment_len=ignore_min_augment_len)
        all_instr = {**train_instr, **dev_instr, **test_instr}
    inst_segment = None
    try:
        for seg in all_instr[env_id][set_idx]["instructions"]:
            if seg["seg_idx"] == seg_idx:
                inst_segment = seg
                break
    except Exception as e:
        print ("ding")
        print(e)
    return inst_segment


def get_segs_available_for_env(env_id, set_idx, ignore_min_augment_len=False, all_instr=None, split=None):
    if all_instr is None:
        train_instr, dev_instr, test_instr, corpus = get_all_instructions(ignore_min_augment_len=ignore_min_augment_len)
        if split == "train":
            all_instr = train_instr
        elif split == "dev":
            all_instr = dev_instr
        elif split == "test":
            all_instr = test_instr
        else:
            all_instr = {**train_instr, **dev_instr, **test_instr}

    segs_available = set([s["seg_idx"] for s in all_instr[env_id][set_idx]["instructions"]])
    return list(sorted(segs_available))


def load_instruction(env_id, set_idx, seg_idx):
    segment = get_instruction_segment(env_id, set_idx, seg_idx)
    return segment["instruction"]


def load_noun_chunk_corpus_and_frequencies():
    with open(get_noun_chunk_corpus_path(), "r") as fp:
        noun_chunk_corpus = json.load(fp)
    with open(get_noun_chunk_frequencies_path(), "r") as fp:
        noun_chunk_frequencies = json.load(fp)
    return noun_chunk_corpus, noun_chunk_frequencies


_loaded_alignment_splits = {}
def load_noun_chunk_landmark_alignments(split="train"):
    global _loaded_alignment_splits
    path = get_chunk_landmark_alignment_path(split)
    if path not in _loaded_alignment_splits:
        with open(path, "r") as fp:
            chunk_landmark_alignments = json.load(fp)
            _loaded_alignment_splits[path] = chunk_landmark_alignments
    return _loaded_alignment_splits[path]


def load_landmark_noun_chunks_by_name():
    noun_chunk_corpus = load_noun_chunk_landmark_alignments()
    lmchunks = { name: [] for name in get_landmark_names(add_empty=True)}
    for chunk, data in noun_chunk_corpus.items():
        for landmark_name in data["landmarks"]:
            lmchunks[landmark_name].append(chunk)
    return lmchunks


def clean_instruction(instruction_str):
    return ''.join(ch if ch not in set(string.punctuation).difference(set(["_", "<", ">"])) else ' ' for ch in instruction_str).strip().lower()


def split_instruction(instruction_str):
    instruction_str = clean_instruction(instruction_str)
    return instruction_str.split()


def tokenize_instruction(instruction_str, word2token):
    instruction_str = clean_instruction(instruction_str)
    instruction_split = split_instruction(instruction_str)
    if word2token is None:
        print("word2token is None")
    tokenized = [word2token[word] if word in word2token else 0 for word in instruction_split]
    if len(tokenized) == 0:
        tokenized = [0]
    return tokenized


def debug_untokenize_instruction(tokens, token2term=None):
    if token2term is None:
        _, _, _, corpus = get_all_instructions()
        token2term, word2token = get_word_to_token_map(corpus)
    if isinstance(tokens, torch.Tensor):
        tokens = [t.item() for t in tokens]
    for w in tokens:
        if w not in token2term:
            raise ValueError(f"Missing token! her's what we have: {token2term}")
    return " ".join([token2term[w] for w in tokens])


def words_to_terms(word_list, word2term):
    return [word2term[word] for word in word_list if word in word2term]


def get_closest_point_idx(path, point):
    dists = [np.linalg.norm(point - p_point) for p_point in path]
    min_idx = np.argmin(dists)
    return min_idx


def anno_to_config(point):
    if UNITY_UNITS:
        return (point - [225, 225]) * 1000 / 50
    else:
        return point


def get_path_end_indices(env_id, entry):
    num_segments = len(entry["instructions"])
    path = load_path(env_id)
    indices = []
    for i in range(num_segments):
        end_x = entry["end_x"][i]
        end_z = entry["end_z"][i]
        end_point = np.asarray([end_x, end_z])
        # Convert from Unity-specific to normalized coordinates
        # TODO FIXME: This check is a potential source of future bugs!
        # The real language data from unity includes ground truth moves. The templated one doesn't
        # Unity data stores end coordinate in unity coords, templated data in config coords.
        if len(entry["moves"]) > 0:
            end_point = (end_point - [225, 225]) * (1000 / 50)
        end_idx = int(get_closest_point_idx(path, end_point))
        indices.append(end_idx)
    return indices


def extract_start_end_pos(entry):
    num_seg = len(entry["instructions"])
    start_pos = []
    end_pos = []
    for i in range(num_seg):
        start_p_unity = np.asarray([entry["start_x"][i], entry["start_z"][i]])
        start_pos.append(anno_to_config(start_p_unity))
        end_p_unity = np.asarray([entry["end_x"][i], entry["end_z"][i]])
        end_pos.append(anno_to_config(end_p_unity))
    return start_pos, end_pos


def parse_dataset(data, corpus, max_size=0):
    word_freq = {}
    all_instructions = defaultdict(list)
    for i, entry in enumerate(data):
        dp = {}
        env = int(entry["id"])
        dp["env"] = env
        segment_end_indices = get_path_end_indices(env, entry)
        start_pos, end_pos = extract_start_end_pos(entry)
        text_instructions = entry["instructions"]
        instructions = []
        prev_seg_end = 0
        for i in range(len(text_instructions)):
            inst = text_instructions[i]
            inst = clean_instruction(inst)
            start_yaw = -np.deg2rad(entry["start_rot"][i]) + np.pi/2
            if "end_rot" in entry:
                end_yaw = -np.deg2rad(entry["end_rot"][i]) + np.pi/2
            else:
                end_yaw = 0
            seg_entry = {
                "instruction": inst,
                "end_idx": segment_end_indices[i],
                "start_idx": prev_seg_end,
                "start_pos": list(start_pos[i]),
                "start_yaw": start_yaw,
                "end_yaw": end_yaw,
                "end_pos": list(end_pos[i])
            }
            instructions.append(seg_entry)
            prev_seg_end = segment_end_indices[i]
            corpus = corpus.union(inst.split())

            # Count word frequencies
            for word in inst.split():
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1

        dp["instructions"] = instructions
        if max_size == 0 or len(all_instructions) < max_size:
            all_instructions[env].append(dp)

    keys = list(all_instructions.keys())
    values = list(all_instructions.values())

    integer_keys = np.asarray(keys)
    sort_indices = np.argsort(integer_keys)

    all_instructions = OrderedDict([(keys[i], values[i]) for i in sort_indices])
    return all_instructions, corpus, word_freq


def slice_list_tail(dataset, size):
    keys_int = sorted(list(dataset.keys()))
    dataset_out = {}
    for i, intkey in enumerate(keys_int):
        if i > size:
            break
        dataset_out[intkey] = dataset[intkey]
    return dataset_out


def load_instruction_data_from_json(json_file):
    proper_data = {}
    with open(json_file, "r") as f:
        data_w_string_keys = json.load(f)
    for key in data_w_string_keys.keys():
        proper_data[int(key)] = data_w_string_keys[key]
    return proper_data


def is_ambiguous(env_id):
    template = load_template(env_id)
    config = load_env_config(env_id)
    #TODO: Handle the case where instructions refer to multiple landmarks
    ref_landmark = template["landmark1"]
    occ_count = 0
    for landmark_name in config["landmarkName"]:
        if has_ambiguous_noun_phrase(landmark_name, ref_landmark):
            occ_count += 1
    if occ_count == 0:
        print("Error! Referred to landmark that's not found in env!")
        exit(-1)
    # More than one such landmark occurs in the test set
    if occ_count > 1:
        return True
    else:
        return False


def prune_ambiguous(instruction_data):
    print("Pruning ambiguous instructions on dataset of size: " + str(len(instruction_data)))
    data_out = {}
    num_pruned = 0
    for key, val in instruction_data.items():
        if not is_ambiguous(key):
            data_out[key] = val
        else:
            num_pruned += 1
    print("Pruned " + str(num_pruned) + " envs from instruction data")
    return data_out


def merge_instruction_sets(train, dev, test=None):
    res = OrderedDict(chain(train.items(), dev.items()))
    if test is not None:
        res = OrderedDict(chain(res.items(), test.items()))
    return res


def get_env_ids(instruction_set):
    keys = instruction_set.keys()
    ids = [int(key) for key in keys]
    return ids


def seg_idx_to_ordinal(instruction_segments, seg_idx):
    for i, seg in enumerate(instruction_segments):
        if seg["seg_idx"] == seg_idx:
            return i
    raise ValueError("Seg idx can't be converted to ordinal - not present!")


def get_restricted_env_id_lists(prune_ambiguous=False, ignore_min_augment_len=False):
    setup = P.get_current_parameters()["Setup"]
    train_instructions, dev_instructions, test_instructions, _ = get_all_instructions(setup["max_envs"], do_prune_ambiguous=prune_ambiguous, ignore_min_augment_len=ignore_min_augment_len)

    train_envs = sorted([int(key) for key in train_instructions.keys()])
    dev_envs = sorted([int(key) for key in dev_instructions.keys()])
    test_envs = sorted([int(key) for key in test_instructions.keys()])

    if setup.get("env_range_start", 0) > 0:
        train_envs = [e for e in train_envs if e >= setup["env_range_start"]]
        dev_envs = [e for e in dev_envs if e >= setup["env_range_start"]]
        test_envs = [e for e in test_envs if e >= setup["env_range_start"]]

    if setup.get("env_range_end", 0) > 0:
        train_envs = [e for e in train_envs if e < setup["env_range_end"]]
        dev_envs = [e for e in dev_envs if e < setup["env_range_end"]]
        test_envs = [e for e in test_envs if e < setup["env_range_end"]]

    if setup["max_envs"] > 0:
        train_envs = train_envs[:setup["max_envs"]]
        dev_envs = dev_envs[:setup["max_envs"]]
        test_envs = test_envs[:setup["max_envs"]]

    specenvs = setup.get("only_specific_envs")
    if specenvs:
        train_envs = [e for e in train_envs if e in specenvs]
        dev_envs = [e for e in dev_envs if e in specenvs]
        test_envs = [e for e in test_envs if e in specenvs]
    return train_envs, dev_envs, test_envs


def get_correct_eval_env_id_list():
    setup = P.get_current_parameters()["Setup"]
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists(setup["prune_ambiguous"])

    eval_envs = dev_envs
    if setup["eval_env_set"] == "train":
        print("Using TRAIN set!")
        eval_envs = train_envs
    elif setup["eval_env_set"] == "test":
        print ("Using TEST set!")
        eval_envs = test_envs
    else:
        print("Using DEV set!")

    if setup.get("only_specific_envs", False):
        print("Using only specific envs!")
        specific_envs = setup.get("only_specific_envs")
        #for specific_env in specific_envs:
        #    assert specific_env in eval_envs, (f"Env id {specific_env} from only_specific_envs does not "
        #        f"match the rest of the specification!: {setup} \n which includes the following envs: {eval_envs}")
        eval_envs = specific_envs

    if setup.get("env_range_start", 0) > 0:
        eval_envs = [e for e in eval_envs if e >= setup["env_range_start"]]
    if setup.get("env_range_end", 0) > 0:
        eval_envs = [e for e in eval_envs if e < setup["env_range_end"]]
    if setup["max_envs"] > 0:
        eval_envs = eval_envs[:setup["max_envs"]]

    return eval_envs


def get_env_id_lists_perception(max_envs, p_train=0.8, p_dev=0.2, p_test=0.0):
    assert (p_train + p_dev + p_test > 0.99)
    assert (p_train + p_dev + p_test < 1.01)
    config_files = [dirname for dirname in os.listdir(get_env_config_dir()) if dirname.endswith(".json")]
    env_ids = [int(dirname.replace('.', '_').split("_")[-2]) for dirname in config_files]

    n = min(len(env_ids), max_envs)
    n_train = round(n * p_train)
    n_dev = round(n * p_dev)
    n_test = n - n_train - n_dev
    print(n_train, n_dev, n_test)
    idx = np.arange(n)
    np.random.shuffle(idx)

    train_envs = [env_ids[i] for i in idx[:n_train]]
    dev_envs = [env_ids[i] for i in idx[n_train:n_train + n_dev]]
    test_envs = [env_ids[i] for i in idx[n_test:]]

    return train_envs, dev_envs, test_envs


def load_english_vocabulary():
    path = get_english_vocab_path()
    with open(path, "r") as fp:
        vocab = json.load(fp)

    # Add words missing in the dictionary:
    vocab["dumpster"] = 1
    vocab["waterfountain"] = 1
    vocab["photobooth"] = 1
    vocab["streetlamp"] = 1
    vocab["kingkong"] = 1
    vocab["gravesite"] = 1
    vocab["trashcan"] = 1

    return vocab


def get_word_to_token_map(corpus, use_thesaurus=True):
    corpus = list(corpus)
    thesaurus = load_landmark_alignments()

    # Final outputs
    word2token = {}
    token2term = {}

    # If we don't have a thesaurus, give each word in the corpus a unique ID
    if thesaurus is None or not use_thesaurus:
        for i, w in enumerate(corpus):
            word2token[w] = i
            token2term[i] = w

        return token2term, word2token

    # Intermediate storage
    term2token = {}
    word2term = thesaurus["word2term"]
    term2word = thesaurus["term2word"]

    # If we do have a thesaurus, tokenize in a smarter way.
    # We will have a token for each term in the thesaurus
    # every word that's not found in the thesaurus will be assigned a special "unknown" token
    # every token will map back to it's term in the thesaurus (not the word). Unknown tokens will map to a special term
    terms = list(sorted(term2word.keys()))
    for i, t in enumerate(terms):
        term2token[t] = i

    unknown_token = len(terms)
    unknown_term = "NA"

    for word in corpus:
        if word in word2term:
            term = word2term[word]
            token = term2token[term]
            word2token[word] = token
            token2term[token] = term
        else:
            word2token[word] = unknown_token

    token2term[unknown_token] = unknown_term

    return token2term, word2token


def save_landmark_alignments(alignments):
    path = get_thesaurus_path()
    with open(path, "w") as fp:
        json.dump(alignments, fp, indent=4)
    print("Saved thesaurus")


def save_chunk_landmark_alignments(alignments, unaligned_chunks=None, split="train"):
    path = get_chunk_landmark_alignment_path(split)
    with open(path, "w") as fp:
        json.dump(alignments, fp, indent=4)
    print("Saved chunk-landmark alignments")
    if unaligned_chunks:
        upath = get_unaligned_chunk_list_path()
        with open(upath, "w") as fp:
            json.dump(unaligned_chunks, fp, indent=4)
        print("Saved list of unaligned noun chunks")


def load_landmark_alignments():
    global loaded_alignments
    if loaded_alignments is None:
        path = get_thesaurus_path()
        try:
            with open(path, "r") as fp:
                loaded_alignments = json.load(fp)
            #print("Loaded thesaurus with " + str(len(loaded_alignments["term2word"])) + " terms")
        except Exception as e:
            print(f"Failed loading thesaurus from: {path}")
            loaded_alignments = None

    return loaded_alignments


def get_mentioned_landmarks(thesaurus, str_instruction):
    split_instr = split_instruction(clean_instruction(str_instruction))
    word2term = thesaurus["word2term"]
    term_groundings = thesaurus["term_groundings"]
    lm_name2index = get_landmark_name_to_index()

    # Map each word in the instruction to it's corresponding term:
    split_instr_terms = words_to_terms(split_instr, word2term)

    mentioned_landmark_names = set()

    # For each term, find all the landmarks that have been mentioned
    for term in split_instr_terms:
        for landmark_name in term_groundings[term]["landmarks"]:
            mentioned_landmark_names.add(landmark_name)

    mentioned_landmark_names = list(mentioned_landmark_names)
    mentioned_landmark_indices = [lm_name2index[name] for name in mentioned_landmark_names]
    return mentioned_landmark_names, mentioned_landmark_indices

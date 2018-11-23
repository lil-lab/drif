import json
import os
import string
from collections import defaultdict, OrderedDict
from itertools import chain

import numpy as np
import math

from data_io.helpers import load_json, save_json
from env_config.definitions.nlp_templates import has_ambiguous_noun_phrase
from data_io.env import load_path, load_env_config, load_template
import parameters.parameter_server as P
from data_io.paths import get_config_dir, get_instruction_cache_dir, get_english_vocab_path, get_thesaurus_path, get_env_config_dir

# TODO: Get rid of these:
DATA_6000 = False
UNITY_UNITS = False

loaded_thesaurus = None
loaded_train_instructions = None
loaded_test_instructions = None
loaded_dev_instructions = None
loaded_corpus = None
loaded_size = None


def load_train_dev_test_annotations():
    config_dir = get_config_dir()
    if not DATA_6000:
        data = load_json(os.path.join(config_dir, "annotation_results.json"))
        return data["train"], data["dev"], data["test"]
    else:
        train_data = load_json(os.path.join(config_dir, "train_annotations_6000.json"))
        dev_data = load_json(os.path.join(config_dir, "dev_annotations_6000.json"))
        test_data = load_json(os.path.join(config_dir, "test_annotations_6000.json"))
        return train_data, dev_data, test_data


def get_all_instructions(max_size=0, do_prune_ambiguous=False):
    #print("max_size:", max_size)

    # If instructions already loaded in memory, return them
    global loaded_train_instructions
    global loaded_test_instructions
    global loaded_dev_instructions
    global loaded_corpus
    global loaded_size
    if loaded_train_instructions is not None and loaded_size == max_size:
        train_instructions = loaded_train_instructions
        dev_instructions = loaded_dev_instructions
        test_instructions = loaded_test_instructions
        corpus = loaded_corpus

    # Otherwise see if they've been pre-build in tmp files
    else:
        # Cache
        cache_dir = get_instruction_cache_dir()
        corpus_dir = get_config_dir()

        train_file = os.path.join(cache_dir,"train.json")
        dev_file = os.path.join(cache_dir, "dev.json")
        test_file = os.path.join(cache_dir, "test.json")
        corpus_file = os.path.join(corpus_dir, "corpus.json")
        wfreq_file = os.path.join(corpus_dir, "word_freq.json")

        corpus_already_exists = False
        if os.path.isfile(corpus_file):
            with open(corpus_file, "r") as f:
                corpus = list(json.load(f))
                print("corpus: ", len(corpus))
            corpus_already_exists = True

        # If they have been saved in tmp files, load them
        if os.path.isfile(train_file):
            train_instructions = load_instruction_data_from_json(train_file)
            dev_instructions = load_instruction_data_from_json(dev_file)
            test_instructions = load_instruction_data_from_json(test_file)

        # Otherwise rebuild instruction data from annotations
        else:
            print("REBUILDING INSTRUCTION DATA! CORPUS WILL NOT BE VALID!")
            os.makedirs(cache_dir, exist_ok=True)

            all_instructions, corpus = defaultdict(list), set()

            train_an, dev_an, test_an = load_train_dev_test_annotations()

            print("Loaded JSON Data")

            train_instructions, corpus, word_freq = parse_dataset(train_an, corpus)
            dev_instructions, corpus, _ = parse_dataset(dev_an, corpus)
            test_instructions, corpus, _ = parse_dataset(test_an, corpus)

            #train_instructions = augment_dataset(train_instructions)
            #dev_instructions = augment_dataset(dev_instructions)
            #test_instructions = augment_dataset(test_instructions)

            save_json(train_instructions, train_file)
            save_json(dev_instructions, dev_file)
            save_json(test_instructions, test_file)

            if not corpus_already_exists:
                save_json(list(corpus), corpus_file)
                save_json(word_freq, wfreq_file)
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
    loaded_train_instructions = train_instructions
    loaded_dev_instructions = dev_instructions
    loaded_test_instructions = test_instructions
    loaded_corpus = corpus
    loaded_size = max_size

    return train_instructions, dev_instructions, test_instructions, corpus


def get_instruction_segment(env_id, set_idx, seg_idx):
    train_instr, dev_instr, test_instr, corpus = get_all_instructions()
    all_instr = {**train_instr, **dev_instr, **test_instr}
    try:
        inst_segment = all_instr[env_id][set_idx]["instructions"][seg_idx]
    except Exception as e:
        print ("ding")
    return inst_segment


def load_instruction(env_id, set_idx, seg_idx):
    segment = get_instruction_segment(env_id, set_idx, seg_idx)
    return segment["instruction"]


def clean_instruction(instruction_str):
    return ''.join(ch if ch not in set(string.punctuation) else ' ' for ch in instruction_str).strip().lower()


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


def get_all_env_id_lists(max_envs=0, prune_ambiguous=False):
    train_instructions, dev_instructions, test_instructions, _ = get_all_instructions(max_envs, do_prune_ambiguous=prune_ambiguous)

    train_envs = sorted([int(key) for key in train_instructions.keys()])
    dev_envs = sorted([int(key) for key in dev_instructions.keys()])
    test_envs = sorted([int(key) for key in test_instructions.keys()])

    return train_envs, dev_envs, test_envs


def get_correct_eval_env_id_list():
    setup = P.get_current_parameters()["Setup"]
    train_envs, dev_envs, test_envs = get_all_env_id_lists(setup["max_envs"], setup["prune_ambiguous"])

    eval_envs = dev_envs
    if setup["eval_env_set"] == "train":
        print("Using TRAIN set!")
        eval_envs = train_envs
    elif setup["eval_env_set"] == "test":
        print ("Using TEST set!")
        eval_envs = test_envs
    else:
        print("Using DEV set!")

    if setup["env_range_start"] > 0:
        eval_envs = eval_envs[setup["env_range_start"]:]
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
    thesaurus = load_thesaurus()

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


def save_thesaurus(thesaurus, giza=False):
    path = get_thesaurus_path(giza)
    with open(path, "w") as fp:
        json.dump(thesaurus, fp, indent=4)


def load_thesaurus():
    global loaded_thesaurus
    if loaded_thesaurus is None:
        path = get_thesaurus_path()
        try:
            with open(path, "r") as fp:
                loaded_thesaurus = json.load(fp)
            print("Loaded thesaurus with " + str(len(loaded_thesaurus["term2word"])) + " terms")
        except Exception as e:
            print("Failed loading thesaurus")
            loaded_thesaurus = None

    return loaded_thesaurus
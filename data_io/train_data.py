import gc
import os
import pickle
import sys
from multiprocessing.pool import Pool

from data_io.instructions import get_all_instructions
from data_io.paths import get_dataset_dir, get_model_dir
from data_io.instructions import get_env_ids

import parameters.parameter_server as P

MIN_SAMPLE_LENGTH = 5


def save_dataset(dataset, name):
    full_path = os.path.join(get_dataset_dir(), str(name))
    dirname = os.path.dirname(full_path)
    if not os.path.isdir(dirname):
        try: os.makedirs(dirname)
        except Exception: pass

    gc.disable()
    f = open(full_path, "wb")
    pickle.dump(dataset, f)
    f.close()
    gc.enable()


def load_dataset(name):
    path = os.path.join(get_dataset_dir(), str(name))
    gc.disable()
    f = open(path, "rb")
    try:
        dataset = pickle.load(f)
    except Exception as e:
        print("Error loading dataset pickle")
        dataset = []
    f.close()
    gc.enable()
    return dataset


def load_single_env_from_dataset(env, dataset_name):
    if dataset_name == "supervised":
        return load_single_env_supervised_data(env)
    else:
        return load_dataset(dataset_name + "_" + str(env))


def filter_env_list_has_data(env_list, dataset_name):
    good_envs = []
    cut_some = False
    for env in env_list:
        filename = "supervised_train_data_env_" + str(env) if dataset_name == "supervised" else dataset_name + "_" + str(env)
        path = os.path.join(get_dataset_dir(), filename)
        # Check that the data file exists and
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            good_envs.append(env)
        else:
            cut_some = True
    if cut_some:
        print("Filtered out " + str(len(env_list) - len(good_envs)) + " envs because of missing data")
    return good_envs


def load_single_env_supervised_data(env):
    env_data = []
    #print ("Data env: ", env)
    try:
        env_data = load_dataset("supervised_train_data_env_" + str(env))
    except ImportError as err:
        print(err)
    except OSError:
        print("Data for env " + str(env) + "unavailable! Skipping..")
        print(sys.exc_info()[0])
    return env_data


def sanitize_data(data):
    data_out = []
    max_sample_length = P.get_current_parameters()["Setup"]["trajectory_length"]
    for rollout in data:
        if MIN_SAMPLE_LENGTH < len(rollout) < max_sample_length:
            data_out.append(rollout)
        else:
            print("Skipping rollout of length: " + str(len(rollout)))
    return data_out


def split_in_segments(data):
    output_segs = []
    for env_rollout in data:
        if len(env_rollout) == 0:
            continue
        seg_idx = 0
        prev = 0
        for i, sample in enumerate(env_rollout):
            if sample["metadata"]["seg_idx"] != seg_idx:
                output_segs.append(env_rollout[prev:i])
                prev=i
                seg_idx = sample["metadata"]["seg_idx"]
        if prev < len(env_rollout) - 1:
            output_segs.append(env_rollout[prev:len(env_rollout) - 1])
    return output_segs


def load_supervised_data(max_envs=None, split_segments=False):
    train_i, dev_i, test_i, _ = get_all_instructions(max_envs)
    train_env_ids = get_env_ids(train_i)
    dev_env_ids = get_env_ids(dev_i)

    pool = Pool(processes=20)

    # Load dataset for each env in parallel
    train_env_data = pool.map(load_single_env_supervised_data, train_env_ids)
    dev_env_data = pool.map(load_single_env_supervised_data, dev_env_ids)
    pool.close()

    #dev_data = []
    #train_data = []
    # Combine into a single dataset
    #for data in train_env_data:
    #    train_data += data
    #for data in dev_env_data:
    #    dev_data += data

    train_data = sanitize_data(train_env_data)
    dev_data = sanitize_data(dev_env_data)

    if split_segments:
        train_data = split_in_segments(train_data)
        dev_data = split_in_segments(dev_data)

    return train_data, dev_data


def file_exists(name):
    full_path = os.path.join(get_dataset_dir(), str(name))
    data_exists = os.path.exists(full_path)
    full_path = os.path.join(get_model_dir(), str(name))
    model_exists = os.path.exists(full_path)
    return data_exists or model_exists
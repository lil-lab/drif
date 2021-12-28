import gc
import os
import pickle
import sys
import shutil
from multiprocessing.pool import Pool
from functools import partial

from data_io.instructions import get_all_instructions
from data_io.paths import get_dataset_dir, get_model_dir
from data_io.instructions import get_env_ids
from data_io.paths import get_supervised_data_filename

import parameters.parameter_server as P

from time import sleep

MIN_SAMPLE_LENGTH = 5
LOCK_TIMEOUT = 2.0
LOCK_WAIT_TIME = 0.01


def save_dataset(dataset_name, dataset, name=None, env_id=None, lock=False):
    if name is None:
        name = get_supervised_data_filename(env_id)
    full_path = os.path.join(get_dataset_dir(dataset_name), str(name))
    save_dataset_to_path(full_path, dataset, lock=lock)
    #print(f"Saved {len(dataset)} rollouts for dataset {dataset_name}/{name}")


def lock_path(path):
    return path + "_lock_"


def has_lock(path):
    return os.path.exists(lock_path(path))


def do_lock(path):
    # There is a possible point of contention where both processes simultaneously clear await_lock()
    # and then proceed to execute do_lock at the same time. I'd wager this is a very low probability event
    await_lock(path)
    f = open(lock_path(path), "w")
    f.writelines(["locked"])
    f.close()


def do_unlock(path):
    if os.path.exists(lock_path(path)):
        os.remove(lock_path(path))


def await_lock(path):
    time_slept = 0.0
    removed = False
    while has_lock(path):
        if removed:
            print(f"LOCK REMOVED BUT STILL HAS_LOCK: {path}")
        sleep(LOCK_WAIT_TIME)
        time_slept += LOCK_WAIT_TIME
        if time_slept > LOCK_TIMEOUT:
            do_unlock(path)
            removed = True
    return


def save_dataset_to_path(full_path, dataset, lock=False):
    lock = lock or P.get_current_parameters()["Data"].get("locking")
    dirname = os.path.dirname(full_path)
    if not os.path.isdir(dirname):
        try: os.makedirs(dirname)
        except Exception: pass

    if lock:
        do_lock(full_path)
    gc.disable()
    f = open(full_path, "wb")
    pickle.dump(dataset, f)
    f.close()
    gc.enable()
    if lock:
        do_unlock(full_path)


def load_dataset(dataset_name, name, lock=False):
    path = os.path.join(get_dataset_dir(dataset_name), str(name))
    return load_dataset_from_path(path, lock=lock)


def load_dataset_from_path(path, lock=False):
    lock = lock or P.get_current_parameters().get("Data").get("locking") if "Data" in P.get_current_parameters() else False
    if lock:
        do_lock(path)
    gc.disable()
    f = open(path, "rb")
    try:
        dataset = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset pickle: {path}")
        print(e)
        raise e
    f.close()
    gc.enable()
    if lock:
        do_unlock(path)
    return dataset


def split_into_segs(env_datas):
    segs = []
    for env_data in env_datas:
        seg_idx = -1
        seg = []
        for sample in env_data:
            if "metadata" in sample:
                isthis = sample["metadata"]["seg_idx"] == seg_idx
            else:
                isthis = sample["seg_idx"] != seg_idx
            if isthis:
                if len(seg) > 0:
                    segs.append(seg)
                seg = [sample]
                seg_idx = sample["seg_idx"] if "metadata" not in sample else sample["metadata"]["seg_idx"]
            else:
                seg.append(sample)
        segs.append(seg)
    return segs


def load_single_env_from_dataset(dataset_name, env, prefix):
    if prefix == "supervised":
        return load_single_env_supervised_data(dataset_name, env)
    else:
        return load_dataset(dataset_name, prefix + "_" + str(env) if prefix else str(env))


def load_single_env_metadata_from_dataset(dataset_name, env, prefix):
    try:
        metadata = load_dataset(dataset_name, f"metadata_{env}")
        return metadata
    except Exception as e:
        return None


def save_metadata(dataset_name, env, metadata):
    save_dataset(dataset_name, metadata, f"metadata_{env}")


def filter_env_list_has_data(dataset_name, env_list, prefix):
    good_envs = []
    cut_some = False
    too_small_count = 0
    missing = 0
    for env in env_list:
        filename = "supervised_train_data_env_" + str(env) if prefix == "supervised" else prefix + "_" + str(env)
        path = os.path.join(get_dataset_dir(dataset_name), filename)
        # Check that the data file exists and
        if not os.path.isfile(path):
            missing += 1
            cut_some = True
        elif os.path.getsize(path) < 1000:
            too_small_count += 1
            cut_some = True
        else:
            good_envs.append(env)
    if cut_some:
        print("Filtered out " + str(len(env_list) - len(good_envs)) + " envs because of missing data")
        print(f"   {too_small_count} files too small, {missing} files missing")
    return good_envs


def load_single_env_data(dataset_name, prefix, env):
    env_data = []
    #print ("Data env: ", env)
    try:
        env_data = load_dataset(dataset_name, prefix + "_" + str(env))
    except ImportError as err:
        print(err)
    except OSError:
        print("Data for env " + str(env) + "unavailable! Skipping..")
        print(sys.exc_info()[0])
    return env_data


def load_single_env_supervised_data(dataset_name, env):
    env_data = []
    #print ("Data env: ", env)
    try:
        env_data = load_dataset(dataset_name, "supervised_train_data_env_" + str(env))
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
            md = sample["metadata"] if "meatadata" in sample else sample
            if md["seg_idx"] != seg_idx:
                output_segs.append(env_rollout[prev:i])
                prev=i
                seg_idx = md["seg_idx"]
        if prev < len(env_rollout) - 1:
            output_segs.append(env_rollout[prev:len(env_rollout) - 1])
    return output_segs


def load_multiple_env_data(dataset_name, single_proc=False, split_segments=False):
    ddir = get_dataset_dir(dataset_name)
    return load_multiple_env_data_from_dir(ddir, single_proc, split_segments)


def load_multiple_env_data_from_dir(data_dir, single_proc=False, split_segments=False):
    all_files = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, fname) for fname in all_files]
    if single_proc:
        env_datas = [load_dataset_from_path(p, False) for p in all_paths]
    else:
        pool = Pool(processes=20)
        env_datas = pool.map(load_dataset_from_path, all_paths)
    if split_segments:
        env_datas = split_in_segments(env_datas)
    return env_datas


def load_supervised_data(dataset_name, max_envs=None, split_segments=False):
    train_i, dev_i, test_i, _ = get_all_instructions(max_envs)
    train_env_ids = get_env_ids(train_i)
    dev_env_ids = get_env_ids(dev_i)

    pool = Pool(processes=20)

    # Load dataset for each env in parallel
    train_env_data = pool.map(partial(load_single_env_data, dataset_name, "supervised_train_data_env"), train_env_ids)
    dev_env_data = pool.map(partial(load_single_env_data, dataset_name, "supervised_train_data_env"), dev_env_ids)
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


def file_exists(name, dataset_name):
    full_path = os.path.join(get_dataset_dir(dataset_name), str(name))
    data_exists = os.path.exists(full_path)
    full_path = os.path.join(get_model_dir(), str(name))
    model_exists = os.path.exists(full_path)
    return data_exists or model_exists

import os
import random
import numpy as np
from pprint import pprint

from data_io.paths import get_dataset_dir
from data_io.instructions import get_restricted_env_id_lists

import parameters.parameter_server as P

splits = [
    10,
    25,
    50
]


def first_choice_ok(choice):
    if len(choice) == 0:
        return False
    charray = np.asarray(choice)
    c5 = (charray / 5).astype(np.int64)
    copies = [c5[c] in c5[:c] or c in c5[c+1:] for c in range(len(c5))]
    num_copies = np.asarray(copies).sum()
    print("num_envs_same: ", num_copies)
    if num_copies > 0:
        return False
    return True


if __name__ == "__main__":
    P.initialize_experiment()
    real_data_dir = get_dataset_dir("real")
    files = os.listdir(real_data_dir)

    available_env_ids = set([int(f.split("supervised_train_data_env_")[1]) for f in files])
    train_ids, dev_ids, test_ids = get_restricted_env_id_lists()
    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    test_ids = set(test_ids)

    avail_train_ids = list(train_ids.intersection(available_env_ids))
    avail_dev_ids = list(dev_ids.intersection(available_env_ids))
    avail_test_ids = list(test_ids.intersection(available_env_ids))

    print(f"Making subsets from total envs: {len(avail_train_ids)}")

    splits_out = {}

    choice = []
    prev_split = splits[0]
    while not first_choice_ok(choice):
        choice = random.sample(avail_train_ids, prev_split)
    splits_out[str(prev_split)] = choice + avail_dev_ids + avail_test_ids

    for split in splits[1:]:
        add_k = split - prev_split
        added = random.sample(avail_train_ids, add_k)
        choice = choice + added
        splits_out[str(split)] = choice + avail_dev_ids + avail_test_ids

    print("\n\n\n\n")
    pprint(splits_out)


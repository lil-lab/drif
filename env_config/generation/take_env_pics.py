import os
import time
import numpy as np

from scipy.misc import imsave

from data_io import paths
from data_io.instructions import get_all_instructions
from pomdp.pomdp_interface import PomdpInterface
import parameters.parameter_server as P

NUM_WORKERS = 1
SMALL_ENV = True

"""
This file is used to generate the overhead-view pictures of the environment that are
useful for presentation and debugging purposes.

The images are saved in config_dir/config/env_img
"""

def take_pics():
    P.initialize_experiment()
    train_i, dev_i, test_i, _ = get_all_instructions()
    all_instructions = {**train_i, **dev_i, **test_i}

    save_dir = paths.get_env_image_path(0)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    keylist = list(all_instructions.keys())

    envs = [PomdpInterface(instance_id=i) for i in range(0, NUM_WORKERS)]
    env_id_splits = [[] for _ in range(NUM_WORKERS)]
    keylist = [6825]

    for i, key in enumerate(keylist):
        env_id_splits[i % NUM_WORKERS].append(key)


    time.sleep(1.0)
    for i in range(len(keylist)):

        d = False
        # For each worker, start the correct env
        for w in range(NUM_WORKERS):
            if i >= len(env_id_splits[w]):
                continue
            env_id = env_id_splits[w][i]
            # FIXME: :This assumes that there is only 1 instruction set per env!
            fname = paths.get_env_image_path(env_id)
            if os.path.isfile(fname):
                print("Img exists: " + fname)
                continue

            d = True
            instruction_set = all_instructions[env_id][0]
            envs[w].set_environment(env_id, instruction_set["instructions"], fast=True)
            print("setting env on worker " + str(w) + " iter " + str(i) + " env_id: " + str(env_id))

        # Then for each worker, take a picture and save it
        if d:
            time.sleep(0.1)
        for w in range(NUM_WORKERS):
            if i >= len(env_id_splits[w]):
                continue
            env_id = env_id_splits[w][i]
            fname = paths.get_env_image_path(env_id)
            if os.path.isfile(fname):
                print("Img exists: " + fname)
                continue
            envs[w].snap_birdseye(fast=True, small_env=SMALL_ENV)
            image = envs[w].snap_birdseye(fast=True, small_env=SMALL_ENV)
            image = np.flip(image, 0)
            imsave(fname, image)
            print("saving pic on worker " + str(w) + " iter " + str(i) + " env_id: " + str(env_id))


if __name__ == "__main__":
    take_pics()
import os
import shutil
import sys
import ray
import multiprocessing

from data_io.paths import get_dataset_dir
from data_io.instructions import get_restricted_env_id_lists
from mains.train.sureal import conf
from mains.train.sureal import rl_process
from mains.train.sureal import sl_process
from mains.train.sureal import kvstore

import parameters.parameter_server as P

DEBUG_SUP = False
DEBUG_RL = False


def train_sureal():
    params_name = sys.argv[1]
    P.initialize_experiment(params_name)

    local_ray = P.get("Setup::local_ray", False)
    ray.init(num_cpus=8,
             num_gpus=2,
             memory=40 * (1024**3),
             object_store_memory=8 * (1024**3),
             local_mode=local_ray)

    rlsup_params = P.get_current_parameters()["SuReAL"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")
    run_name = P.get_current_parameters()["Setup"]["run_name"]

    # Trigger rebuild if necessary before going into all the threads and processes
    _ = get_restricted_env_id_lists()
    _ = get_restricted_env_id_lists(ignore_min_augment_len=True)

    if sim_seed_dataset:
        conf.copy_seed_dataset(from_dataset=sim_seed_dataset, to_dataset=conf.rl_dataset_name(run_name))

    kv_store = kvstore.KVStore.remote()
    sl_process.sureal_supervised_learning.remote(kv_store, params_name, send_model_only=False)
    rl_process.train_rl_worker(kv_store, params_name)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    train_sureal()

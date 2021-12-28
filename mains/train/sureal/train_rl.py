import sys
import ray

from data_io.instructions import get_restricted_env_id_lists
from mains.train.sureal import rl_process
from mains.train.sureal import kvstore

import parameters.parameter_server as P

DEBUG_SUP = False
DEBUG_RL = False


def train_rl():
    params_name = sys.argv[1]
    P.initialize_experiment(params_name)

    local_ray = P.get("Setup::local_ray", False)
    ray.init(num_cpus=6,
             num_gpus=2,
             memory=40 * (1024**3),
             object_store_memory=8 * (1024**3),
             local_mode=local_ray)

    # Trigger rebuild of instruction cache if necessary before going into all the threads and processes
    _ = get_restricted_env_id_lists()
    _ = get_restricted_env_id_lists(ignore_min_augment_len=True)

    kv_store = kvstore.KVStore.remote()
    kv_store.put.remote("stage1_model_file", False)
    rl_process.train_rl_worker(kv_store, params_name, save_rollouts=False)


if __name__ == "__main__":
    train_rl()

from multiprocessing import Pool
import numpy as np

from data_io.instructions import get_all_env_id_lists
from data_io.train_data import load_single_env_supervised_data

import parameters.parameter_server as P

NUM_WORKERS = 10


def count_avg_num_steps_on_env(env):

    try:
        env_data = load_single_env_supervised_data(env)
        num_steps = len(env_data)
        step_nums = []
        fwd_vel = 0
        step_cnt = 0
        seg_idx=0

        for sample in env_data:
            fwd_vel += sample.action[0]
            step_cnt += 1
            if sample.metadata["seg_idx"] != seg_idx:
                step_nums.append(step_cnt)
                seg_idx = sample.metadata["seg_idx"]
                step_cnt = 0

        step_nums.append(step_cnt)

        if len(step_nums) == 0:
            avg_num_steps = 0
        else:
            avg_num_steps = np.asarray(step_nums).sum() / (len(step_nums) + 1e-9)

        avg_fwd_vel = fwd_vel / (num_steps + 1e-9)

        stats = {
            "num_steps": avg_num_steps,
            "fwd_vel": avg_fwd_vel
        }
        return stats

    except Exception as e:
        print(e)


def collate_stats(stats):
    total_steps = 0
    weighted_sum_vel = 0
    for stat in stats:
        if stat is None:
            print("NONESTAT!")
            continue
        total_steps += stat["num_steps"]
        weighted_sum_vel += stat["num_steps"] * stat["fwd_vel"]
    avg_vel = weighted_sum_vel / total_steps
    avg_steps = total_steps / len(stats)
    return avg_vel, avg_steps


def count_avg_num_steps_on_data(env_list):
    setup = P.get_current_parameters()["Setup"]

    env_list = env_list[setup["env_range_start"]:]
    env_list = env_list[:setup["max_envs"]]

    pool = Pool(NUM_WORKERS)
    step_lengths = pool.map(count_avg_num_steps_on_env, env_list)
    pool.close()
    pool.join()

    avg_vel, avg_steps = collate_stats(step_lengths)
    print ("Average fwd velocity: " + str(avg_vel))
    print ("Average num fwd steps: " + str(avg_steps))


def count_avg_num_steps():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    train_envs, dev_envs, test_envs = get_all_env_id_lists(setup["max_envs"])

    count_avg_num_steps_on_data(train_envs)


if __name__ == "__main__":
    count_avg_num_steps()
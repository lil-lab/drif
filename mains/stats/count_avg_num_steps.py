from multiprocessing import Pool
import numpy as np

from data_io.instructions import get_restricted_env_id_lists
from data_io.train_data import load_single_env_supervised_data

import parameters.parameter_server as P

NUM_WORKERS = 10


def count_avg_num_steps_on_env(env):

        env_data = load_single_env_supervised_data("real", env)
        print("Running stats for env: ", env)
        num_steps = len(env_data)
        step_nums = []
        fwd_vel = 0
        ang_vel = 0
        step_cnt = 0
        seg_idx=0

        for sample in env_data:
            fwd_vel += sample["ref_action"][0]
            ang_vel += sample["ref_action"][2]
            step_cnt += 1
            if sample["metadata"]["seg_idx"] != seg_idx:
                step_nums.append(step_cnt)
                seg_idx = sample["metadata"]["seg_idx"]
                step_cnt = 0

        step_nums.append(step_cnt)

        if len(step_nums) == 0:
            avg_num_steps = 0
        else:
            avg_num_steps = np.asarray(step_nums).sum() / (len(step_nums) + 1e-9)

        avg_fwd_vel = fwd_vel / (num_steps + 1e-9)
        avg_ang_vel = ang_vel / (num_steps + 1e-9)

        stats = {
            "num_steps": avg_num_steps,
            "fwd_vel": avg_fwd_vel,
            "avg_ang_vel": avg_ang_vel
        }
        return stats

        #print(e)


def collate_stats(stats):
    total_steps = 0
    weighted_sum_vel = 0
    weighted_sum_yawrate = 0
    for stat in stats:
        if stat is None:
            print("NONESTAT!")
            continue
        total_steps += stat["num_steps"]
        weighted_sum_vel += stat["num_steps"] * stat["fwd_vel"]
        weighted_sum_yawrate += stat["num_steps"] * stat["avg_ang_vel"]
    avg_vel = weighted_sum_vel / total_steps
    avg_yawrate = weighted_sum_yawrate / total_steps
    avg_steps = total_steps / len(stats)
    return avg_vel, avg_yawrate, avg_steps


def count_avg_num_steps_on_data(env_list):
    #setup = P.get_current_parameters()["Setup"]

    #env_list = env_list[setup["env_range_start"]:]
    #env_list = env_list[:setup["max_envs"]]
    #env_list = get_restricted_env_id_lists()

    #pool = Pool(NUM_WORKERS)
    #step_lengths = pool.map(count_avg_num_steps_on_env, env_list)
    #pool.close()
    #pool.join()

    step_lengths = [count_avg_num_steps_on_env(e) for e in env_list]

    avg_vel, avg_yawrate, avg_steps = collate_stats(step_lengths)
    print("Average fwd velocity: " + str(avg_vel))
    print("Average yaw rate: " + str(avg_yawrate))
    print("Average num fwd steps: " + str(avg_steps))


def count_avg_num_steps():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    P.get_current_parameters()["Data"]["locking"] = False
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    train_envs = [e for e in train_envs if e >= 6000]

    count_avg_num_steps_on_data(train_envs)


if __name__ == "__main__":
    count_avg_num_steps()

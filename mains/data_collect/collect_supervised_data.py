import itertools
import ray

from data_io.paths import get_supervised_data_filename
from data_io.models import load_model
from rollout.simple_parallel_rollout import SimpleParallelPolicyRoller
from rollout.simple_rollout import SimplePolicyRoller
from data_io.instructions import get_restricted_env_id_lists, get_segs_available_for_env
from data_io.train_data import save_dataset, file_exists

from drones.airsim_interface.droneController import killAirSim

import parameters.parameter_server as P


def filter_uncollected_envs(dataset_name,env_list):
    uncollected = []
    excluded = []
    for env in env_list:
        filename = get_supervised_data_filename(env)
        if not file_exists(filename, dataset_name):
            uncollected.append(env)
        else:
            excluded.append(env)
    print("Including envs: ", uncollected)
    print("Excluding envs: ", excluded)
    return uncollected


def collect_data_on_env_list(env_list):
    setup = P.get_current_parameters()["Setup"]
    dataset_name = P.get_current_parameters()["Data"]["dataset_name"]

    local_ray = P.get("Setup::local_ray", False)
    ray.init(num_cpus=12,
             num_gpus=1,
             memory=40 * (1024**3),
             local_mode=local_ray,
             ignore_reinit_error=True)

    oracle, _ = load_model("oracle")

    if setup["num_workers"] > 1:
        roller = SimpleParallelPolicyRoller(policy=oracle,
                                            num_workers=setup["num_workers"],
                                            oracle=oracle, device=None,
                                            dataset_save_name=dataset_name,
                                            restart_every_n=1000,
                                            no_reward=True)
    else:
        roller = SimplePolicyRoller(instance_id=0,
                                    real_drone=setup["real_drone"],
                                    policy=oracle,
                                    oracle=oracle,
                                    dataset_save_name=dataset_name,
                                    no_reward=True)

    group_size = P.get_current_parameters()["Data"].get("collect_n_at_a_time", 5)

    # Collect training data
    print("Collecting training data!")

    if setup.get("env_range_start") > 0:
        env_list = [e for e in env_list if e >= setup["env_range_start"]]
    if setup.get("env_range_end") > 0:
        env_list = [e for e in env_list if e < setup["env_range_end"]]

    env_list = env_list[:setup["max_envs"]]
    env_list = filter_uncollected_envs(dataset_name, env_list)

    group_size = setup["num_workers"] * group_size

    kill_airsim_every_n_rounds = 50
    round_counter = 0

    for i in range(0, len(env_list), group_size):
        # Rollout on group_size envs at a time. After each group, land the drone and save the data
        round_envs = env_list[i:]
        round_envs = round_envs[:group_size]
        round_segs = [get_segs_available_for_env(e, 0) for e in round_envs]
        round_envs = [[e] * len(segs) for e, segs in zip(round_envs, round_segs)]

        round_segs = list(itertools.chain(*round_segs))
        round_envs = list(itertools.chain(*round_envs))

        rollouts = roller.rollout_segments(round_envs, round_segs)

        #roll_params.setEnvList(round_envs)
        #env_datas = roller.roll_out_policy(roll_params)
        #for j in range(len(rollouts)):
        #    env_data = rollouts[j]
        #    if len(env_data) > 0:
        #        # KeyError: 0:
        #        env_id = env_data[0]["env_id"]
        #        filename = get_supervised_data_filename(env_id)
        #        save_dataset(dataset_name, env_data, filename)
        #    else:
        #        print("Empty rollout!")
        # AirSim tends to clog up and become slow. Kill it every so often to restart it.
        round_counter += 1
        if round_counter > kill_airsim_every_n_rounds:
            round_counter = 0
            killAirSim(do_kill=True)


def collect_supervised_data():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()#

    if P.get_current_parameters()["Setup"].get("env_set") == "train":
        print("Collecting data for training envs")
        collect_data_on_env_list(train_envs)
    elif P.get_current_parameters()["Setup"].get("env_set") == "dev":
        print("Collecting data for dev envs")
        collect_data_on_env_list(dev_envs)
    else:
        print("Collecting data for both training and dev envs")
        collect_data_on_env_list(train_envs)
        collect_data_on_env_list(dev_envs)


if __name__ == "__main__":
    collect_supervised_data()

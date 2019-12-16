from data_io.paths import get_supervised_data_filename
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams
from data_io.instructions import get_restricted_env_id_lists
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

    if setup["num_workers"] > 1:
        roller = ParallelPolicyRoller(num_workers=setup["num_workers"])
    else:
        roller = PolicyRoller()

    group_size = P.get_current_parameters()["Data"].get("collect_n_at_a_time", 5)

    wrong_paths_p = P.get_current_parameters()["Rollout"].get("wrong_path_p", 0.0)

    # setSetupName is important - it allows the threads to load the same json file and initialize stuff correctly
    roll_params = RollOutParams() \
        .setModelName("oracle") \
        .setRunName(setup["run_name"]) \
        .setSetupName(P.get_setup_name()) \
        .setSavePlots(False) \
        .setSaveSamples(False) \
        .setSegmentLevel(False) \
        .setPlot(False) \
        .setBuildTrainData(False) \
        .setRealDrone(setup["real_drone"]) \
        .setCuda(setup["cuda"]) \
        .setSegmentReset("always") \
        .setWrongPathP(wrong_paths_p)

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
        roll_params.setEnvList(round_envs)
        env_datas = roller.roll_out_policy(roll_params)
        for j in range(len(env_datas)):
            env_data = env_datas[j]
            if len(env_data) > 0:
                # KeyError: 0:
                env_id = env_data[0]["env_id"]
                filename = get_supervised_data_filename(env_id)
                save_dataset(dataset_name, env_data, filename)
            else:
                print("Empty rollout!")
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

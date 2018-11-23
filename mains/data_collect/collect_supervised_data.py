from data_io.paths import get_supervised_data_filename
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out_params import RollOutParams
from data_io.instructions import get_all_env_id_lists
from data_io.train_data import save_dataset, file_exists

import parameters.parameter_server as P


def filter_uncollected_envs(env_list):
    uncollected = []
    excluded = []
    for env in env_list:
        filename = get_supervised_data_filename(env)
        if not file_exists(filename):
            uncollected.append(env)
        else:
            excluded.append(env)
    print("Including envs: ", uncollected)
    print("Excluding envs: ", excluded)
    return uncollected


def collect_data_on_env_list(env_list):
    setup = P.get_current_parameters()["Setup"]

    roller = ParallelPolicyRoller(num_workers=setup["num_workers"], reduce=False)

    roll_params = RollOutParams() \
        .setModelName("oracle") \
        .setRunName(setup["run_name"]) \
        .setSetupName(P.get_setup_name()) \
        .setSavePlots(True) \
        .setSaveSamples(False) \
        .setPlot(False) \
        .setBuildTrainData(False) \
        .setCuda(setup["cuda"]) \
        .setSegmentReset("always")

    # Collect training data
    print("Collecting training data!")

    env_list = env_list[setup["env_range_start"]:]
    env_list = env_list[:setup["max_envs"]]
    env_list = filter_uncollected_envs(env_list)

    group_size = setup["num_workers"] * 10

    for i in range(0, len(env_list), group_size):
        round_envs = env_list[i:]
        round_envs = round_envs[:group_size]
        roll_params.setEnvList(round_envs)
        env_datas = roller.roll_out_policy(roll_params)
        for j in range(len(env_datas)):
            env_data = env_datas[j]
            if len(env_data) > 0:
                env_id = env_data[0]["metadata"]["env_id"]
                filename = get_supervised_data_filename(env_id)
                save_dataset(env_data, filename)
            else:
                print("Empty rollout!")


def collect_supervised_data():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    train_envs, dev_envs, test_envs = get_all_env_id_lists(setup["max_envs"])#

    collect_data_on_env_list(train_envs)
    collect_data_on_env_list(dev_envs)


if __name__ == "__main__":
    collect_supervised_data()
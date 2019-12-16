from data_io.instructions import get_restricted_env_id_lists

from learning.datasets.rss_noisy_poses import save_noisy_poses
from learning.inputs.pose import get_pose_noise_np

from parameters.parameter_server import initialize_experiment, get_current_parameters


def build_noisy_pose_data():
    """
    Randomly sample pose noise for every observation in every environment for the RSS experiment with noisy poses.
    This needs to be pre-computed once before training to simulate the noise being measured during trajectory collection.
    If we were to randomize poses during training, that would be akin to regularization,
    which could actually improve instead of hurt performance.
    :return:
    """
    initialize_experiment()
    params = get_current_parameters()
    setup_params = params["Setup"]

    train_envs, dev_envs, test_envs = get_restricted_env_id_lists(
        max_envs=setup_params["max_envs"],
        prune_ambiguous=setup_params["prune_ambiguous"])

    envs = dev_envs + train_envs + test_envs
    print("Num envs:" + str(len(envs)))

    pos_noise = params["Data"]["noisy_pos_variance"]
    rot_noise = params["Data"]["noisy_rot_variance"]

    noisy_poses = get_pose_noise_np(setup_params["max_envs"], setup_params["trajectory_length"], pos_noise, rot_noise)
    save_noisy_poses(noisy_poses)
    print("saved noisy poses for " + str(setup_params["max_envs"]) + " envs")


if __name__ == "__main__":
    build_noisy_pose_data()
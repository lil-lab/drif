import os
from learning.models.visualization.viz_html_rpn_fs_stage1_bidomain import visualize_model_dashboard_from_rollout
import data_io.paths
from data_io.train_data import load_multiple_env_data_from_dir, split_into_segs
import parameters.parameter_server as P


def visualize_rollouts_rpn_fspvn():
    P.initialize_experiment()
    eval_dataset_name = data_io.paths.get_eval_tmp_dataset_name(P.get("Setup::model"), P.get("Setup::run_name"))
    eval_dataset_path = data_io.paths.get_dataset_dir(eval_dataset_name)
    if not os.path.exists(eval_dataset_path):
        raise FileNotFoundError(
            f"Dataset does not exist at: {eval_dataset_path}. Check your Setup::model and Setup::run_name parameters")

    eval_dataset = split_into_segs(load_multiple_env_data_from_dir(eval_dataset_path, single_proc=True))
    for rollout in eval_dataset:
        visualize_model_dashboard_from_rollout(rollout, P.get("Setup::run_name"))


if __name__ == "__main__":
    visualize_rollouts_rpn_fspvn()
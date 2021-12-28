import os
from evaluation.evaluate_nl import DataEvalNL
from learning.models.visualization.viz_html_rpn_fs_stage1_bidomain import visualize_model_dashboard_from_rollout
from learning.models.visualization.viz_graphics_rpn_fs_stage1_bidomain import generate_graphics_visualizations_from_rollout
from data_io.instructions import get_correct_eval_env_id_list, get_all_instructions
from data_io.paths import get_eval_tmp_dataset_name, get_results_dir
from data_io.train_data import load_multiple_env_data

import parameters.parameter_server as P

GRAPHICS = True


def log(text, logdir):
    print(text)
    os.makedirs(logdir, exist_ok=True)
    with open(f"{logdir}/log.txt", "a") as fp:
        fp.write(text + "\n")


def gen_dashboards_saved_rollouts():
    params = P.get_current_parameters()
    setup = params["Setup"]
    model_name = setup["model"]
    run_name = setup["run_name"]
    eval_dname = get_eval_tmp_dataset_name(model_name, run_name)

    eval_envs = set(list(sorted(get_correct_eval_env_id_list())))
    rollouts = load_multiple_env_data(eval_dname, single_proc=True, split_segments=True)
    present_envs = set([rollout[0]["env_id"] for rollout in rollouts if len(rollout) > 0])
    missing_envs = eval_envs - present_envs

    logdir = get_results_dir(run_name)

    if len(missing_envs) > 0:
        print(f"Warning! {len(missing_envs)} envs missing: {missing_envs}")
        #sys.exit(1)

    log("", logdir)
    log("--------------------------------------------------------------------------------------------", logdir)
    log(f"Generating dashboards for rollouts for run {run_name}", logdir)
    log(f"   using dataset {eval_dname}", logdir)
    log(f"   missing envs {missing_envs}", logdir)
    log("--------------------------------------------------------------------------------------------", logdir)

    for rollout in rollouts:
        if GRAPHICS:
            #try:
                generate_graphics_visualizations_from_rollout(rollout, run_name)
            #except Exception as e:
            #    print("EXCEPTION")
            #if len(rollout) > 0:
            #    break
        else:
            visualize_model_dashboard_from_rollout(rollout, run_name)


if __name__ == "__main__":
    P.initialize_experiment()
    gen_dashboards_saved_rollouts()

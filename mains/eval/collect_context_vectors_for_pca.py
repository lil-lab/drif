import os
import numpy as np
from evaluation.evaluate_nl import DataEvalNL
from learning.models.visualization.viz_html_rpn_fs_stage1_bidomain import visualize_model_dashboard_from_rollout
from learning.models.visualization.viz_graphics_rpn_fs_stage1_bidomain import generate_graphics_visualizations_from_rollout, process_rollout_with_stage1_model
from data_io.instructions import get_correct_eval_env_id_list, get_all_instructions
from data_io.paths import get_eval_tmp_dataset_name, get_results_dir
from data_io.train_data import load_multiple_env_data

from sklearn.decomposition import PCA

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

    vectors = []

    for i, rollout in enumerate(rollouts):
        print(f"Rollout {i}/{len(rollouts)}")
        if len(rollout) == 0:
            continue
        ms = process_rollout_with_stage1_model(rollout)
        context_embeddings = ms.get("obj_ref_context_embeddings")
        for j in range(context_embeddings.shape[1]):
            vec = context_embeddings[0, j, :]
            vectors.append(vec.detach().cpu().numpy())

    X = np.stack(vectors)
    NORM = False
    if NORM:
        mean = X.mean(0)
        X = X - mean
        std = X.std(0)
        X = X / std
    pca = PCA(n_components=3)
    pca.fit(X)
    components = pca.components_ # 3 x C matrix
    print("--------------------------------------------")
    print("--------------------------------------------")
    print(components.tolist())
    print("--------------------------------------------")
    if NORM:
        print(mean.tolist())
        print("--------------------------------------------")
        print(std.tolist())
        print("--------------------------------------------")
        print("--------------------------------------------")
        print(f"components = torch.tensor({components.tolist()})"
              f"mean = torch.tensor({mean.tolist()}"
              f"std = torch.tensor({std.tolist()}")
    else:
        print(f"components = torch.tensor({components.tolist()})")


if __name__ == "__main__":
    P.initialize_experiment()
    gen_dashboards_saved_rollouts()

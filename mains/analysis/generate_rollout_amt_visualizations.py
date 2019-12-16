import os
from data_io.models import load_model
from data_io.instructions import get_restricted_env_id_lists
from data_io.train_data import load_single_env_from_dataset
from data_io.paths import get_rollout_viz_dir, get_eval_tmp_dataset_name

import parameters.parameter_server as P

from visualization import Presenter
from rollout_vizualizer import RolloutVisualizer


def split_into_segs(env_data):
    segs = []
    seg = []
    seg_idx = -1
    for sample in env_data:
        if sample["seg_idx"] != seg_idx:
            if len(seg) > 0:
                segs.append(seg)
            seg = [sample]
            seg_idx = sample["seg_idx"]
        else:
            seg.append(sample)
    segs.append(seg)
    return segs


# Supervised learning parameters
def generate_rollout_amt_visualizations():

    setup = P.get_current_parameters()["Setup"]

    dataset_name = setup.get("viz_dataset_name") or get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
    print(f"Generating AMT animations for dataset: {dataset_name}")
    pic_domain = "sim"
    data_domain = "real"
    # Some quick params. TODO: Bring this into json
    viz_params = {
        "draw_drone": True,
        "draw_trajectory": True,
        "draw_fov": True,
        "include_vdist": False,
        "include_layer": None,
        "include_instr": False
    }

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # TODO: Grab the correct env list
    env_list = test_envs

    viz = RolloutVisualizer(resolution=400)
    base_dir = os.path.join(get_rollout_viz_dir(), f"{dataset_name}-{data_domain}")
    os.makedirs(base_dir, exist_ok=True)

    for env_id in env_list:
        try:
            env_data = load_single_env_from_dataset(dataset_name, env_id, "supervised")
        except FileNotFoundError as e:
            print(f"Skipping env: {env_id}")
            continue
        if len(env_data) == 0:
            print(f"Skipping env: {env_id}. Rollout exists but is EMPTY!")
            continue
        segs = split_into_segs(env_data)
        for seg in segs:
            seg_idx = seg[0]["seg_idx"]
            seg_name = f"{env_id}:0:{seg_idx}-{data_domain}"
            gif_filename = f"{seg_name}-roll.gif"
            instr_filename = f"{seg_name}-instr.txt"

            # Generate and save gif
            frames = viz.top_down_visualization(env_id, seg_idx, seg, pic_domain, viz_params)
            print("Saving GIF")
            viz.presenter.save_gif(frames, os.path.join(base_dir, gif_filename), fps=5.0)

            # Save instruction
            with open(os.path.join(base_dir, instr_filename), "w") as fp:
                fp.write(seg[0]["instruction"])


        print("ding")


if __name__ == "__main__":
    P.initialize_experiment()
    generate_rollout_amt_visualizations()

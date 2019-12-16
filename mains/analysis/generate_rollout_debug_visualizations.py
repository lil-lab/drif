import os
import shutil
from data_io.instructions import get_restricted_env_id_lists
from data_io.train_data import load_single_env_from_dataset
from data_io.paths import get_rollout_debug_viz_dir, get_eval_tmp_dataset_name, get_rollout_video_dir
from copy import deepcopy

import parameters.parameter_server as P

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


Y_FMT = ("mp4", "png")
D_FMT = ("mp4")


def save_frames(viz, frames, base_path, fps=5.0, start_lag=0.0, end_lag=0.0, formats=Y_FMT):
    print(f"Saving files: {base_path}")
    if start_lag > 0:
        frames = [frames[0]] * int(fps * start_lag) + frames
    if end_lag > 0:
        frames = frames + [frames[-1]] * int(fps * end_lag)
    if "gif" in formats:
        viz.presenter.save_gif(frames, f"{base_path}.gif", fps=5.0)
    if "mp4" in formats:
        viz.presenter.save_video(frames, f"{base_path}.mp4", fps=fps)
    if "png" in formats:
        viz.presenter.save_frames(frames, f"{base_path}-frames")


# Supervised learning parameters
def generate_rollout_debug_visualizations():
    setup = P.get_current_parameters()["Setup"]

    dataset_name = setup.get("viz_dataset_name") or get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
    domain = setup.get("viz_domain") or ("real" if setup.get("real_drone") else "sim")
    run_name = setup.get("original_run_name") or setup.get("run_name")
    specific_envs = setup.get("only_specific_envs")

    # For collecting information for visualization examples
    specific_segments = [
        # running example
        (6827, 0, 4),
        # successful examples
        (6169, 0, 9),
        (6825, 0, 8),
        (6857, 0, 9),
        # failure examples
        (6169, 0, 2),
        (6299, 0, 9),
        (6634, 0, 8),
        (6856, 0, 9),
        (6857, 0, 8),
    ]
    specific_segments += [
        # good sim, lousy real
        (6419, 0, 5),
        (6569, 0, 6),
        (6634, 0, 6),
        (6917, 0, 7),
    ]
    specific_envs = [s[0] for s in specific_segments]

    # Generate all
    #specific_envs = list(range(6000, 7000, 1))
    #specific_segments = None

    # Some quick params. TODO: Bring this into json
    viz_params = {
        "ego_vdist": False,
        "draw_landmarks": False,
        "draw_topdown": True,
        "draw_drone": True,
        "draw_trajectory": True,
        "draw_fov": False,
        "include_vdist": False,
        "include_layer": None,
        "include_instr": False
    }

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # TODO: Grab the correct env list
    env_list = dev_envs

    viz = RolloutVisualizer(resolution=576)
    base_dir = os.path.join(get_rollout_debug_viz_dir(), f"{dataset_name}-{domain}")
    os.makedirs(base_dir, exist_ok=True)

    for env_id in env_list:
        if specific_envs and env_id not in specific_envs:
            print("Skipping", env_id)
            continue
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
            lag_start = 1.5
            end_lag = 1.5
            seg_idx = seg[0]["seg_idx"]
            if specific_segments and (env_id, 0, seg_idx) not in specific_segments:
                continue
            seg_name = f"{env_id}:0:{seg_idx}-{domain}"
            gif_filename = f"{seg_name}-roll"
            instr_filename = f"{seg_name}-instr.txt"
            this_dir = os.path.join(base_dir, gif_filename)
            os.makedirs(this_dir, exist_ok=True)
            base_path = os.path.join(this_dir, gif_filename)
            if os.path.exists(os.path.join(this_dir, instr_filename)):
                continue

            # Animation with just the drone
            frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, viz_params)
            save_frames(viz, frames, f"{base_path}-exec", fps=5.0, start_lag=lag_start, end_lag=end_lag, formats=Y_FMT)

            # Save instructionto
            with open(os.path.join(this_dir, instr_filename), "w") as fp:
                fp.write(seg[0]["instruction"])

            # Animation of action
            frames = viz.action_visualization(env_id, seg_idx, seg, domain, "action")
            save_frames(viz, frames, f"{base_path}-action", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Animation of actions
            #action_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "action", scale=4)
            #save_frames(viz, action_frames, f"{base_path}-action", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Generate and save gif
            # Bare top-down view
            mod_params = deepcopy(viz_params)
            mod_params["draw_drone"] = False
            mod_params["draw_trajectory"] = False
            frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
            save_frames(viz, frames, f"{base_path}-top-down", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            mod_params["draw_drone"] = True
            mod_params["draw_trajectory"] = False
            frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
            save_frames(viz, frames, f"{base_path}-top-down-drn", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Egocentric visitation distributions
            vdist_r_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "v_dist_r_inner")
            save_frames(viz, vdist_r_frames, f"{base_path}-ego-vdist", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Map struct
            map_struct_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "map_struct")
            save_frames(viz, map_struct_frames, f"{base_path}-ego-map-struct", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Egocentric observation mask
            ego_obs_mask_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "ego_obs_mask")
            save_frames(viz, ego_obs_mask_frames, f"{base_path}-ego-obs-mask", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            def save_map_permutations(file_prefix, incl_layer):
                mod_params = deepcopy(viz_params)
                if incl_layer == "vdist":
                    mod_params["include_vdist"] = True
                else:
                    mod_params["include_layer"] = incl_layer
                print(f"GENERATING: {file_prefix}")
                # Non-overlaid, without trajectory
                mod_params["draw_drone"] = False
                mod_params["draw_topdown"] = False
                mod_params["draw_trajectory"] = False
                frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
                save_frames(viz, frames, f"{file_prefix}", fps=5.0, start_lag=lag_start, end_lag=end_lag, formats=Y_FMT)

                print(f"GENERATING: {file_prefix}-ov")
                # Overlaid, without trajectory
                mod_params["draw_topdown"] = True
                frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
                save_frames(viz, frames, f"{file_prefix}-ov", fps=5.0, start_lag=lag_start, end_lag=end_lag, formats=D_FMT)

                print(f"GENERATING: {file_prefix}-ov-path")
                # Overlaid, with trajectory
                mod_params["draw_drone"] = True
                mod_params["draw_trajectory"] = True
                frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
                save_frames(viz, frames, f"{file_prefix}-ov-path", fps=5.0, start_lag=lag_start, end_lag=end_lag, formats=Y_FMT)

                print(f"GENERATING: {file_prefix}-path")
                # Non-overlaid, with trajectory
                mod_params["draw_topdown"] = False
                mod_params["draw_drone"] = True
                mod_params["draw_trajectory"] = True
                frames = viz.top_down_visualization(env_id, seg_idx, seg, domain, mod_params)
                save_frames(viz, frames, f"{file_prefix}-path", fps=5.0, start_lag=lag_start, end_lag=end_lag, formats=D_FMT)

            save_map_permutations(f"{base_path}-vdist", "vdist")

            save_map_permutations(f"{base_path}-semantic-map", "S_W")

            save_map_permutations(f"{base_path}-semantic-map-gray", "S_W_Gray")

            save_map_permutations(f"{base_path}-proj-features", "F_W")

            save_map_permutations(f"{base_path}-grounding-map", "R_W")

            save_map_permutations(f"{base_path}-grounding-map-gray", "R_W_Gray")

            save_map_permutations(f"{base_path}-mask", "M_W")

            save_map_permutations(f"{base_path}-accum-mask", "M_W_accum")

            save_map_permutations(f"{base_path}-accum-mask-inv", "M_W_accum_inv")

            # Animation of FPV features
            fpv_feature_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "F_C")
            save_frames(viz, fpv_feature_frames, f"{base_path}-features-fpv", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            # Animation of FPV images
            fpv_image_frames = viz.grab_frames(env_id, seg_idx, seg, domain, "image", scale=4)
            save_frames(viz, fpv_image_frames, f"{base_path}-image", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            frames = viz.overlay_frames(fpv_image_frames, fpv_feature_frames)
            save_frames(viz, frames, f"{base_path}-features-fpv-ov", fps=5.0, start_lag=lag_start, end_lag=end_lag)

            num_frames = len(frames)

            # Clip rollout videos to correct rollout duration and re-save
            rollout_dir = get_rollout_video_dir(run_name=run_name)
            if os.path.isdir(rollout_dir):
                print("Processing rollout videos")
                actual_rollout_duration = num_frames / 5.0
                ceiling_clip = viz.load_video_clip(env_id, seg_idx, seg, domain, "ceiling", rollout_dir)
                duration_with_lag = lag_start + actual_rollout_duration + end_lag
                try:
                    if ceiling_clip is not None:
                        if ceiling_clip.duration > duration_with_lag:
                            start = ceiling_clip.duration - end_lag - duration_with_lag
                            ceiling_clip = ceiling_clip.cutout(0, start)
                            #ceiling_clip = ceiling_clip.cutout(duration_with_lag, ceiling_clip.duration)
                        save_frames(viz, ceiling_clip, f"{base_path}-ceiing_cam-clipped", fps=ceiling_clip.fps)
                    corner_clip = viz.load_video_clip(env_id, seg_idx, seg, domain, "corner", rollout_dir)
                    if corner_clip is not None:
                        if corner_clip.duration > actual_rollout_duration + end_lag:
                            start = corner_clip.duration - end_lag - duration_with_lag
                            corner_clip = corner_clip.cutout(0, start)
                            #corner_clip = corner_clip.cutout(duration_with_lag, corner_clip.duration)
                        save_frames(viz, corner_clip, f"{base_path}-corner_cam-clipped", fps=corner_clip.fps)
                except Exception as e:
                    print("Video encoding error! Copying manually")
                    print(e)

                try:
                    in_ceil_file = os.path.join(rollout_dir, f"rollout_ceiling_{env_id}-0-{seg_idx}.mkv")
                    in_corn_file = os.path.join(rollout_dir, f"rollout_corner_{env_id}-0-{seg_idx}.mkv")
                    out_ceil_file = f"{base_path}-ceiling_cam-full.mkv"
                    out_corn_file = f"{base_path}-corner_cam-full.mkv"
                    shutil.copy(in_ceil_file, out_ceil_file)
                    shutil.copy(in_corn_file, out_corn_file)
                except Exception as e:
                    print("Failed copying videos! SKipping")

        print("ding")


if __name__ == "__main__":
    P.initialize_experiment()
    generate_rollout_debug_visualizations()

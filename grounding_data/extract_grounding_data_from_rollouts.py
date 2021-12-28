"""
This script generates fake instance-recognition data from drone rollouts.
The intent is to measure how well a trained instance-recognition model recognizes the
"""
import cv2
import os
import math
import torch
import imageio
import json
import numpy as np
from transforms3d import euler

from data_io.train_data import load_multiple_env_data_from_dir, split_into_segs
from data_io.env import load_and_convert_env_config
import learning.datasets.aux_data_providers as aup
from grounding_data.composit_data import MIN_CROP_SIZE
from grounding_data.geom_calc import calc_object_radius_px, crop_square_recenter
from data_io.instructions import get_restricted_env_id_lists

from env_config.definitions.landmarks import get_landmark_index_to_name

import parameters.parameter_server as P

#rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/real"
#data_out_dir = "/media/clic/shelf_space/grounding_data_real_test"

#data_out_dir = "/media/clic/BigStore/gd_from_rollouts/real_7kto8k_0.3"
#rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/real_7kto8k"
#data_out_dir = "/media/clic/BigStore/gd_from_rollouts/real_6kto7k_0.4"
#rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/real_seen"

#data_out_dir = "/media/clic/BigStore/gd_from_rollouts/simulator_6kto7k_t_oldsim"
#rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/6kto7k_sim_t_oldsim"

data_out_dir = "/media/clic/BigStore/grounding_data_rss2020/grounding_data_sim_test_6kto7k_widerange"
rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/6kto7k_sim"

OBJECT_RADIUS_M = 0.3
HFOV = 84


def make_mask(image, pos_px, radius):
    mask = np.zeros_like(image).astype(np.float32)
    cv2.circle(mask, (int(pos_px[1].item()), int(pos_px[0].item())), int(radius), (1.0, 1.0, 1.0), -1)
    return mask

"""
def extract_crop(image, pos_px, radius):
    # Simulator uses object bottom-centers, shift the crop to include the object center better
    #t = max(int(pos_px[0] - int(radius * 1.7)), 0)
    #b = min(int(pos_px[0] + int(radius * 0.3)), image.shape[0])
    t = max(int(pos_px[0] - int(radius)), 0)
    b = min(int(pos_px[0] + int(radius)), image.shape[0])
    l = max(int(pos_px[1] - int(radius)), 0)
    r = min(int(pos_px[1] + int(radius)), image.shape[1])
    crop = image[t:b, l:r, :]
    return crop
"""


def generate_real_test_data():
    rollouts = load_multiple_env_data_from_dir(rollout_dir, single_proc=True)
    rollouts = split_into_segs(rollouts)
    provider_lm_pos_lm_indices_fpv = aup.resolve_data_provider(aup.PROVIDER_CHUNK_LM_POS_DATA)
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
    env_list = train_envs

    scenes_out_dir = os.path.join(data_out_dir, "scenes")
    objects_out_dir = os.path.join(data_out_dir, "objects")
    masks_out_dir = os.path.join(data_out_dir, "masks")
    os.makedirs(scenes_out_dir, exist_ok=True)
    os.makedirs(objects_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    for r, rollout in enumerate(rollouts):
        if len(rollout) == 0:
            continue
        print(f"Processing rollout {r} out of {len(rollouts)}")

        metadata = rollout[0] if "env_id" in rollout[0] else rollout[0]["metadata"]
        env_id = metadata["env_id"]
        if env_id not in env_list:
            print(f"Skipping excluded env: {env_id}")
            continue
        config = load_and_convert_env_config(env_id)

        res = provider_lm_pos_lm_indices_fpv(rollout, None, allow_oob_projections=True)
        lm_pos_and_indices = {a: b for a, b in res}
        lm_pos_fpv = lm_pos_and_indices["lm_pos_fpv"]
        lm_indices = lm_pos_and_indices["lm_indices"]
        lm_pos_map = lm_pos_and_indices["lm_pos_map"]

        seg_idx = metadata["seg_idx"]

        for t, sample in enumerate(rollout):
            cam_pos = sample["state"].get_cam_pos_3d()
            # cam_rot_quat = sample["state"].get_cam_rot()
            cam_pos_2d = cam_pos[:2]

            # The trajectories tend to have around 15 identical frames in the beginning. Skip those
            if t < 12 or t > 30:
                continue
            image = sample["state"].image

            scene_id = f"{env_id}_{seg_idx}_{t}"

            # Get object locations in the image.
            lm_indices_t = lm_indices[t]
            lm_pos_fpv_t = lm_pos_fpv[t]
            lm_pos_map_t = lm_pos_map[t]

            imageio.imsave(os.path.join(scenes_out_dir, f"{scene_id}--composed_scene.png"), image)

            obj_mask_dir = os.path.join(masks_out_dir, scene_id)
            os.makedirs(obj_mask_dir, exist_ok=True)

            # If no landmarks are visible, skip this timestep
            present_landmark_names = []
            if lm_pos_map_t is not None:
                # For each object, extract a crop and save in the objects directory
                # Also extract a mask, and save it as a label.
                for lm_pos_fpv_t_i, lm_pos_map_t_i, lm_index_t_i in zip(lm_pos_fpv_t, lm_pos_map_t, lm_indices_t):
                    obj_id = get_landmark_index_to_name()[lm_index_t_i.item()]
                    if obj_id == "0Null":
                        continue
                    dst_to_lm = torch.sqrt(torch.sum((torch.from_numpy(cam_pos_2d).float() - lm_pos_map_t_i) ** 2)).item()
                    obj_radius_px = calc_object_radius_px(dst_to_lm, image.shape[1])
                    crop = crop_square_recenter(image, lm_pos_fpv_t_i, obj_radius_px)
                    mask = make_mask(image, lm_pos_fpv_t_i, obj_radius_px)

                    # Save the crop and mask according to dataset structure
                    obj_crop_dir = os.path.join(objects_out_dir, obj_id)
                    os.makedirs(obj_crop_dir, exist_ok=True)

                    # Create metadata so that we can sort by distance
                    query_metadata = {
                        "obj_id": obj_id,
                        "dst_to_obj": dst_to_lm,
                        "obj_rad_px": obj_radius_px,
                        "obj_index": lm_index_t_i.item()
                    }
                    # Don't save tiny crops from far away
                    query_path = os.path.join(obj_crop_dir, f"query_{obj_id}_{scene_id}.png")
                    metad_path = os.path.join(obj_crop_dir, f"query_{obj_id}_{scene_id}.json")
                    if min(crop.shape[:2]) > MIN_CROP_SIZE:
                        imageio.imsave(query_path, crop)
                        with open(metad_path, "w") as fp:
                            json.dump(query_metadata, fp)
                    imageio.imsave(os.path.join(obj_mask_dir, f"{obj_id}.png"), (mask * 255).astype(np.uint8))
                    present_landmark_names.append(obj_id)

            # For every absent object, generate a plain black mask where that object is absent.
            black_mask = np.zeros_like(image)
            unseen_objects = set(config["landmarkName"]).difference(set(present_landmark_names))
            for obj_id in unseen_objects:
                if obj_id == "0Null":
                    continue
                imageio.imsave(os.path.join(obj_mask_dir, f"{obj_id}.png"), black_mask)


if __name__ == "__main__":
    P.initialize_experiment()
    generate_real_test_data()

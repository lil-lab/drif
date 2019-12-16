import csv
import os
import random
import shutil

from env_config.definitions.landmarks import get_landmark_stage_name
from data_io.paths import get_rollout_viz_dir, get_eval_tmp_dataset_name, get_amt_batch_dir, get_landmark_images_dir
from data_io.env import get_landmark_locations_airsim
from mains.eval.multiple_eval_rollout import setup_parameter_namespaces

import parameters.parameter_server as P

BASE_URL = "https://lani-real.s3.amazonaws.com/human_eval/rollout_animations/"
LANDMARK_DIR = "https://lani-real.s3.amazonaws.com/human_eval/landmarks/"

DOMAIN = "real"
# Columns in CSV to be uploaded to AMT
CSV_KEYS = ["instruction", "id", "image_url", "landmarks_html"]
# Columns in CSV where we keep a mapping between AMT IDs and the rollouts they came from.
ID_MAP_KEYS = ["id", "agent", "env_id", "set_idx", "seg_idx"]

def landmarks_in_env(env_id):
    lm_names, lm_idx, lm_pos = get_landmark_locations_airsim(env_id=env_id)
    stage_names = [get_landmark_stage_name(l) for l in lm_names]
    return stage_names

def build_landmark_html(stage_names):
    html = ""
    # Assume the stage_names are numbers. Otherwise
    stage_ints = [int(s) for s in stage_names]
    for stage_int in sorted(stage_ints):
        html += f'<img src="{LANDMARK_DIR}{stage_int}.png" class="landmark-image" alt="ERROR! DO NOT SUBMIT!"/>'
    return html

def read_text_file(text_file_path):
    text = ""
    with open(text_file_path, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        text += line
    return text

def generate_multiple_rollout_visualizations():
    params, system_namespaces = setup_parameter_namespaces()
    csv_rows = []
    id_map_rows = []

    # Have a random ID for each example, so that AMT workers can't figure out which IDs correspond to which agent.
    unique_ids = list(range(10000))
    random.shuffle(unique_ids)
    example_ordinal = 0

    for system_namespace in system_namespaces:
        P.switch_to_namespace(system_namespace)
        setup = P.get_current_parameters()["Setup"]
        dataset_name = get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
        base_dir = os.path.join(get_rollout_viz_dir(), f"{dataset_name}-{DOMAIN}")
        batch_out_dir = get_amt_batch_dir()
        images_out_dir = os.path.join(batch_out_dir, "rollout_animations")
        landmarks_out_dir = os.path.join(batch_out_dir, "landmarks")
        os.makedirs(images_out_dir, exist_ok=True)

        # Copy landmark images to the batch dir
        try:
            shutil.copytree(get_landmark_images_dir(), landmarks_out_dir)
        except Exception as e:
            print("Failed to copy landmark images. Already did?")

        files = os.listdir(base_dir)
        gif_files = [f for f in files if f.endswith("-roll.gif")]
        instr_files = [f for f in files if f.endswith("-instr.txt")]
        assert len(gif_files) == len(instr_files)

        for gif_file in gif_files:
            example_id = unique_ids[example_ordinal]
            example_ordinal += 1

            # Collect required info about this example
            seg_string, domain, suffix = gif_file.split("-")
            env_id, set_idx, seg_idx = seg_string.split(":")
            env_id, set_idx, seg_idx = int(env_id), int(set_idx), int(seg_idx)
            instr_file = f"{env_id}:{set_idx}:{seg_idx}-{domain}-instr.txt"
            instruction = read_text_file(os.path.join(base_dir, instr_file))

            lm_stage_names = landmarks_in_env(env_id)
            landmarks_html = build_landmark_html(lm_stage_names)

            old_gif_path = os.path.join(base_dir, gif_file)
            new_gif_filename = f"{example_id}.gif"
            new_gif_local_path = os.path.join(images_out_dir, new_gif_filename)
            gif_url = BASE_URL + new_gif_filename

            # Copy the image to it's destination
            shutil.copy(old_gif_path, new_gif_local_path)

            # Create a row for AMT batch table
            amt_table_row = {
                "id": example_id,
                "image_url": gif_url,
                "landmarks_html": landmarks_html,
                "instruction": instruction
            }
            map_table_row = {
                "id": example_id,
                "agent": setup["run_name"],
                "env_id": env_id,
                "set_idx": set_idx,
                "seg_idx": seg_idx
            }
            csv_rows.append(amt_table_row)
            id_map_rows.append(map_table_row)

    random.shuffle(csv_rows)

    # Save the tables
    # CSV batch table
    amt_table_path = os.path.join(batch_out_dir, "amt_human_eval_batch.csv")
    with open(amt_table_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_KEYS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    amt_sandbox_table_path = os.path.join(batch_out_dir, "amt_human_eval_batch_sandbox.csv")
    with open(amt_sandbox_table_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_KEYS)
        writer.writeheader()
        for row in csv_rows[100:599]:
            writer.writerow(row)

    amt_tiny_table_path = os.path.join(batch_out_dir, "amt_human_eval_batch_tinytrial.csv")
    with open(amt_tiny_table_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_KEYS)
        writer.writeheader()
        for row in csv_rows[0:20]:
            writer.writerow(row)

    # Reverse mapping table
    id_mapping_path = os.path.join(batch_out_dir, "id_map.csv")
    with open(id_mapping_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ID_MAP_KEYS)
        writer.writeheader()
        for row in id_map_rows:
            writer.writerow(row)


if __name__ == "__main__":
    generate_multiple_rollout_visualizations()

import json
import os
import numpy as np

#from config import CONFIG_SAVE_DIR, CURVE_SAVE_DIR, INSTRUCTION_SAVE_DIR, ANNOTATION_FILE
import data_io.paths as paths
from data_io.instructions import clean_instruction
from env_config.definitions.nlp_templates import TemplateType
from geometry import vec_to_yaw
import parameters.parameter_server as P

"""
This file takes the generated configurations, curves and instructions, and produces an
annotation file that is compatible with the Lani dataset.
This is used to generate the synthetic language dataset for the RSS paper

Annotation format:
{"test": 
    [
    {
        "end_z": [240, 235, 236, 251, 258], 
        "end_x": [233, 243, 251, 256, 257],
        "config_file": "configs/random_config_59.json",
        "valid": true,
        "num_tokens": [7, 10, 10, 19, 15],
        "instructions_file": "instructions/instructions_59.txt",
        "end_rot": [110, 125, 20, 20, 290],
        "path_file": "paths/random_curve_59.json", 
        "moves": ["FFF", "FFFFFFFRFFFF", "FFFLLFFFFLLFLLLFF", "FFFFFFFFFFFFFFFF", "LLLFFRFFRRFFRRFFLLLLLLLL"], 
        "id": "59", 
        "instructions": 
            ["walk straight until you reach the tree .",
             "go right around the tree until you reach the flower .",
             "go left past the flower until you face the well .",
             "go towards the fruit crossing past the well and chair on right hand side until you reach the fruit .",
             "curve right round the fruit until you reach the other end facing the blue fence ."]
    },
    ...
    ],
"train:" : [...],
"dev:" : [...]
}
"""

template_types = [TemplateType.GOTO__LANDMARK_LANDMARK]
from env_config.generation.generate_random_config import START_I, END_I

TRAIN_SPLIT = [0, 0.7]
TEST_SPLIT = [0.7, 0.85]
DEV_SPLIT = [0.85, 1.0]


def get_split(place_idx):
    if  TRAIN_SPLIT[0] <= place_idx < TRAIN_SPLIT[1]:
        return "train"
    elif TEST_SPLIT[0] <= place_idx < TEST_SPLIT[1]:
        return "test"
    elif DEV_SPLIT[0] <= place_idx < DEV_SPLIT[1]:
        return "dev"


def make_annotations(start_i, end_i):
    P.initialize_experiment()

    annotations = {
        "train": [],
        "test": [],
        "dev": []
    }

    for config_id in range(start_i, end_i):
        config_path = paths.get_env_config_path(config_id)
        path_path = paths.get_curve_path(config_id)
        instruction_path = paths.get_instructions_path(config_id)

        with open(config_path) as fp:
            config = json.load(fp)

        with open(path_path) as fp:
            curve = json.load(fp)

        with open(instruction_path) as fp:
            instruction = fp.readline()
        token_list = clean_instruction(instruction)

        split = get_split((config_id % 100) / 100.0)

        start_dir = np.asarray(config["startHeading"]) - np.asarray(config["startPos"])
        start_yaw = vec_to_yaw(start_dir)
        start_yaw_cfg = np.rad2deg(-start_yaw + np.pi/2)

        dataset = {
            "id": str(config_id),
            "start_z": [curve["z_array"][0]],
            "start_x": [curve["x_array"][0]],
            "end_z": [curve["z_array"][-1]],
            "end_x": [curve["x_array"][-1]],
            "start_rot": [start_yaw_cfg],
            "config_file": "configs/random_config_%d.json" % config_id,
            "instructions_file": "instructions/instructions_%d.txt" % config_id,
            "path_file": "paths/random_curve_%d.json" % config_id,
            "moves": [],
            "valid": True,
            "num_tokens": [len(token_list)],
            "instructions": [instruction]
        }
        annotations[split].append(dataset)
        print ("Added annotations for env: " + str(config_id))

    with open(paths.get_instruction_annotations_path(), "w") as fp:
        json.dump(annotations, fp)


if __name__ == "__main__":
    make_annotations(START_I, END_I)
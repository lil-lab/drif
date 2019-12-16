import json
import os
import numpy as np

#from config import CONFIG_SAVE_DIR, CURVE_SAVE_DIR, INSTRUCTION_SAVE_DIR, ANNOTATION_FILE
import data_io.paths as paths
from data_io.instructions import clean_instruction
from env_config.definitions.nlp_templates import TemplateType
from geometry import vec_to_yaw
import parameters.parameter_server as P
from env_config.generation.generate_random_config import NEW_CONFIG_EVERY_N

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
DEV_SPLIT = [0.7, 0.85]
TEST_SPLIT = [0.85, 1.0]

"""
def get_split(place_idx):
    if  TRAIN_SPLIT[0] <= place_idx < TRAIN_SPLIT[1]:
        return "train"
    elif TEST_SPLIT[0] <= place_idx < TEST_SPLIT[1]:
        return "test"
    elif DEV_SPLIT[0] <= place_idx < DEV_SPLIT[1]:
        return "dev"
"""

def get_split_ranges(num_envs):
    train_range = [int(x*num_envs) for x in TRAIN_SPLIT]
    dev_range = [int(x*num_envs) for x in DEV_SPLIT]
    test_range = [int(x*num_envs) for x in TEST_SPLIT]
    return train_range, dev_range, test_range

def make_annotations(end_i):
    P.initialize_experiment()

    annotations = {
        "train": [],
        "test": [],
        "dev": []
    }

    train_range, dev_range, test_range = get_split_ranges(end_i)
    assert (train_range[1] - train_range[0]) % NEW_CONFIG_EVERY_N == 0, "training set size must be a multiple of NEW_CONFIG_EVERY_N"

    for config_id in range(end_i):
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

        curve_np = np.asarray(list(zip(curve["x_array"], curve["z_array"])))

        split = "train" if train_range[0] <= config_id < train_range[1] else \
                "dev" if dev_range[0] <= config_id < dev_range[1] else \
                "test" if test_range[0] <= config_id < test_range[1] else None

        #start_dir = np.asarray(config["startHeading"]) - np.asarray(config["startPos"])
        start_dir = curve_np[1] - curve_np[0]
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
    make_annotations(END_I)
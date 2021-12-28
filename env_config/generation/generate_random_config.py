import os
import json
import random
import sys

import numpy as np

from data_io import paths
from env_config.definitions.landmarks import LANDMARK_RADII, PORTABLE_LANDMARK_RADII, UNSEEN_LANDMARK_RADII
from env_config.generation.terrain_generation import add_lake_to_config
import parameters.parameter_server as P

MESH_RES = 100
X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
LANDMARK_EDGE_CLEARANCE = 120

START_I = 7000
END_I = 8000

MIN_NUM_OBJECTS = 6
MAX_NUM_OBJECTS = 13
LANDMARK_MIN_SCALE = 1.0
LANDMARK_MAX_SCALE = 1.3
MIN_LANDMARK_BUFFER = 60
all_landmark_radii = LANDMARK_RADII

MAKE_LAKES = False
# Have this many configs in a row with the same landmark arrangement.
# Means have to reconfigure landmarks less frequently on the real drone. For RSS data this = 1
NEW_CONFIG_EVERY_N = 5

SUPPORT_REAL_DRONE = True
UNIFORM_LANDMARK_RADIUS = None
REAL_DRONE_NL = True

FEW_SHOT_TEST = True

if SUPPORT_REAL_DRONE:
    # This is for testing with child restraints
    DRONE_EDGE_CLEARANCE = 160
    UNIFORM_LANDMARK_RADIUS = 100
    MIN_LANDMARK_BUFFER = 60
    if REAL_DRONE_NL:
        LANDMARK_EDGE_CLEARANCE = 100
    else:
        LANDMARK_EDGE_CLEARANCE = DRONE_EDGE_CLEARANCE + UNIFORM_LANDMARK_RADIUS
    MIN_NUM_OBJECTS = 5
    MAX_NUM_OBJECTS = 8
    LANDMARK_MIN_SCALE = 1.0
    LANDMARK_MAX_SCALE = 1.0
    all_landmark_radii = PORTABLE_LANDMARK_RADII

if FEW_SHOT_TEST:
    all_landmark_radii = UNSEEN_LANDMARK_RADII
    MIN_NUM_OBJECTS = 6
    MAX_NUM_OBJECTS = 9

#FORCE_LANDMARK_SELECTION = ["Banana", "Gorilla"]
FORCE_LANDMARK_SELECTION = False

def generate_config_files(start_i, end_i):
    P.initialize_experiment()

    config_metadata = {
        "all_landmark_names": list(PORTABLE_LANDMARK_RADII.keys()),
        "all_landmark_radii": PORTABLE_LANDMARK_RADII,
        "new_config_every_n": NEW_CONFIG_EVERY_N
    }
    os.makedirs(os.path.dirname(paths.get_config_metadata_path()), exist_ok=True)
    with open(paths.get_config_metadata_path(), "w") as fp:
        json.dump(config_metadata, fp)

    config = None
    for config_num in range(start_i, end_i):
        # attempt to space landmarks
        attempts = 0
        # It's easier to generate a config with less objects, so to have a truly uniform distribution, we must sample it here.
        if FORCE_LANDMARK_SELECTION:
            num_objects = len(FORCE_LANDMARK_SELECTION)
        else:
            num_objects = int(random.uniform(MIN_NUM_OBJECTS, MAX_NUM_OBJECTS))

        if config_num % NEW_CONFIG_EVERY_N == 0 or config is None:
            print("making config %d with %d objects" % (config_num, num_objects))
            while True:
                config = try_make_config(num_objects)
                attempts += 1
                sys.stdout.write("\r Attempts: " + str(attempts))
                if config is not None:
                    print("")
                    break
            if MAKE_LAKES:
                config = add_lake_to_config(config, X_RANGE, Y_RANGE)

        path = paths.get_env_config_path(config_num)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(config, fp)


def is_pos_proposal_valid(config, pos_x, pos_z, radius):
    # check if any landmarks too close to others
    for i in range(len(config["xPos"])):
        other_x = config["xPos"][i]
        other_z = config["zPos"][i]
        other_radius = config["radius"][i] if not UNIFORM_LANDMARK_RADIUS else UNIFORM_LANDMARK_RADIUS
        other_pos = np.asarray([other_x, other_z])
        pos = np.asarray([pos_x, pos_z])
        min_dist = other_radius + radius + MIN_LANDMARK_BUFFER
        dist = np.linalg.norm(pos - other_pos)
        if dist < min_dist:
            return False
    return True


def try_make_config(num_objects, include_landmark=None):
    config = {
        "landmarkName": [],
        "radius": [],
        "xPos": [],
        "zPos": [],
        "isEnabled": [],
        "lakeCoords": []
    }
    # landmark_names = sorted(LANDMARK_RADII)
    global all_landmark_radii
    landmark_radii = {}
    # Scale up each landmark radius by a random factor in the provided interval
    all_landmark_names = list(all_landmark_radii.keys())
    if FORCE_LANDMARK_SELECTION:
        all_landmark_names = FORCE_LANDMARK_SELECTION
    landmark_names = random.sample(all_landmark_names, num_objects)

    # If we have to enclude a specific landmark, make sure that we include it.
    # WARNING: The added landmark is the first element. Later this assumption is made.
    if include_landmark is not None:
        landmark_names[0] = include_landmark

    for name in landmark_names:
        landmark_radii[name] = all_landmark_radii[name] * random.uniform(LANDMARK_MIN_SCALE, LANDMARK_MAX_SCALE)

    for landmark_name in landmark_names:
        config["landmarkName"].append(landmark_name)
        x_sample_range = (X_RANGE[0] + LANDMARK_EDGE_CLEARANCE,
                          X_RANGE[1] - LANDMARK_EDGE_CLEARANCE)
        y_sample_range = (Y_RANGE[0] + LANDMARK_EDGE_CLEARANCE,
                          Y_RANGE[1] - LANDMARK_EDGE_CLEARANCE)

        radius = landmark_radii[landmark_name] if not UNIFORM_LANDMARK_RADIUS else UNIFORM_LANDMARK_RADIUS
        proposed_x = None; proposed_y = None
        attempts = 0
        while True:
            proposed_x = random.randint(*x_sample_range)
            proposed_y = random.randint(*y_sample_range)
            attempts += 1
            if is_pos_proposal_valid(config, proposed_x, proposed_y, radius):
                #print ("Added: ", proposed_x, proposed_y, landmark_name)
                break
            if attempts > 100:
                return None
            #else:
            #    print ("Rejected: ", proposed_x, proposed_y)

        config["xPos"].append(proposed_x)
        config["zPos"].append(proposed_y)
        config["isEnabled"].append(True)
        config["radius"].append(radius)

    # check if any landmarks too close to others
    for i, landmark_i in enumerate(landmark_names):
        radius_i = landmark_radii[landmark_i]
        for j, landmark_j in enumerate(landmark_names):
            if j <= i:
                continue
            radius_j = landmark_radii[landmark_j]
            pos_i = np.array([float(config["xPos"][i]),
                              float(config["zPos"][i])])
            pos_j = np.array([float(config["xPos"][j]),
                              float(config["zPos"][j])])
            dist = ((pos_i - pos_j) ** 2).sum() ** 0.5
            min_dist = radius_i + radius_j + MIN_LANDMARK_BUFFER
            if dist < min_dist:
                return None
    return config


def make_config_with_landmark(landmark_name=None, num_objects=10):
    config = None

    while config is None:
        while config is None:
            config = try_make_config(num_objects, landmark_name)
        config = add_lake_to_config(config, X_RANGE, Y_RANGE)

    pos_x = config["xPos"][0]
    pos_z = config["zPos"][0]

    return config, pos_x, pos_z


if __name__ == "__main__":
    generate_config_files(START_I, END_I)

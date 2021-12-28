import json
import math
import os
import random
import sys
from enum import Enum

import numpy as np
from sympy.utilities.iterables import multiset_permutations

import data_io.paths as paths
from env_config.definitions.landmarks import LANDMARK_RADII, PORTABLE_LANDMARK_RADII
from env_config.generation.terrain_generation import add_lake_to_config
from env_config.generation.generate_random_config import START_I, END_I

import parameters.parameter_server as P

class ConfigType(Enum):
    RANDOM = 0,
    CIRCLE_OF_LANDMARKS = 1,
    CIRCLE_PERMUTATIONS = 2,
    RANDOM_CORNER = 3


DRONE_FOV = 90

MESH_RES = 100
X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
EDGE_WIDTH = 90

GENERATE_FOR_TRANSFER = False

MIN_NUM_OBJECTS = 6
MAX_NUM_OBJECTS = 13
LANDMARK_MIN_SCALE = 1.0
LANDMARK_MAX_SCALE = 1.3
MIN_LANDMARK_BUFFER = 60
all_landmark_radii = LANDMARK_RADII

if GENERATE_FOR_TRANSFER:
    MIN_NUM_OBJECTS = 6
    MAX_NUM_OBJECTS = 6
    LANDMARK_MIN_SCALE = 1.0
    LANDMARK_MAX_SCALE = 1.0
    MIN_LANDMARK_BUFFER = 150
    all_landmark_radii = PORTABLE_LANDMARK_RADII

FORCE_LANDMARK_SELECTION = ["Banana", "Gorilla"]
MAKE_LAKES = False

# Settings for generating environments with randomly scattered objects and where drone starts in a random location
"""
CONFIG_TYPE = ConfigType.RANDOM
START_I = 0
END_I = 10000
MIN_NUM_OBJECTS = 6
MAX_NUM_OBJECTS = 13
"""

# Settings for generating environments for a circle of randomly selected landmarks
"""
CONFIG_TYPE = ConfigType.CIRCLE_OF_LANDMARKS
START_I = 0
END_I = 10000
MIN_NUM_OBJECTS = 4
MAX_NUM_OBJECTS = 7
"""

# Settings for generating environments for a circle of few landmarks in every possible permutation
"""
CONFIG_TYPE = ConfigType.CIRCLE_PERMUTATIONS

# Total number of permutations of landmark locations, which landmark is selected and which side is selected
PERM_LANDMARK_NAMES = ["Soldier", "Pillar", "Mushroom", "Rock", "ConniferCluster"]
MIN_NUM_OBJECTS = len(PERM_LANDMARK_NAMES)
MAX_NUM_OBJECTS = MIN_NUM_OBJECTS
START_I = 0
END_I = math.factorial(MIN_NUM_OBJECTS) * MIN_NUM_OBJECTS * 2
"""

# Settings for generating randomly scattered objects, where drone starts in the corner
CONFIG_TYPE = ConfigType.RANDOM_CORNER
MIN_NUM_OBJECTS = 4
MAX_NUM_OBJECTS = 13
DRONE_START = [0, 0]
NO_OBJ_RANGE_X = [0, 200]
NO_OBJ_RANGE_Y = [0, 200]
NO_OBJ_RANGE = np.asarray([NO_OBJ_RANGE_X, NO_OBJ_RANGE_Y])


def main(start_i, end_i, config_type):
    P.initialize_experiment()
    for config_num in range(start_i, end_i):
        # attempt to space landmarks
        config = None
        attempts = 0
        # It's easier to generate a config with less objects, so to have a truly uniform distribution, we must sample it here.
        num_objects = int(random.uniform(MIN_NUM_OBJECTS, MAX_NUM_OBJECTS))
        if FORCE_LANDMARK_SELECTION:
            num_objects = len(FORCE_LANDMARK_SELECTION)

        print("making config %d with %d objects" % (config_num, num_objects))

        while True:
            if config_type == ConfigType.RANDOM or config_type == ConfigType.RANDOM_CORNER:
                start_pos, start_heading = None, None
                if config_type == ConfigType.RANDOM_CORNER:
                    start_pos, start_heading = pick_drone_start_pos()

                config = try_make_config_random(num_objects, start_pos, start_heading)
                if config is not None and MAKE_LAKES:
                    config = add_lake_to_config(config, X_RANGE, Y_RANGE)

            elif config_type == ConfigType.CIRCLE_OF_LANDMARKS:
                config = try_make_config_circle_of_landmarks(num_objects)
            elif config_type == ConfigType.CIRCLE_PERMUTATIONS:
                config = try_make_config_circle_permutations(num_objects, config_num)
            else:
                print ("Invalid config type!" + str(config_type))
                quit(-1)

            attempts += 1
            sys.stdout.write("\r Attemtps: " + str(attempts))
            if config is not None:
                print("")
                break

        os.makedirs(paths.get_env_config_dir(), exist_ok=True)
        path = os.path.join(paths.get_env_config_path(config_num))

        with open(path, 'w') as fp:
            json.dump(config, fp)


def is_pos_proposal_valid(config, pos_x, pos_z, radius, drone_start_pos=None):

    # check if the proposal is too close to any existing landmarks
    pos = np.asarray([pos_x, pos_z])
    for i in range(len(config["xPos"])):
        other_x = config["xPos"][i]
        other_z = config["zPos"][i]
        other_radius = config["radius"][i]
        other_pos = np.asarray([other_x, other_z])
        min_dist = other_radius + radius + MIN_LANDMARK_BUFFER
        dist = np.linalg.norm(pos - other_pos)
        if dist < min_dist:
            return False

    # Check if it's too close to the drone's starting position (i.e. will likely block view of other objects)
    if drone_start_pos is not None:
        drone_start_pos = np.asarray(drone_start_pos)
        min_dist = radius + 150
        dist = np.linalg.norm(drone_start_pos - pos)
        if dist < min_dist:
            return False

    return True


def empty_config():
    return {
        "landmarkName": [],
        "radius": [],
        "xPos": [],
        "zPos": [],
        "isEnabled": [],
        "lakeCoords": []
    }


def pick_drone_start_pos():
    """
    :return: Drone's starting position and starting heading, where the heading is a point towards which the drone will face
    """
    pick = random.choice([0, 1, 2, 3])
    if pick == 0:
        return [0.0, 0.0], [100.0, 100.0]
    elif pick == 1:
        return [1000.0, 0.0], [900.0, 100.0]
    elif pick == 2:
        return [0.0, 1000.0], [100.0, 900.0]
    elif pick == 3:
        return [1000.0, 1000.0], [900.0, 900.0]


def place_landmarks_on_circle_config(landmark_names, landmark_radii):
    config = empty_config()

    num_objects = len(landmark_radii)
    x_width = X_RANGE[1] - X_RANGE[0]
    y_width = Y_RANGE[1] - Y_RANGE[0]
    centroid = np.asarray([X_RANGE[0] + x_width / 2,
                           Y_RANGE[0] + y_width / 2])
    circle_radius = x_width / 2.5

    fov = DRONE_FOV * 3.14159 / 180
    theta_min = -fov
    theta_step = 2 * fov / (num_objects - 1)

    theta_drone = -3.14159
    offset_drone = [circle_radius * math.sin(theta_drone),
                    circle_radius * math.cos(theta_drone)]
    drone_pos = centroid + np.asarray(offset_drone)

    # Add extra variables in the config for curve generator
    config["startPos"] = list(drone_pos)
    config["startHeading"] = list(centroid)

    # Add landmarks to config
    for l, landmark_name in enumerate(landmark_names):
        config["landmarkName"].append(landmark_name)

        theta = theta_min + theta_step * l
        offset = [circle_radius * math.sin(theta),
                  circle_radius * math.cos(theta)]
        pos = centroid + np.asarray(offset)

        config["xPos"].append(pos[0])
        config["zPos"].append(pos[1])
        config["isEnabled"].append(True)
        config["radius"].append(landmark_radii[landmark_name])

    return config

def get_permutation_side_selection(config_num, num_objects):
    indices = list(range(num_objects))
    permutations = list(multiset_permutations(indices))
    remainder = config_num

    side_idx = remainder % 2
    remainder = int(remainder / 2)

    landmark_idx = remainder % num_objects
    remainder = int(remainder / num_objects)

    perm_idx = remainder
    if perm_idx > len(permutations) - 1:
        print("pomdp number too large!")
    return permutations[perm_idx], landmark_idx, side_idx


def try_make_config_circle_permutations(num_objects, config_num):
    permutation, landmark_idx, side_idx = get_permutation_side_selection(config_num, num_objects)
    landmark_names = [PERM_LANDMARK_NAMES[i] for i in permutation]
    landmark_radii = {}
    for name in landmark_names:
        landmark_radii[name] = all_landmark_radii[name]

    config = place_landmarks_on_circle_config(landmark_names, landmark_radii)
    return config


def try_make_config_circle_of_landmarks(num_objects):

    global all_landmark_radii
    landmark_names = random.sample(list(all_landmark_radii.keys()), num_objects)
    landmark_radii = {}
    for name in landmark_names:
        landmark_radii[name] = all_landmark_radii[name]

    config = place_landmarks_on_circle_config(landmark_names, landmark_radii)
    return config


def try_make_config_random(num_objects, drone_start_pos=None, drone_start_heading=None):
    config = empty_config()
    # landmark_names = sorted(LANDMARK_RADII)
    global all_landmark_radii
    landmark_radii = {}
    # Scale up each landmark radius by a random factor in the provided interval
    if FORCE_LANDMARK_SELECTION:
        num_objects = len(FORCE_LANDMARK_SELECTION)
        all_landmark_names = FORCE_LANDMARK_SELECTION
    else:
        all_landmark_names = list(all_landmark_radii.keys())
    landmark_names = random.sample(all_landmark_names, num_objects)
    for name in landmark_names:
        landmark_radii[name] = all_landmark_radii[name] * random.uniform(LANDMARK_MIN_SCALE, LANDMARK_MAX_SCALE)

    for landmark_name in landmark_names:
        config["landmarkName"].append(landmark_name)
        x_sample_range = (X_RANGE[0] + EDGE_WIDTH,
                          X_RANGE[1] - EDGE_WIDTH)
        y_sample_range = (Y_RANGE[0] + EDGE_WIDTH,
                          Y_RANGE[1] - EDGE_WIDTH)

        radius = landmark_radii[landmark_name]
        proposed_x = None; proposed_y = None
        attempts = 0
        while True:
            proposed_x = random.randint(*x_sample_range)
            proposed_y = random.randint(*y_sample_range)
            attempts += 1
            if is_pos_proposal_valid(config, proposed_x, proposed_y, radius, drone_start_pos):
                #print ("Added: ", proposed_x, proposed_y, landmark_name)
                break
            if attempts > 1000:
                return None
            #else:
            #    print ("Rejected: ", proposed_x, proposed_y)

        config["xPos"].append(proposed_x)
        config["zPos"].append(proposed_y)
        config["isEnabled"].append(True)
        config["radius"].append(radius)

        if drone_start_pos is not None and drone_start_heading is not None:
            config["startPos"] = drone_start_pos
            config["startHeading"] = drone_start_heading

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


if __name__ == "__main__":
    main(START_I, END_I, CONFIG_TYPE)
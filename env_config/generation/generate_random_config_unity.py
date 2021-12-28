import os
import sys
import json
import random
import numpy as np
from landmarks import LANDMARK_RADII, PORTABLE_LANDMARK_RADII
from terrain_generation import add_lake_to_config

#CONFIG_BASE_DIR = "/Users/awbennett/Documents/research_project/unity/ProceduralField/Assets/Resources/configs"
CONFIG_BASE_DIR = "/Users/valts/Documents/Cornell/droning/unity/ProceduralField_backup/Assets/Resources/configs"

MESH_RES = 100
X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
EDGE_WIDTH = 90

START_I = 0
END_I = 100

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


def main():
    for config_num in range(START_I, END_I):
        # attempt to space landmarks
        config = None
        attempts = 0
        # It's easier to generate a config with less objects, so to have a truly uniform distribution, we must sample it here.
        num_objects = int(random.uniform(MIN_NUM_OBJECTS, MAX_NUM_OBJECTS))

        print("making config %d with %d objects" % (config_num, num_objects))

        while True:
            config = try_make_config(num_objects)
            attempts += 1
            sys.stdout.write("\r Attemtps: " + str(attempts))
            if config is not None:
                print("")
                break
        config = add_lake_to_config(config, X_RANGE, Y_RANGE)

        path = os.path.join(CONFIG_BASE_DIR,
                            "random_config_%d.json" % config_num)
        with open(path, 'w') as fp:
            json.dump(config, fp)

def is_pos_proposal_valid(config, pos_x, pos_z, radius):
    # check if any landmarks too close to others
    for i in range(len(config["xPos"])):
        other_x = config["xPos"][i]
        other_z = config["zPos"][i]
        other_radius = config["radius"][i]
        other_pos = np.asarray([other_x, other_z])
        pos = np.asarray([pos_x, pos_z])
        min_dist = other_radius + radius + MIN_LANDMARK_BUFFER
        dist = np.linalg.norm(pos - other_pos)
        if dist < min_dist:
            return False
    return True

def try_make_config(num_objects):
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
    landmark_names = random.sample(all_landmark_radii, num_objects)
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
            if is_pos_proposal_valid(config, proposed_x, proposed_y, radius):
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
    main()

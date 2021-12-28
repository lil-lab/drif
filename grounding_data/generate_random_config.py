import random
import sys
import numpy as np

MESH_RES = 100
X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
EDGE_WIDTH = 90

START_I = 0
END_I = 500

MIN_NUM_OBJECTS = 6
MAX_NUM_OBJECTS = 16
LANDMARK_MIN_SCALE = 1.0
LANDMARK_MAX_SCALE = 1.3
MIN_LANDMARK_BUFFER = 60
DUPLICATE_PROB = 0.3
MIN_RADIUS = 30
MAX_RADIUS = 130

ROLL_RANGE = [-3.14159, 3.14159]
PITCH_RANGE = [-0.1, 0.1]
YAW_RANGE = [-0.1, 0.1]


def gen_config(all_landmark_names):
    num_objects = int(random.uniform(MIN_NUM_OBJECTS, MAX_NUM_OBJECTS))
    attempts = 0
    while True:
        config = try_make_config(num_objects, all_landmark_names)
        attempts += 1
        sys.stdout.write("\r Attempts: " + str(attempts))
        if config is not None:
            break
    return config


def is_pos_proposal_valid(config, pos_x, pos_z, radius):

    # If any landmark is too close to the border, consider this proposal invalid
    if (pos_x < MIN_LANDMARK_BUFFER or
            pos_z < MIN_LANDMARK_BUFFER or
            pos_x > 1000 - MIN_LANDMARK_BUFFER or
            pos_z > 1000 - MIN_LANDMARK_BUFFER):
        return False

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


def try_make_config(num_objects, all_landmark_names):
    config = {
        "landmarkName": [],
        "radius": [],
        "xPos": [],
        "zPos": [],
        "rpy": [],
        "isEnabled": [],
        "lakeCoords": []
    }
    landmark_names = random.sample(all_landmark_names, num_objects)

    # With some probability, duplicate some landmarks
    duplicates = np.random.binomial(1, DUPLICATE_PROB, len(landmark_names) - 1)
    for i, dup in enumerate(duplicates):
        if dup > 0.5:
            landmark_names[i+1] = landmark_names[i]

    #print("Total: ", len(landmark_names), "Duplicates: ", len(landmark_names) - len(set(landmark_names)))

    for landmark_name in landmark_names:
        config["landmarkName"].append(landmark_name)
        x_sample_range = (X_RANGE[0] + EDGE_WIDTH,
                          X_RANGE[1] - EDGE_WIDTH)
        y_sample_range = (Y_RANGE[0] + EDGE_WIDTH,
                          Y_RANGE[1] - EDGE_WIDTH)

        attempts = 0
        while True:
            proposed_x = random.randint(*x_sample_range)
            proposed_y = random.randint(*y_sample_range)
            radius = random.randint(MIN_RADIUS, MAX_RADIUS)
            attempts += 1
            if is_pos_proposal_valid(config, proposed_x, proposed_y, radius):
                break
            if attempts > 1000:
                return None

        roll = random.uniform(*ROLL_RANGE)
        pitch = random.uniform(*PITCH_RANGE)
        yaw = random.uniform(*YAW_RANGE)
        rpy = [roll * 180 / 3.14159, pitch * 180 / 3.14159, yaw * 180 / 3.14159]

        config["xPos"].append(proposed_x)
        config["zPos"].append(proposed_y)
        config["radius"].append(radius)
        config["rpy"].append(rpy)
        config["isEnabled"].append(True)

    # check if any landmarks too close to others
    for i, landmark_i in enumerate(landmark_names):
        for j, landmark_j in enumerate(landmark_names):
            if j <= i:
                continue
            pos_i = np.array([float(config["xPos"][i]),
                              float(config["zPos"][i])])
            pos_j = np.array([float(config["xPos"][j]),
                              float(config["zPos"][j])])
            dist = ((pos_i - pos_j) ** 2).sum() ** 0.5
            min_dist = MIN_LANDMARK_BUFFER * 2
            if dist < min_dist:
                return None

    return config

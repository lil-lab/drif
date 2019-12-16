import json
import math
import os
import random
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from data_io import paths
from data_io.env import load_env_config
from env_config.definitions.landmarks import LANDMARK_RADII
from env_config.definitions.nlp_templates import TemplateType, generate_template
from env_config.generation.generate_curriculum_config import get_permutation_side_selection
import parameters.parameter_server as P

from env_config.generation.generate_random_config import START_I, END_I, SUPPORT_REAL_DRONE, DRONE_EDGE_CLEARANCE, UNIFORM_LANDMARK_RADIUS

sys.path.append(".")

MESH_RES = 100
X_RANGE = (0, 1000)
Y_RANGE = (0, 1000)
EDGE_WIDTH = 90

POINT_DST_RANGE = (200, 2000)

DEBUG = False

LANDMARK_NAMES = sorted(LANDMARK_RADII)

INTERVAL_LEN = 40
START_EDGE_WIDTH = 100
MOVE_DIST = 4
NOISE_VAL = 0
SAMPLE_RATE = 20
LAKE_REPULSION_DIST = 120
LAKE_REPULSION = 0.2
MIN_DIST_TO_NEXT = 200

MAX_SIM_STEPS = 500

STRAIGHT_STEPS = 2

template_types = [TemplateType.GOTO__LANDMARK_SIDE]
SAMPLING_MODE = "consistent"
#SAMPLING_MODE = "random"

#START_POS_SAMPLING = "random"
START_POS_SAMPLING = "corner"

if SUPPORT_REAL_DRONE:
    START_EDGE_WIDTH = DRONE_EDGE_CLEARANCE

PERMUTATION_TEST = False


def generate_template_curves(start_i, end_i):
    P.initialize_experiment()
    pool = Pool(18)
    pool.map(generate_template_curve, range(start_i, end_i))
    pool.close()
    pool.join()


def generate_template_curve(config_id):

    config = load_env_config(config_id)
    # Make a curve that follows the given template
    make_template_curve(config, config_id)


def make_template_curve(config, config_id):
    # try making curve until it succeeds (doesn't go out of bounds)
    print("making curve for config id " + str(config_id))

    cnt = 0
    while True:
        if PERMUTATION_TEST:
            num_objects = len(config["landmarkName"])
            permutation, landmark_idx, side_idx = get_permutation_side_selection(config_id, num_objects)
            landmark_choices = [config["landmarkName"][landmark_idx]]
            side_choices = ["left"] if side_idx == 0 else ["right"]
            template = generate_template(template_types, landmark_choices, sampling="consistent", side_choices=side_choices)
        else:
            # Create a template for one of the template types that we can choose from
            template = generate_template(template_types, config["landmarkName"], sampling=SAMPLING_MODE)

        pos_array = try_make_template_curve(config, template)
        cnt += 1
        if pos_array is not None:
            break
        elif cnt > 500:
            print(f"FAILED GENERATING CURVES FOR ENV: {config_id}")

    assert isinstance(pos_array, list)

    # write pos array to file
    pos_lists = {"x_array": [x for x, _ in pos_array],
                 "z_array": [y for _, y in pos_array]}
    curve_path = paths.get_curve_path(config_id)
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    with open(curve_path, 'w') as fp:
        json.dump(pos_lists, fp)

    # Write the template to file
    template_data = {
        "type": str(template.type),
        "landmark1": str(template.landmark1),
        "landmark2": str(template.landmark2),
        "side": str(template.side),
        "dir": str(template.dir),
        "instruction": str(template.instruction)
    }

    template_path = paths.get_template_path(config_id)
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    with open(template_path, "w") as fp:
        json.dump(template_data, fp)

    instruction_path = paths.get_instructions_path(config_id)
    os.makedirs(os.path.dirname(instruction_path), exist_ok=True)
    with open(instruction_path, "w") as fp:
        fp.write(template.instruction)

    landmark_pos = get_landmark_pos(config)
    lake_pos = get_lake_pos(config)

    plt.figure()
    axes = plt.axes()
    # plot landmarks
    axes.plot([x for x, _ in landmark_pos],
              [y for _, y in landmark_pos], "bo")
    # plot route starting point
    x_start, y_start = pos_array[0]
    axes.plot([x_start], [y_start], "ro", ms=10.0)
    axes.plot([x for x, _ in pos_array[1:]],
              [y for _, y in pos_array[1:]], "r.")
    # plot lake
    axes.plot([x for x, _ in lake_pos],
              [y for _, y in lake_pos], "b.")

    axes.set_xlim(list(X_RANGE))
    axes.set_ylim(list(Y_RANGE))
    #plt.show()
    plot_path = os.path.join(paths.get_plots_dir(), "generated_path_" + str(config_id) + ".png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)


def try_make_template_curve(config, template):
    if template.type == TemplateType.GOTO__LANDMARK:
        return try_make_curve_landmark(config, template.landmark1)
    elif template.type == TemplateType.GOTO__LANDMARK_SIDE:
        return try_make_curve_landmark_side(config, template.landmark1, template.side)
    elif template.type == TemplateType.GOTO__LANDMARK_LANDMARK:
        return try_make_curve_2_landmarks(config, template.landmark1, template.landmark2)
    elif template.type == TemplateType.GOAROUND__LANDMARK_DIR:
        return try_make_curve_goaround_landmark(config, template.landmark1, template.dir)
    print("Unrecognized template type")
    return None


def try_make_curve_landmark(config, landmark):
    pass

def center_point():
    return np.asarray([500.0, 500.0])

def try_make_curve_landmark_side(config, landmark, side):

    # Don't use landmark radii - just use a constant distance for the real drone
    if UNIFORM_LANDMARK_RADIUS:
        landmark_radii = [UNIFORM_LANDMARK_RADIUS for _ in config["radius"]]
    else:
        landmark_radii = config["radius"]

    landmark1_idx = get_landmark_index(config, landmark)

    if "startPos" in config:
        start_pos = np.asarray(config["startPos"])
        start_heading = np.asarray(config["startHeading"])
    else:
        start_pos = sample_start_pos(config)
        start_heading = center_point()

    points_to_visit = []
    landmark_side_point = sample_landmark_point(config, start_pos, landmark1_idx, landmark_radii, method=side)

    dir = start_heading - start_pos
    dst = np.linalg.norm(dir)
    dir /= dst
    takeoff_pos = start_pos + dir * STRAIGHT_STEPS * INTERVAL_LEN

    points_to_visit.append(takeoff_pos)
    points_to_visit.append(landmark_side_point)

    landmark_dist = np.linalg.norm(landmark_side_point - start_pos)

    # We want reasonable distances between start pos and first landmark and between landmarks
    # otherwise the drone often spawns on top of the solution, or picks the same landmark twice
    valid = valid_point_dst(landmark_dist)
    if not valid:
        return None

    return try_visit_points(config, start_pos, points_to_visit, landmark_radii)

def try_make_curve_2_landmarks(config, landmark1, landmark2):

    landmark_radii = config["radius"]

    start_pos = sample_start_pos(config)

    landmark1_idx = get_landmark_index(config, landmark1)
    landmark2_idx = get_landmark_index(config, landmark2)

    points_to_visit = []
    points_to_visit.append(sample_landmark_point(config, start_pos, landmark1_idx, landmark_radii, method="infront"))
    points_to_visit.append(sample_landmark_point(config, points_to_visit[0], landmark2_idx, landmark_radii, method="infront"))

    dst1 = np.linalg.norm(points_to_visit[0] - start_pos)
    dst2 = np.linalg.norm(points_to_visit[1] - points_to_visit[0])

    # We want reasonable distances between start pos and first landmark and between landmarks
    # otherwise the drone often spawns on top of the solution, or picks the same landmark twice
    valid = valid_point_dst(dst1) and valid_point_dst(dst2)
    if not valid:
        return None

    return try_visit_points(config, start_pos, points_to_visit, landmark_radii)


def valid_point_dst(dst):
    return POINT_DST_RANGE[0] < dst < POINT_DST_RANGE[1]

def sample_landmark_point(config, pos, landmark_index, landmark_radii, method="infront"):

    all_landmark_pos = get_landmark_pos(config)
    landmark_pos = all_landmark_pos[landmark_index]

    landmark_dir = landmark_pos - pos

    theta = math.atan2(landmark_dir[1], landmark_dir[0]) + math.pi
    if method == "front":
        theta = theta
    elif method == "random":
        theta = random.random() * 2 * math.pi
    elif method == "back":
        theta = theta + math.pi
    elif method == "left":
        theta = theta - math.pi / 2
    elif method == "right":
        theta = theta + math.pi / 2

    x, z = landmark_pos[0], landmark_pos[1]
    landmark_radius = landmark_radii[landmark_index]
    sample_point = np.array([x + math.cos(theta) * landmark_radius,
                             z + math.sin(theta) * landmark_radius])
    return sample_point


def try_make_curve_goaround_landmark(config, landmark1, dir):
    pass


def sample_start_pos(config):
    # initialise physics state
    start_pos_x_range = (X_RANGE[0] + START_EDGE_WIDTH,
                         X_RANGE[1] - START_EDGE_WIDTH)
    start_pos_y_range = (Y_RANGE[0] + START_EDGE_WIDTH,
                         Y_RANGE[1] - START_EDGE_WIDTH)

    if START_POS_SAMPLING == "random":
        pos = np.array([float(random.randint(*start_pos_x_range)),
                        float(random.randint(*start_pos_y_range))])
    elif START_POS_SAMPLING == "corner":
        pos = np.asarray([float(start_pos_x_range[random.randint(0, 1)]),
                          float(start_pos_y_range[random.randint(0, 1)])])
    else:
        raise NotImplementedError("Unknown start pos sampling mode")
    return pos


def get_landmark_index(config, landmark_name):
    return config["landmarkName"].index(landmark_name)


def get_lake_pos(config):
    if "lakeCoords" in config:
        lake_pos = np.array([[float(c["x"]), float(c["y"])] for c in
                             [config["lakeCoords"][i] for i in
                              range(0, len(config["lakeCoords"]), 1)]])

        if len(lake_pos) > 100:
            lake_pos_choices = np.random.choice(np.array(range(len(lake_pos))), int(len(lake_pos)/SAMPLE_RATE))
            lake_pos = lake_pos[lake_pos_choices]

        if len(lake_pos) > 0:
            lake_pos[:, 0] = (lake_pos[:, 0] * (X_RANGE[1] - X_RANGE[0]) / MESH_RES) + X_RANGE[0]
            lake_pos[:, 1] = (lake_pos[:, 1] * (Y_RANGE[1] - Y_RANGE[0]) / MESH_RES) + Y_RANGE[0]
    else:
        lake_pos = np.array([[0, 0], [0, 0]])
    return lake_pos


def get_landmark_pos(config):
    return np.array([[float(x), float(y)] for x, y in zip(config["xPos"], config["zPos"])])


def try_visit_points(config, pos, points, landmark_radii):
    # Make a curve that sequentially visits the provided points
    landmark_pos = get_landmark_pos(config)
    lake_pos = get_lake_pos(config)
    pos_array = [pos]
    total_distance = 0.0
    for next_target in points:
        sim_steps = 0
        while True:
            sim_steps += 1
            old_interval_num = get_interval_num(total_distance)

            # calculate movement by simulating physics
            movement = calc_movement(pos, next_target)
            repulsion = get_repulsion(pos + movement, landmark_pos,
                                      landmark_radii)
            if repulsion is not None:
                movement += repulsion
            lake_repulsion = get_lake_repulsion(pos + movement, lake_pos)
            movement += lake_repulsion

            # update state
            pos = pos + movement
            distance = math.sqrt(sum(movement ** 2))
            total_distance += distance

            # check if we have passed into new interval
            new_interval_num = get_interval_num(total_distance)
            if new_interval_num > old_interval_num:
                pos_array.append(pos)

            # check to see if we are out of bounds or path overlaps (failure)
            x, y = pos
            if is_oob(x, X_RANGE) or is_oob(y, Y_RANGE):
                # print ("OOB")
                return None
            if does_path_overlap(pos_array):
                # print ("Overlap")
                return None

            # check to see if we are near target
            dist_to_target = ((pos - next_target) ** 2).sum() ** 0.5
            if dist_to_target < INTERVAL_LEN / 2:
                break

            if sim_steps > MAX_SIM_STEPS:
                print("Exceeded sim steps!")
                return None

    return pos_array


def calc_movement(pos, next_target):
    movement_vector = (next_target - pos)
    movement_vector_len = ((next_target - pos) ** 2).sum() ** 0.5
    movement_signal = movement_vector / movement_vector_len * MOVE_DIST
    noise_angle = random.random() * math.pi * 2
    noise_val = random.random() * MOVE_DIST * NOISE_VAL
    noise = np.array([math.cos(noise_angle) * noise_val,
                      math.sin(noise_angle) * noise_val])
    return movement_signal + noise


def get_repulsion(pos, landmark_pos, landmark_radii):
    landmark_dist = ((landmark_pos - pos) ** 2).sum(1) ** 0.5
    min_dist, landmark_i = min((d, i) for i, d in enumerate(landmark_dist))
    for landmark_i, dist in enumerate(landmark_dist):
        landmark_radius = landmark_radii[landmark_i]
        if dist < landmark_radius:
            repulsion_dir = pos - landmark_pos[landmark_i]
            repulsion_dir_len = (repulsion_dir ** 2).sum() ** 0.5
            repulsion_dist = landmark_radius - min_dist
            repulsion = repulsion_dir * repulsion_dist / repulsion_dir_len
            return repulsion
    return None


def get_lake_repulsion(pos, lake_pos):
    if len(lake_pos) == 0:
        return np.zeros(2)

    lake_dist = [np.linalg.norm(lake_cell - pos) for lake_cell in lake_pos]
    NUM_POINTS = 20
    closest_n = np.argsort(lake_dist)[:NUM_POINTS]
    close_points = [lake_pos[i] for i in closest_n]

    points_added = 0
    repulsion = np.zeros(2)
    for point in close_points:
        repulsion_dir = pos - point
        repulsion_norm = np.linalg.norm(repulsion_dir)
        if repulsion_norm > LAKE_REPULSION_DIST:
            break
        repulsion += (repulsion_dir / repulsion_norm)
        points_added += 1

    if points_added == 0:
        return repulsion

    repulsion_dist = float(LAKE_REPULSION) / points_added
    output_repulsion = repulsion * repulsion_dist
    return output_repulsion


def get_interval_num(total_distance):
    return int(total_distance / INTERVAL_LEN)


def is_oob(val, val_range):
    if val < val_range[0] or val > val_range[1]:
        return True
    else:
        return False


def does_path_overlap(pos_array):
    NUM_SKIP = 3
    if len(pos_array) <= NUM_SKIP:
        return False
    current_pos = pos_array[-1]
    prev_pos_array = np.array(pos_array[:-NUM_SKIP])
    distances = ((prev_pos_array - current_pos) ** 2).sum(1) ** 0.5
    min_distance = min(distances)
    if min_distance <= INTERVAL_LEN:
        return True
    else:
        return False


#START_I = 0
#END_I = 10000



if __name__ == "__main__":
    generate_template_curves(START_I, END_I)

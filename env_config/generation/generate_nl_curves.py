import os
import random
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
#TODO: Adapt this
from data_io.paths import get_curve_path, get_curve_plot_path
from data_io.env import load_env_config, load_path, convert_path#, load_and_convert_env_config
from env_config.generation.generate_random_config import \
    X_RANGE, Y_RANGE, LANDMARK_RADII, START_I, END_I, MESH_RES, SUPPORT_REAL_DRONE, DRONE_EDGE_CLEARANCE, NEW_CONFIG_EVERY_N

import parameters.parameter_server as P

#CURVE_BASE_DIR = "/Users/awbennett/Documents/research_project/unity/ProceduralField/Assets/Resources/paths"
#CURVE_BASE_DIR = "/Users/valts/Documents/Cornell/droning/unity/ProceduralField_backup/Assets/Resources/paths"

DEBUG = False

CONTINUE_IF_ENVS_SAME = True

LANDMARK_NAMES = sorted(LANDMARK_RADII)

INTERVAL_LEN = 40
START_EDGE_WIDTH = 45

NUM_LANDMARKS_VISISTED = 5
CLOSE_LANDMARK_NUM = 4

MOVE_DIST = 4
NOISE_VAL = 1.5

SAMPLE_RATE = 20
LAKE_REPULSION_DIST = 120
LAKE_REPULSION = 3.8

MIN_DIST_TO_NEXT = 200

MAX_SIM_STEPS = 500

if SUPPORT_REAL_DRONE:
    NUM_LANDMARKS_VISISTED = 4

from multiprocessing import Pool

def make_curves_for_unique_config(config_id):
    start_pos, start_lm = None, None
    for env_id in range(config_id, config_id + NEW_CONFIG_EVERY_N, 1):
        config = load_env_config(env_id)
        curve_path = get_curve_path(env_id)
        plot_path = get_curve_plot_path(env_id)
        start_pos, start_lm = make_new_curve(config, curve_path, plot_path, start_pos, start_lm)

def main():
    P.initialize_experiment()
    global last_end_pos
    last_end_pos = None
    pool = Pool(16)
    every_nth_env = range(START_I, END_I, NEW_CONFIG_EVERY_N)
    pool.map(make_curves_for_unique_config, every_nth_env)
    pool.join()
    pool.close()


def debug():
    path = os.path.join(SUPPORT_REAL_DRONE, "random_config.json")
    with open(path) as fp:
        config = json.load(fp)
    for i in range(10):
        curve_path = "debug_curve.%d.json" % i
        plot_path = "debug_plot.%d.png" % i
        make_new_curve(config, curve_path, plot_path)


def try_make_curve(config, start_pos=None, start_landmark=None):
    # initialise physics state
    start_pos_x_range = (X_RANGE[0] + START_EDGE_WIDTH,
                         X_RANGE[1] - START_EDGE_WIDTH)
    start_pos_y_range = (Y_RANGE[0] + START_EDGE_WIDTH,
                         Y_RANGE[1] - START_EDGE_WIDTH)
    pos = np.array([float(random.randint(*start_pos_x_range)),
                    float(random.randint(*start_pos_y_range))])
    landmark_pos = np.array([[float(x), float(y)]
                             for x, y in zip(config["xPos"], config["zPos"])])
    landmark_radii = config["radius"]

    # Get the coordinates of every SAMPLE_RATE lake vertex in unity units (vs mesh indices)
    if "lakeCoords" in config:
        lake_pos = np.array([[float(c["x"]), float(c["y"])] for c in
                             [config["lakeCoords"][i] for i in
                              range(0, len(config["lakeCoords"]), SAMPLE_RATE)]])
    else:
        lake_pos = []

    if len(lake_pos) > 0:
        lake_pos[:, 0] = (lake_pos[:, 0] * (X_RANGE[1] - X_RANGE[0]) / MESH_RES) + X_RANGE[0]
        lake_pos[:, 1] = (lake_pos[:, 1] * (Y_RANGE[1] - Y_RANGE[0]) / MESH_RES) + Y_RANGE[0]
    #print ("Lake Pos: ", len(lake_pos))

    # get starting point and landmark visited
    landmarks_visited = []
    if start_pos is not None:
        pos = start_pos
        landmarks_visited.append(start_landmark)
    else:
        pos = sample_random_landmark_point(pos, landmark_pos, landmarks_visited,
                                       landmark_radii)
    if pos is None:
        return None

    num_landmarks_visited = NUM_LANDMARKS_VISISTED if NUM_LANDMARKS_VISISTED < len(landmark_pos) else len(landmark_pos) - 1

    # simulate path
    pos_array = [pos]
    total_distance = 0.0
    for _ in range(num_landmarks_visited - 1):
        next_target = sample_random_landmark_point(pos, landmark_pos,
                                                   landmarks_visited,
                                                   landmark_radii)
        if next_target is None:
            return None

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
                #print ("OOB")
                return None
            if does_path_overlap(pos_array):
                #print ("Overlap")
                return None

            # check to see if we are near target
            dist_to_target = ((pos - next_target) ** 2).sum() ** 0.5
            if dist_to_target < INTERVAL_LEN / 2:
                break

            if sim_steps > MAX_SIM_STEPS:
                print("Exceeded sim steps!")
                return None

    # check if path is "boring"
    if is_path_boring(pos_array, landmark_pos, landmarks_visited,
                      landmark_radii):
        #print ("Boring")
        return None
    elif is_any_point_oob(pos_array):
        return None
    else:
        return pos_array, pos_array[-1], landmarks_visited[-1]


def make_new_curve(config, curve_path, plot_path, start_pos=None, start_landmark=None):
    # try making curve until it succeeds (doesn't go out of bounds)
    pos_array = []
    print("making curve for path=%s" % curve_path)
    # min_num_landmarks = 2
    while True:
        ret = try_make_curve(config, start_pos, start_landmark)
        if ret is not None:
            break
    pos_array, last_pos, last_landmark_visited = ret
    assert isinstance(pos_array, list)

    # write pos array to file
    pos_lists = {"x_array": [x for x, _ in pos_array],
                 "z_array": [y for _, y in pos_array]}
    with open(curve_path, 'w') as fp:
        json.dump(pos_lists, fp)

    landmark_pos = [[float(x), float(y)]
                    for x, y in zip(config["xPos"], config["zPos"])]
    plt.figure()
    axes = plt.axes()
    # plot landmarks
    axes.plot([x for x, _ in landmark_pos],
              [y for _, y in landmark_pos], "bo")
    # plot route starting point
    x_start, y_start = pos_array[0]
    axes.plot([x_start], [y_start], "ro", ms=10.0)
    # plot arrows
    """
    for i, (x, y) in enumerate(pos_array):
        if i == len(pos_array) - 1:
            continue
        x_new, y_new = pos_array[i + 1]
        x_d, y_d = x_new - x, y_new - y
        axes.arrow(x, y, x_d, y_d, head_width=50.0, head_length=80.0,
                   fc="r", ec="r")
    """
    axes.plot([x for x, _ in pos_array[1:]],
              [y for _, y in pos_array[1:]], "r.")
    axes.set_xlim(list(X_RANGE))
    axes.set_ylim(list(Y_RANGE))
    plt.savefig(plot_path)
    return last_pos, last_landmark_visited


def sample_random_landmark_point(current_pos, landmark_pos,
                                 landmarks_already_chosen, landmark_radii):
    landmark_dist = ((landmark_pos - current_pos) ** 2).sum(1) ** 0.5
    landmark_sort_index = [(dist, i) for i, dist in enumerate(landmark_dist)
                           if dist >= MIN_DIST_TO_NEXT]
    landmarks_sorted = [i for _, i in sorted(landmark_sort_index)]
    landmarks_sorted_filtered = [i for i in landmarks_sorted
                                 if i not in landmarks_already_chosen]
    close_landmark_num = CLOSE_LANDMARK_NUM if CLOSE_LANDMARK_NUM >= len(landmarks_sorted) else len(landmarks_sorted) - 1
    landmark_choices = landmarks_sorted_filtered[:close_landmark_num]
    if len(landmark_choices) == 0:
        return None
    next_landmark = random.choice(landmark_choices)
    landmarks_already_chosen.append(next_landmark)
    theta = random.random() * 2 * math.pi
    x, z = landmark_pos[next_landmark]
    landmark_radius = landmark_radii[next_landmark]
    sample_point = np.array([x + math.cos(theta) * landmark_radius,
                             z + math.sin(theta) * landmark_radius])

    if is_any_point_oob([sample_point]):
        return None

    return sample_point


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

def is_any_point_oob(list_or_array_of_points):
    for pt in list_or_array_of_points:
        if pt[0] < DRONE_EDGE_CLEARANCE or \
           pt[0] > 1000 - DRONE_EDGE_CLEARANCE or \
           pt[1] < DRONE_EDGE_CLEARANCE or \
           pt[1] > 1000 - DRONE_EDGE_CLEARANCE:
            return True
    return False

def is_path_boring(pos_array, landmark_pos, landmarks_visited, landmark_radii):
    # make sure we spend a reasonable amount of time circling each landmark
    for landmark_i in landmarks_visited[1:]:
        target_pos = landmark_pos[landmark_i]
        pos_dist = ((pos_array - target_pos) ** 2).sum(1) ** 0.5
        landmark_radius = landmark_radii[landmark_i]
        if (pos_dist < landmark_radius + INTERVAL_LEN).sum() < 4:
            return True
    return False

def generate_random_wrong_path(env_id, start_idx, end_idx):
    env_config = load_env_config(env_id)
    current_path = load_path(env_id)[start_idx:end_idx]

    start_pos = current_path[0]
    landmark_locations = np.asarray(list(zip(env_config["xPos"], env_config["zPos"])))
    distances = np.asarray([np.linalg.norm(start_pos - p) for p in landmark_locations])
    closest_lm_idx = np.argmin(distances)
    start_landmark = env_config["landmarkName"][closest_lm_idx]

    # For segment-level, we're never (if ever) gonna need more than 3 landmarks
    global NUM_LANDMARKS_VISISTED, DRONE_EDGE_CLEARANCE
    NUM_LANDMARKS_VISISTED = 3
    DRONE_EDGE_CLEARANCE = 0

    i = 0
    while True:
        print(f"Attempt: {i}")
        i += 1
        ret = try_make_curve(env_config, start_pos, start_landmark)
        if ret is not None:
            break
    pos_array, last_pos, last_landmark_visited = ret

    # Return a trajectory of the same length as the one which is being replaced
    return convert_path(pos_array[:(end_idx - start_idx)])


if __name__ == "__main__":
    if DEBUG:
        debug()
    else:
        print("main")
        main()

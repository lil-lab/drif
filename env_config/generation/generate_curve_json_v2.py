import os
import random
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
#TODO: Adapt this
from make_random_config import CONFIG_BASE_DIR,\
    X_RANGE, Y_RANGE, LANDMARK_RADII, START_I, END_I, MESH_RES, GENERATE_FOR_TRANSFER

#CURVE_BASE_DIR = "/Users/awbennett/Documents/research_project/unity/ProceduralField/Assets/Resources/paths"
CURVE_BASE_DIR = "/Users/valts/Documents/Cornell/droning/unity/ProceduralField_backup/Assets/Resources/paths"

DEBUG = False

LANDMARK_NAMES = sorted(LANDMARK_RADII)

INTERVAL_LEN = 40
START_EDGE_WIDTH = 45

NUM_LANDMARKS_VISISTED = 5
CLOSE_LANDMARK_NUM = 4

MOVE_DIST = 4
NOISE_VAL = 1.0

SAMPLE_RATE = 20
LAKE_REPULSION_DIST = 120
LAKE_REPULSION = 3.8

MIN_DIST_TO_NEXT = 200

MAX_SIM_STEPS = 500

if GENERATE_FOR_TRANSFER:
    NUM_LANDMARKS_VISISTED = 3


def main():
    for config_id in range(START_I, END_I):
        fname = "random_config_%d.json" % config_id
        path = os.path.join(CONFIG_BASE_DIR, fname)
        print(path)
        with open(path) as fp:
            config = json.load(fp)
        curve_fname = "random_curve_%d.json" % config_id
        curve_path = os.path.join(CURVE_BASE_DIR, curve_fname)
        plot_path = "plots/random_curve_%d.png" % config_id
        make_new_curve(config, curve_path, plot_path)


def debug():
    path = os.path.join(CONFIG_BASE_DIR, "random_config.json")
    with open(path) as fp:
        config = json.load(fp)
    for i in range(10):
        curve_path = "debug_curve.%d.json" % i
        plot_path = "debug_plot.%d.png" % i
        make_new_curve(config, curve_path, plot_path)


def try_make_curve(config):
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
    lake_pos = np.array([[float(c["x"]), float(c["y"])] for c in
                         [config["lakeCoords"][i] for i in
                          range(0, len(config["lakeCoords"]), SAMPLE_RATE)]])
    if len(lake_pos) > 0:
        lake_pos[:, 0] = (lake_pos[:, 0] * (X_RANGE[1] - X_RANGE[0]) / MESH_RES) + X_RANGE[0]
        lake_pos[:, 1] = (lake_pos[:, 1] * (Y_RANGE[1] - Y_RANGE[0]) / MESH_RES) + Y_RANGE[0]
    #print ("Lake Pos: ", len(lake_pos))

    # get starting point and landmark visited
    landmarks_visited = []
    pos = sample_random_landmark_point(pos, landmark_pos, landmarks_visited,
                                       landmark_radii)

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
    else:
        return pos_array


def make_new_curve(config, curve_path, plot_path):
    # try making curve until it succeeds (doesn't go out of bounds)
    pos_array = []
    print("making curve for path=%s" % curve_path)
    # min_num_landmarks = 2
    while True:
        pos_array = try_make_curve(config)
        if pos_array is not None:
            break
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


def is_path_boring(pos_array, landmark_pos, landmarks_visited, landmark_radii):
    # make sure we spend a reasonable amount of time circling each landmark
    for landmark_i in landmarks_visited[1:]:
        target_pos = landmark_pos[landmark_i]
        pos_dist = ((pos_array - target_pos) ** 2).sum(1) ** 0.5
        landmark_radius = landmark_radii[landmark_i]
        if (pos_dist < landmark_radius + INTERVAL_LEN).sum() < 4:
            return True
    return False


if __name__ == "__main__":
    if DEBUG:
        debug()
    else:
        print("main")
        main()

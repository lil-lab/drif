import numpy as np
import scipy.misc as misc
import scipy.ndimage.filters as filters
import random

LAKE_RES = 10
NUM_LANDMARKS = 15
MESH_RES = 100

MIN_NUM_LAKES = 1
MAX_NUM_LAKES = 3

SAMPLING_ATTEMPTS = 1000

"""
This file generates pretty, dynamical lakes that are added to the environments.
The lakes are interspread between the already allocated landmarks, as specified by the config.
"""

def add_lake_to_config(config, x_range, z_range):
    # First extract a list of landmarks and discretize their length
    landmarks = []
    for i in range(len(config["xPos"])):
        pos_x, pos_z = config["xPos"][i], config["zPos"][i]
        idx_x = int((float(pos_x) - x_range[0]) / (x_range[1] - x_range[0]) * LAKE_RES)
        idx_z = int((float(pos_z) - z_range[0]) / (z_range[1] - z_range[0]) * LAKE_RES)
        landmarks.append([idx_x, idx_z])

    num_lakes = int(random.uniform(MIN_NUM_LAKES, MAX_NUM_LAKES + 0.99))

    all_lake_coords = []
    for i in range(num_lakes):
        lake_img = __make_lake(landmarks)
        if lake_img is not None:
            lake_coords_np = np.argwhere(lake_img > 0.5)
            for coord in lake_coords_np:
                coord = list(coord)
                if coord not in all_lake_coords:
                    all_lake_coords.append(coord)
        else:
            print ("FAILED TO MAKE LAKE!")
    print ("Made " + str(num_lakes) + " lakes")
    for lake_coord in all_lake_coords:
        config["lakeCoords"].append({"x": int(lake_coord[0]), "y": int(lake_coord[1])})

    # Debugging:
    #paint_landmarks(lake_img, landmarks, scale=10)
    #plt.imshow(lake_img)
    #plt.show()
    return config


def __random_point(res):
    return [int(random.uniform(2, res - 2)), int(random.uniform(2, res - 2))]


def __upsample(land_img, res):
    big_img = misc.imresize(land_img, (res, res), interp="nearest")
    return big_img


def __gaussian_filter(lake_img):
    filter_size = random.uniform(4, 10)
    #print ("Filter size: ", filter_size)
    return filters.gaussian_filter(lake_img, filter_size, mode="nearest")


def __collides_with_landmark(cell, landmarks):
    dists = [np.linalg.norm(np.asarray(landmark) - cell) for landmark in landmarks]
    min_dist = min(dists)
    return min_dist < 2


def __sample_initial_cell(landmarks, res):
    for i in range(SAMPLING_ATTEMPTS):
        proposal = np.asarray(__random_point(LAKE_RES))
        if not __collides_with_landmark(proposal, landmarks):
            return proposal
        #else:
        #    print ("Reject initial cell: ", proposal)
    return None


def __is_cell_valid(cell, res):
    valid = cell[0] >= 0 and cell[1] >= 0 and cell[0] < res and cell[1] < res
    return valid


def __contains_cell(lake, cell):
    for lake_cell in lake:
        if lake_cell[0] == cell[0] and lake_cell[1] == cell[1]:
            # print ("Lake contains: ", cell)
            return True
    return False


def __generate_new_cell(lake, res):
    proposed_cell = None
    while True:
        index = int(random.uniform(0, len(lake)))
        lake_cell = lake[index]
        rand_x = random.uniform(0, 4)
        x_offset = -1 if rand_x < 1 else 1 if rand_x > 3 else 0
        rand_y = random.uniform(0, 3)
        y_offset = (-1 if rand_y < 1 else 1 if rand_x > 2 else 0) if x_offset == 0 else 0
        proposed_cell = np.asarray(lake_cell) + np.asarray([x_offset, y_offset])
        if __is_cell_valid(proposed_cell, LAKE_RES) and not __contains_cell(lake, proposed_cell):
            break
    return proposed_cell


def __paint_lake_on_img(land_img, lake):
    for cell in lake:
        land_img[cell[0]][cell[1]] = 1


def __paing_landmarks_on_img(land_img, landmarks, scale=1):
    for landmark in landmarks:
        land_img[landmark[0] * scale][landmark[1] * scale] = 3


def __generate_lake(landmarks):
    initial_cell = __sample_initial_cell(landmarks, LAKE_RES)
    if initial_cell is None:
        return None
    lake = [initial_cell]
    lake_size = int(random.uniform(5, 50))
    for i in range(lake_size):
        neighbour_cell = __generate_new_cell(lake, LAKE_RES)
        if not __collides_with_landmark(neighbour_cell, landmarks):
            lake.append(neighbour_cell)
    #print ("Lake size: ", lake_size, len(lake))
    return np.asarray(lake)


def __make_lake(landmarks):
    lake = __generate_lake(landmarks)
    if lake is None:
        return None

    lake_proc = np.zeros((LAKE_RES, LAKE_RES))
    __paint_lake_on_img(lake_proc, lake)
    # paint_landmarks (lake_proc, landmarks)

    upsampled = __upsample(lake_proc, MESH_RES)
    filtered = __gaussian_filter(upsampled)
    T = 0.5
    threshold = np.max(filtered) * T
    thesholded = np.where(filtered > threshold, 1, 0)
    return thesholded


import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import parameters.parameter_server as P

from visualization import Presenter

GRID_RES = 20
CLEARANCE_M = 1

def _landmark_cost_grid(landmark_locations):
    """
    Assign cost to areas near landmarks following gaussian distributions
    :param landmark_locations:
    :return:
    """
    cost_grid = np.zeros((GRID_RES, GRID_RES))
    world_size = P.get_current_parameters()["Setup"]["world_size_m"]
    for lm_coord in landmark_locations:
        lm_x = lm_coord[0]
        lm_y = lm_coord[1]

        lm_coord_x = int(lm_x * GRID_RES / world_size)
        lm_coord_y = int(lm_y * GRID_RES / world_size)
        cost_grid[lm_coord_x, lm_coord_y] = 1.0

    clearance_grid = int(CLEARANCE_M * GRID_RES / world_size)

    # Gaussian blur
    cost_grid = gaussian_filter(cost_grid, clearance_grid)

    # Normalize in the range of 0-1
    cost_grid -= cost_grid.min()
    cost_grid /= (cost_grid.max() + 1e-9)

    return cost_grid


def _drone_pos_cost_grid(curr_pos):
    """
    Assign higher cost to areas further from the drone
    :param curr_pos:
    :return:
    """
    world_size = P.get_current_parameters()["Setup"]["world_size_m"]
    c = np.linspace(0, world_size, 20)
    coord_grid = np.asarray(np.meshgrid(c, c))
    curr_pos = np.asarray(curr_pos[:2])
    vec_to_drone = coord_grid - curr_pos[:, np.newaxis, np.newaxis]
    dst_to_drone = np.linalg.norm(vec_to_drone, axis=0, keepdims=False)
    abs_dst_to_drone = np.abs(dst_to_drone)

    abs_dst_to_drone -= np.min(abs_dst_to_drone)
    abs_dst_to_drone /= (np.max(abs_dst_to_drone) + 1e-9)

    return abs_dst_to_drone

def _cage_boundary_cost_grid():
    cost_grid = np.zeros((GRID_RES, GRID_RES))
    cost_grid[0,:] = 1.0
    cost_grid[-1,:] = 1.0
    cost_grid[:, 0] = 1.0
    cost_grid[:, -1] = 1.0

    world_size = P.get_current_parameters()["Setup"]["world_size_m"]
    clearance_grid = int(CLEARANCE_M * GRID_RES / world_size)
    cost_grid = gaussian_filter(cost_grid, clearance_grid)

    # Normalize in the range of 0-1
    cost_grid -= cost_grid.min()
    cost_grid /= (cost_grid.max() + 1e-9)
    return cost_grid

def find_safe_landing_location(config_a, config_b=None, current_pos=None):
    """
    Finds a safe landing location that does not intersect any of the landmarks in either of the configs
    and is close to the drone's current position
    :param config_a: the first config (e.g. current config)
    :param config_b: the second config (e.g. what we're reconfiguring to)
    :param curr_pos: 2D position of where the drone currently is
    :return:
    """
    world_size = P.get_current_parameters()["Setup"]["world_size_m"]
    landmark_locations = []
    for i, lm_name in enumerate(config_a["landmarkName"]):
        x = config_a["x_pos_as"][i]
        y = config_a["y_pos_as"][i]
        landmark_locations.append([x,y])
    if config_b:
        for i, lm_name in enumerate(config_b["landmarkName"]):
            x = config_a["x_pos_as"][i]
            y = config_a["y_pos_as"][i]
            landmark_locations.append([x, y])

    landmark_cost_grid = _landmark_cost_grid(landmark_locations)
    if current_pos is not None:
        drone_cost_grid = _drone_pos_cost_grid(current_pos)
    else:
        drone_cost_grid = np.zeros((GRID_RES, GRID_RES))

    cage_cost_grid = _cage_boundary_cost_grid()

    # TODO: Maybe it's better to land close to landmarks than close to cage edges
    cost_grid = landmark_cost_grid + 0.5 * drone_cost_grid + cage_cost_grid

    best_loc = np.argmin(cost_grid)
    best_loc = np.asarray(np.unravel_index(best_loc, cost_grid.shape))
    best_loc = best_loc * world_size / GRID_RES

    #cv2.imshow("landing cost grid", cost_grid)
    #cv2.waitKey(0)
    #Presenter().show_image(cost_grid, "landing cost grid", scale=5, waitkey=True)

    return best_loc
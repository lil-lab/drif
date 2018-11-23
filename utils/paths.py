import numpy as np
from scipy.spatial import distance


def condense_path(path):
    """
    The ground truth trajectories often include multiple subsequent points at the same location.
    This function takes a path and returns a path with only unique positions in the original path
    :param path:
    :return:
    """
    out_path = []
    if len(path) == 0:
        return np.asarray(out_path)

    prev_point = path[0]
    out_path.append(prev_point)
    for point in path:
        if np.linalg.norm(point - prev_point) > 1e-9:
            out_path.append(point)
        prev_point = point
    return np.asarray(out_path)


def condense_path_with_mapping(path):
    """
    The ground truth trajectories often include multiple subsequent points at the same location.
    This function takes a path and returns a path with only unique positions in the original path
    It also returns a map (dict) from an index in the previous path to the corresponding index in the new path
    :param path:
    :return:
    """
    out_path = []
    out_map = {}
    if len(path) == 0:
        return np.asarray(out_path), out_map

    prev_point = path[0]
    out_path.append(prev_point)
    out_map[0] = 0

    for i, point in enumerate(path):
        if np.linalg.norm(point - prev_point) > 1e-9:
            out_path.append(point)
        out_map[i] = len(out_path) - 1
        prev_point = point
    return np.asarray(out_path), out_map


def get_closest_point_in_path(path, pos):
    distance_matrix = distance.cdist(path, np.array(pos).reshape(1,2))
    counter = np.argsort(distance_matrix.flatten())[0]
    return counter
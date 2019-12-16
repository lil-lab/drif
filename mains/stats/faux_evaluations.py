import math
import random
from collections import namedtuple

import numpy as np

#from data_io.units import UnrealUnits
# TODO: Refactor away this one
from data_io.env import load_template, load_env_config, load_path
from evaluation.evaluate_t_landmark_side import DataEvalLandmarkSide
from pomdp.state import DroneState
from data_io.instructions import get_correct_eval_env_id_list

import parameters.parameter_server as P

Sample = namedtuple("Sample", ("instruction", "state", "action", "reward", "done", "metadata"))


def bare_min_sample(state, done, env_id):
    act = np.zeros(4)
    if done:
        act[3] = 1.0
    return Sample("none", state, np.zeros(4), 0, done, {"env_id":env_id, "seg_idx": 0, "set_idx": 0, "pol_action":act})


def faux_dataset_random_pt(eval_envs):
    print ("Generating faux dataset")
    #units = UnrealUnits(scale=1.0)
    dataset = []
    for env_id in eval_envs:
        segment_dataset = []
        config = load_env_config(env_id)
        template = load_template(env_id)

        start_pt = np.asarray(config["startPos"])
        second_pt = np.asarray(config["startHeading"])
        end_x = random.uniform(0, 1000)
        end_y = random.uniform(0, 1000)
        end_pt = np.asarray([end_x, end_y])

        state1 = DroneState(None, start_pt)
        state2 = DroneState(None, second_pt)
        state3 = DroneState(None, end_pt)

        segment_dataset.append(bare_min_sample(state1, False, env_id))
        segment_dataset.append(bare_min_sample(state2, False, env_id))
        segment_dataset.append(bare_min_sample(state3, True, env_id))

        dataset.append(segment_dataset)

    return dataset


def faux_dataset_random_landmark(eval_envs):
    print ("Generating faux dataset")
    #units = UnrealUnits(scale=1.0)
    dataset = []
    for env_id in eval_envs:
        segment_dataset = []
        config = load_env_config(env_id)
        template = load_template(env_id)
        path = load_path(env_id)

        landmark_radii = config["radius"]

        start_pt = np.asarray(config["startPos"])
        second_pt = np.asarray(config["startHeading"])

        landmark_choice_ids = list(range(len(config["landmarkName"])))
        choice_id = random.choice(landmark_choice_ids)

        target_x = config["xPos"][choice_id]
        target_y = config["zPos"][choice_id]
        target_lm_pos = np.asarray([target_x, target_y])

        landmark_dir = target_lm_pos - start_pt

        method = template["side"]

        theta = math.atan2(landmark_dir[1], landmark_dir[0]) + math.pi
        if method == "infront":
            theta = theta
        elif method == "random":
            theta = random.random() * 2 * math.pi
        elif method == "behind":
            theta = theta + math.pi
        elif method == "left":
            theta = theta - math.pi / 2
        elif method == "right":
            theta = theta + math.pi / 2

        x, z = target_lm_pos[0], target_lm_pos[1]
        landmark_radius = landmark_radii[choice_id]
        sample_point = np.array([x + math.cos(theta) * landmark_radius,
                                 z + math.sin(theta) * landmark_radius])

        state1 = DroneState(None, start_pt)
        state2 = DroneState(None, second_pt)
        state3 = DroneState(None, sample_point)

        segment_dataset.append(bare_min_sample(state1, False, env_id))
        segment_dataset.append(bare_min_sample(state2, False, env_id))
        segment_dataset.append(bare_min_sample(state3, True, env_id))

        dataset.append(segment_dataset)

    return dataset


def evaluate():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    # At this point test and dev have been swapped.
    # Whatever we've been developing on called "test" is hereafter called dev
    # Test is the data that hasn't been touched at all
    eval_envs = get_correct_eval_env_id_list()

    dataset = faux_dataset_random_pt(eval_envs)
    #dataset = faux_dataset_random_landmark(eval_envs)

    results = {}
    if setup["eval_landmark_side"]:
        evaler = DataEvalLandmarkSide(setup["run_name"], save_images=False)
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()

    results["all_dist"] = []
    print("Results:", results)


if __name__ == "__main__":
    evaluate()
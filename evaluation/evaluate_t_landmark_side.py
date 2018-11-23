import json
import os

import numpy as np
import math

from evaluation.evaluate_base import EvaluateBase
from evaluation.results_t_landmark_side import ResultsLandmarkSide, K_RATE, K_AVG_DIST
from data_io.env import load_template, load_path, load_env_config
from data_io.instructions import get_all_instructions
from data_io.paths import get_results_path, get_results_dir
from env_config.definitions.nlp_templates import N_SIDES
from utils.logging_summary_writer import LoggingSummaryWriter
from visualization import Presenter

LANDMARK_REGION_RADIUS = 200


class DataEvalLandmarkSide (EvaluateBase):

    def __init__(self, run_name="", save_images=True):
        super(EvaluateBase, self).__init__()
        self.train_i, self.test_i, self.dev_i, corpus = get_all_instructions()
        self.passing_distance = LANDMARK_REGION_RADIUS
        self.results = ResultsLandmarkSide()
        self.presenter = Presenter()
        self.run_name = run_name
        self.save_images = save_images

    def evaluate_dataset(self, list_of_rollouts):
        for rollout in list_of_rollouts:
            self.results += self.evaluate_rollout(rollout)

    def get_landmark_pos(self, env_id):
        template = load_template(env_id)
        config = load_env_config(env_id)
        landmark_idx = config["landmarkName"].index(template["landmark1"])
        pos_x = config["xPos"][landmark_idx]
        pos_y = config["zPos"][landmark_idx]
        landmark_pos = np.asarray([pos_x, pos_y])
        return landmark_pos

    def correct_side(self, rollout, env_id):
        template = load_template(env_id)
        landmark_pos = self.get_landmark_pos(env_id)

        last_pos = rollout[-1].state.get_pos()
        first_pos = rollout[0].state.get_pos()
        dir_landmark = landmark_pos - first_pos

        if len(N_SIDES) == 4:
            dir_lm_to_last = last_pos - landmark_pos
            dir_landmark_norm = dir_landmark / (np.linalg.norm(dir_landmark) + 1e-18)
            dir_ortho_norm = np.asarray([dir_landmark_norm[1], -dir_landmark_norm[0]])

            proj = np.dot(dir_lm_to_last, dir_landmark_norm)
            opp_proj = np.dot(dir_lm_to_last, dir_ortho_norm)

            angle = math.atan2(proj, opp_proj)

            DEG45 = 0.785398
            if template["side"] == "right":
                return -DEG45 < angle < DEG45
            elif template["side"] == "back":
                return DEG45 < angle < 3 * DEG45
            elif template["side"] == "left":
                return 3*DEG45 < angle < math.pi or -math.pi < angle < -3*DEG45
            elif template["side"] == "front":
                return -3 * DEG45 < angle < -DEG45
            else:
                print("Unknown side: ", template["side"])

            print("Angle: ", angle)
        else: # len(N_SIDES) = 2
            dir_end = last_pos - first_pos
            z = np.cross(dir_landmark, dir_end)

            if template["side"] == "left":
                return z > 0
            else:
                return z < 0

    def evaluate_rollout(self, rollout):
        last_sample = rollout[-1]
        env_id = last_sample["metadata"]["env_id"]
        seg_idx = last_sample["metadata"]["seg_idx"]
        set_idx = last_sample["metadata"]["set_idx"]
        # TODO: Allow multiple templates / instructions per env
        path = load_path(env_id)

        end_pos = np.asarray(last_sample["state"].get_pos())
        landmark_pos = self.get_landmark_pos(env_id)

        target_end_pos = np.asarray(path[-1])
        end_goal_dist = np.linalg.norm(end_pos - target_end_pos)
        end_lm_dist = np.linalg.norm(end_pos - landmark_pos)
        correct_landmark_region = end_lm_dist < LANDMARK_REGION_RADIUS
        correct_quadrant = self.correct_side(rollout, env_id)

        if last_sample["metadata"]["pol_action"][3] > 0.5:
            who_stopped = "Policy Stopped"
        elif last_sample["metadata"]["ref_action"][3] > 0.5:
            who_stopped = "Oracle Stopped"
        else:
            who_stopped = "Veered Off"

        success = correct_landmark_region and correct_quadrant

        side_txt = "Correct landmark" if correct_landmark_region else "Wrong landmark"
        result = "Success" if success else "Fail"
        texts = [who_stopped, result, side_txt, "run:" + self.run_name]

        if self.save_images:
            dir = get_results_dir(self.run_name, makedir=True)
            self.presenter.plot_paths(rollout, interactive=False, texts=[])#texts)
            filename = os.path.join(dir, str(env_id) + "_" + str(set_idx) + "_" + str(seg_idx))
            self.presenter.save_plot(filename)
            self.save_results()

        return ResultsLandmarkSide(success, end_goal_dist, correct_landmark_region)

    def write_summaries(self, run_name, name, iteration):
        results_dict = self.get_results()
        writer = LoggingSummaryWriter(log_dir="runs/" + run_name, restore=True)
        if not K_AVG_DIST in results_dict:
            print("nothing to write")
            return
        writer.add_scalar(name + "/avg_dist_to_goal", results_dict[K_AVG_DIST], iteration)
        writer.add_scalar(name + "/success_rate", results_dict[K_RATE], iteration)
        writer.save_spied_values()

    def get_results(self):
        return self.results.get_dict()

    def save_results(self):
        # Write results dict
        path = get_results_path(self.run_name, makedir=True)
        with open(path, "w") as fp:
            json.dump(self.get_results(), fp)
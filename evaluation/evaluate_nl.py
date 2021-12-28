import json
import csv
import os

import numpy as np
import math
import ot

from evaluation.evaluate_base import EvaluateBase
from evaluation.results_t_landmark_side import ResultsLandmarkSide, K_RATE, K_AVG_DIST
from data_io.env import load_template, load_and_convert_path, load_and_convert_env_config
from data_io.instructions import get_all_instructions, seg_idx_to_ordinal, get_instruction_segment, tokenize_instruction, get_word_to_token_map
from data_io.paths import get_results_path, get_results_dir, get_logging_dir
from utils.logging_summary_writer import LoggingSummaryWriter
from visualization import Presenter
from geometry import vec_to_yaw, clip_angle

import matplotlib.pyplot as plt

import parameters.parameter_server as P


class DataEvalNL (EvaluateBase):

    def __init__(self, run_name="", save_images=True, entire_trajectory=True, custom_instr=None, aug_len=None):
        super(EvaluateBase, self).__init__()
        self.train_i, self.test_i, self.dev_i, corpus = get_all_instructions()
        self.all_i = {**self.train_i, **self.test_i, **self.dev_i}
        self.passing_distance = P.get_current_parameters()["Units"]["passing_distance"]
        self.results = ResultsLandmarkSide()
        self.presenter = Presenter()
        self.run_name = run_name
        self.save_images = save_images
        self.entire_trajectory = entire_trajectory
        self.custom_instr = custom_instr
        self.aug_len = aug_len

        self.token2term, self.word2token = get_word_to_token_map(corpus)

        self.visible_map = {}

        self.meta_success_table = [["Segment", "Success"]]

        self.hfov = P.get_current_parameters()["ModelPVN"]["Stage1"]["cam_h_fov"]

    def _has_multiple_segments(self, rollout):
        prev_idx = rollout[0]["metadata"]["seg_idx"] if "metadata" in rollout[0] else rollout[0]["seg_idx"]
        for sample in rollout:
            if "metadata" not in sample:
                sample["metadata"] = sample
            if sample["metadata"]["seg_idx"] != prev_idx:
                return True
        return False

    def _split_rollout_in_segments(self, rollout):
        segments = []
        current_segment = [rollout[0]]
        for sample in rollout[1:]:
            if "metadata" not in sample:
                sample["metadata"] = sample
            if sample["metadata"]["seg_idx"] != current_segment[0]["metadata"]["seg_idx"]:
                segments.append(current_segment)
                current_segment = [sample]
            else:
                current_segment.append(sample)
        segments.append(current_segment)
        return segments

    def _segment_matches_auglen(self, segment):
        if not self.aug_len:
            return True
        env_id = segment[0]["env_id"]
        seg_idx = segment[0]["seg_idx"]
        set_idx = segment[0]["set_idx"]
        instr_seg = get_instruction_segment(env_id, set_idx, seg_idx, all_instr=self.all_i)
        return instr_seg["merge_len"] == self.aug_len

    def evaluate_dataset(self, list_of_rollouts):
        for rollout in list_of_rollouts:
            if len(rollout) == 0:
                continue
            if self._has_multiple_segments(rollout):
                segments = self._split_rollout_in_segments(rollout)
                for segment in segments:
                    if self._segment_matches_auglen(segment):
                        seg_results = self.evaluate_rollout(segment)
                        if seg_results is not None:
                            self.results += seg_results
            else:
                if self._segment_matches_auglen(rollout):
                    seg_results = self.evaluate_rollout(rollout)
                    if seg_results is not None:
                        self.results += seg_results
        self.save_results()

    def rollout_success(self, env_id, set_idx, seg_idx, rollout):
        path = load_and_convert_path(env_id)
        seg_ordinal = seg_idx_to_ordinal(self.all_i[env_id][set_idx]["instructions"], seg_idx)
        path_end_idx = self.all_i[env_id][set_idx]["instructions"][seg_ordinal]["end_idx"]
        if path_end_idx > len(path) - 1:
            path_end_idx = len(path) - 1
        end_pos = np.asarray(rollout[-1]["state"].get_pos_2d())
        target_end_pos = np.asarray(path[path_end_idx])
        end_dist = np.linalg.norm(end_pos - target_end_pos)
        success = end_dist < self.passing_distance
        return success

    def is_goal_visible(self, instr_seg):
        end = np.asarray(instr_seg["end_pos"])
        start = np.asarray(instr_seg["start_pos"])
        vec_start_to_end = end - start
        endp_yaw = vec_to_yaw(vec_start_to_end)
        start_yaw = instr_seg["start_yaw"]
        yaw_diff = endp_yaw - start_yaw
        yaw_diff_abs = math.fabs(clip_angle(yaw_diff))
        goal_visible = 2 * yaw_diff_abs < math.radians(self.hfov)
        return goal_visible

    def _filter_path(self, posseq, dst=0.02):
        """Replace original points in the path with equally spaced points"""
        cumdist = 0
        cumdists = [cumdist]
        for prev_pos, pos in zip(posseq[:-1], posseq[1:]):
            gap = np.linalg.norm(pos - prev_pos)
            cumdist += gap
            cumdists.append(cumdist)

        total_path_length = cumdists[-1]
        p = 0
        ptr = 0
        traj_out = []
        # Add the starting point, and move to the next point
        pt = posseq[ptr]
        traj_out.append(pt)
        p += dst
        # Reconstruct the trajectory with equidistant points of fixed precision.
        while p < total_path_length and ptr < len(posseq):
            # Get how far along until the next point this is
            frac = (p - cumdists[ptr-1]) / (cumdists[ptr] - cumdists[ptr-1] + 1e-10)
            # Grab interpolated intermediate point
            pt = posseq[ptr-1] + (posseq[ptr] - posseq[ptr-1]) * frac
            traj_out.append(pt)
            p += dst
            # Advance past the correct starting point
            while ptr < len(cumdists) and p > cumdists[ptr]:
                ptr += 1

        out = np.asarray(traj_out)

        if False:
            plt = np.zeros((470, 470, 3))
            for pt in posseq:
                pt *= 100
                plt[int(pt[0]):int(pt[0]) + 2, int(pt[1]):int(pt[1]) + 2, 0] = 1.0
            for pt in out:
                pt *= 100
                plt[int(pt[0]):int(pt[0])+2, int(pt[1]):int(pt[1])+2, 2] = 1.0
            Presenter().show_image(plt, "filter_paths", scale=4, waitkey=True)

        return out

    def _calculate_emd(self, exec_path, gt_path):
        exec_len = len(exec_path)
        gt_len = len(gt_path)
        if gt_len == 0:
            return None
        p2p_differences = exec_path[np.newaxis, :, :] - gt_path[:, np.newaxis, :]
        p2p_distances = np.linalg.norm(p2p_differences, axis=2)
        # rows index over ground truth path, columns index over executed path
        # Distribute probability mass of 1 evenly over executed and ground-truth trajectories
        prob_masses_exec = np.asarray([1/float(exec_len + 1e-10)] * exec_len)
        prob_masses_gt = np.asarray([1/float(gt_len + 1e-10)] * gt_len)

        assert np.isclose(prob_masses_exec.sum(), 1.0)
        assert np.isclose(prob_masses_gt.sum(), 1.0)
        #print("ding")
        ot_plan, log = ot.emd(prob_masses_gt, prob_masses_exec, p2p_distances, log=True, numItermax=10000)
        emd = log["cost"]
        assert emd > 0, "There is no way that a drone will perfectly follow a trajectory! Something is wrong. EMD error?"
        return emd

    def evaluate_rollout(self, rollout):
        last_sample = rollout[-1]
        if "metadata" not in last_sample:
            last_sample["metadata"] = last_sample
        env_id = last_sample["metadata"]["env_id"]
        seg_idx = last_sample["metadata"]["seg_idx"]
        set_idx = last_sample["metadata"]["set_idx"]

        path = load_and_convert_path(env_id)

        seg_ordinal = seg_idx_to_ordinal(self.all_i[env_id][set_idx]["instructions"], seg_idx)
        instr_seg = self.all_i[env_id][set_idx]["instructions"][seg_ordinal]

        if self.entire_trajectory:
            path_end_idx = len(path) - 1
            path_start_idx = 0
        else:
            # Find the segment end index
            path_end_idx = self.all_i[env_id][set_idx]["instructions"][seg_ordinal]["end_idx"] + 1
            path_start_idx = self.all_i[env_id][set_idx]["instructions"][seg_ordinal]["start_idx"]
            if path_end_idx > len(path) - 1:
                path_end_idx = len(path) - 1
            if path_end_idx < path_start_idx:
                path_start_idx = path_end_idx

        if path_end_idx == path_start_idx:
            path_end_idx = path_start_idx + 1

        seg_path = path[path_start_idx:path_end_idx]
        goal_visible = self.is_goal_visible(instr_seg)
        self.visible_map[f"{env_id}_{seg_idx}"] = (1 if goal_visible else 0)
        exec_path = np.asarray([r["state"].get_pos_2d() for r in rollout])

        end_pos = np.asarray(exec_path[-1])#["state"].get_pos_2d())
        target_end_pos = np.asarray(seg_path[-1])
        end_dist = np.linalg.norm(end_pos - target_end_pos)
        success = end_dist < self.passing_distance

        # EMD between trajectories, and EMD between start position and trajectory.
        exec_path = self._filter_path(exec_path)
        gt_path = self._filter_path(seg_path)
        emd = self._calculate_emd(exec_path, gt_path)
        stop_emd = self._calculate_emd(exec_path[0:1], gt_path)

        # Success weighted by earth-mover's distance
        nemd = emd / stop_emd
        semd = max((1 if success else 0) * (1 - nemd), 0)

        if last_sample["metadata"]["pol_action"][3] > 0.5:
            who_stopped = "Policy Stopped"
        elif last_sample["metadata"]["ref_action"][3] > 0.5:
            who_stopped = "Oracle Stopped"
        else:
            who_stopped = "Veered Off"

        result = "Success" if success else "Fail"
        texts = [who_stopped, result, "run:" + self.run_name]

        #print(seg_idx, result, semd)
        tok_instruction = tokenize_instruction(instr_seg["instruction"], self.word2token)

        self.meta_success_table.append([f"{env_id}-{seg_idx}", success])

        if self.save_images and emd:
            dir = get_results_dir(self.run_name, makedir=True)
            print("Results dir: ", dir)
            # TODO: Refactor this to not pull path from rollout, but provide it explicitly
            self.presenter.plot_paths(rollout,
                                      segment_path=gt_path,
                                      interactive=False,
                                      texts=texts,
                                      entire_trajectory=self.entire_trajectory,
                                      world_size=P.get_current_parameters()["Setup"]["world_size_m"],
                                      real_drone=P.get_current_parameters()["Setup"]["real_drone"])
            filename = os.path.join(dir, str(env_id) + "_" + str(set_idx) + "_" + str(seg_idx))
            if self.custom_instr is not None:
                filename += "_" + last_sample["metadata"]["instruction"][:24] + "_" + last_sample["metadata"]["instruction"][-16:]
            self.presenter.save_plot(filename)

        #if emd:
        #    self.save_results()

        return ResultsLandmarkSide(success=success, end_dist=end_dist, goal_visible=goal_visible, emd=emd, semd=semd, nemd=nemd, instr_len=len(tok_instruction))

    def write_summaries(self, run_name, name, iteration):
        results_dict = self.get_results()
        writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}", restore=True)
        if not K_AVG_DIST in results_dict:
            print("nothing to write")
            return
        writer.add_scalar(name + "/avg_dist_to_goal", results_dict[K_AVG_DIST], iteration)
        writer.add_scalar(name + "/success_rate", results_dict[K_RATE], iteration)
        #writer.save_spied_values()

    def get_results(self):
        return self.results.get_dict()

    def save_results(self):
        # Write results dict
        path = get_results_path(self.run_name, makedir=True)
        with open(path, "w") as fp:
            json.dump(self.get_results(), fp)

        # Write tabular summary
        meta_path = get_results_path(self.run_name + "_meta", makedir=True).replace(".json", ".csv")
        with open(meta_path, "w") as fp:
            writer = csv.writer(fp)
            writer.writerows(self.meta_success_table)

        # Plot emd vs instruction length
        instr_len_emd = [t[0] for t in self.results.instr_len_emd]
        instr_len_succ = [t[0] for t in self.results.instr_len_succ]
        emds = [t[1] for t in self.results.instr_len_emd]
        succ = [t[1] for t in self.results.instr_len_succ]
        #assert [a[0] == b[0] for a, b in zip(self.results.instr_len_emd, self.results.instr_len_succ)]

        emd_plot_path = get_results_path(self.run_name + "_emd_vs_len", makedir=True).replace(".json", ".png")
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([0, 100])
        axes.set_ylim([0, 2.2])
        plt.scatter(instr_len_emd, emds)
        plt.title("EMD vs instruction length")
        plt.draw()
        plt.savefig(emd_plot_path)
        plt.clf()

        succ_plot_path = get_results_path(self.run_name + "_succ_vs_len", makedir=True).replace(".json", ".png")
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([0, 100])
        axes.set_ylim([0, 1.0])
        plt.scatter(instr_len_succ, succ)
        plt.title("Success vs instruction length")
        plt.draw()
        plt.savefig(succ_plot_path)
        plt.clf()

        emd_vs_len_data_path = get_results_path(self.run_name + "_emd_vs_len", makedir=True).replace(".json", ".npy")
        emd_vs_len = np.asarray((instr_len_emd, emds))
        np.save(emd_vs_len_data_path, emd_vs_len)

        succ_vs_len_data_path = get_results_path(self.run_name + "_succ_vs_len", makedir=True).replace(".json", ".npy")
        succ_vs_len = np.asarray((instr_len_succ, succ))
        np.save(succ_vs_len_data_path, succ_vs_len)
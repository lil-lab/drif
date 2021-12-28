import numpy as np

from evaluation.results_base import EvaluationResults

K_N_SUCCESS = "total_success"
K_N_FAIL = "total_fail"
K_N_SEG = "total_segments"
K_N_LM = "total_correct_landmarks"
K_RATE = "%success"
K_RATE_LM = "%correct_landmark"
K_DIST = "total_dist"
K_AVG_DIST = "avg_dist"

K_EMD = "emd"
K_AVG_EMD = "avg_emd"
K_VIS_EMD = "visible_goal_emd"
K_INVIS_EMD = "invisible_goal_emd"
K_VIS_AVG_EMD = "visible_goal_avg_emd"
K_INVIS_AVG_EMD = "invisible_goal_avg_emd"

K_SEMD = "semd"
K_AVG_SEMD = "avg_semd"
K_VIS_SEMD = "visible_goal_semd"
K_INVIS_SEMD = "invisible_goal_semd"
K_VIS_AVG_SEMD = "visible_goal_avg_semd"
K_INVIS_AVG_SEMD = "invisible_goal_avg_semd"

K_NEMD = "nemd"
K_AVG_NEMD = "avg_nemd"
K_VIS_NEMD = "visible_goal_nemd"
K_INVIS_NEMD = "invisible_goal_nemd"
K_VIS_AVG_NEMD = "visible_goal_avg_nemd"
K_INVIS_AVG_NEMD = "invisible_goal_avg_nemd"

K_N_VIS = "visible_goal_segments"
K_N_VIS_SUCCESS = "visible_goal_segments_correct"
K_N_INVIS = "invisible_goal_segments"
K_N_INVIS_SUCCESS = "invisible_goal_segments_correct"

K_RATE_VIS_SUCCESS = "visible_goal_success_rate"
K_RATE_INVIS_SUCCESS = "invisible_goal_success_rate"
K_RATE_VIS = "visible_goal_rate"

K_LAST_DIST = "last_dist"

K_MEDIAN_DIST = "median_dist"
K_ALL_DIST = "all_dist"


class ResultsLandmarkSide(EvaluationResults):

    def __init__(self, success=None, end_dist=0, goal_visible=False, correct_landmark=False, emd=0, semd=0, nemd=0, instr_len=None):
        super(EvaluationResults, self).__init__()
        self.state = {
            K_N_SUCCESS: 1 if success else 0 if success is not None else 0,
            K_N_FAIL: 0 if success else 1 if success is not None else 0,
            K_N_SEG: 1 if success is not None else 0,
            K_RATE: 1.0 if success else 0.0 if success is not None else 0,
            K_N_LM: 1 if correct_landmark else 0,
            K_N_VIS: 1 if goal_visible else 0,
            K_N_INVIS: 0 if goal_visible else 1 if success is not None else 0,
            K_N_VIS_SUCCESS: 1 if (goal_visible and success) else 0,
            K_N_INVIS_SUCCESS: 1 if ((not goal_visible) and success) else 0,
            K_EMD: emd,
            K_VIS_EMD: emd if goal_visible else 0,
            K_INVIS_EMD: emd if (not goal_visible) else 0,
            K_DIST: end_dist,
            K_LAST_DIST: end_dist,
            K_SEMD: semd,
            K_VIS_SEMD: semd if goal_visible else 0,
            K_INVIS_SEMD: semd if (not goal_visible) else 0,
            K_NEMD: nemd,
            K_VIS_NEMD: nemd if goal_visible else 0,
            K_INVIS_NEMD: nemd if (not goal_visible) else 0,
        }
        self.metastate_distances = [end_dist]
        if instr_len is not None:
            self.instr_len_emd = [(instr_len, emd)]
            self.instr_len_succ = [(instr_len, 1 if success else 0)]
        else:
            self.instr_len_emd = []
            self.instr_len_succ = []

    def __add__(self, past_results):
        # Ignore results if EMD is None - it means ground truth path length is zero and there is no example
        if past_results.state[K_EMD] is None:
            print("Warning! ignoring results due to None emd")
            return self

        self.state[K_N_SUCCESS] += past_results.state[K_N_SUCCESS]
        self.state[K_N_FAIL] += past_results.state[K_N_FAIL]
        self.state[K_N_SEG] += past_results.state[K_N_SEG]
        self.state[K_RATE] = self.state[K_N_SUCCESS] / (self.state[K_N_SEG] + 1e-28)

        self.state[K_DIST] += past_results.state[K_DIST]
        self.state[K_AVG_DIST] = self.state[K_DIST] / (self.state[K_N_SEG] + 1e-28)
        self.state[K_N_LM] = self.state[K_N_LM] + past_results.state[K_N_LM]
        self.state[K_RATE_LM] = self.state[K_N_LM] / (self.state[K_N_SEG] + 1e-28)

        self.state[K_N_VIS] = self.state[K_N_VIS] + past_results.state[K_N_VIS]
        self.state[K_N_INVIS] = self.state[K_N_INVIS] + past_results.state[K_N_INVIS]
        self.state[K_N_VIS_SUCCESS] = self.state[K_N_VIS_SUCCESS] + past_results.state[K_N_VIS_SUCCESS]
        self.state[K_N_INVIS_SUCCESS] = self.state[K_N_INVIS_SUCCESS] + past_results.state[K_N_INVIS_SUCCESS]

        self.state[K_RATE_VIS_SUCCESS] = self.state[K_N_VIS_SUCCESS] / (self.state[K_N_VIS] + 1e-28)
        self.state[K_RATE_INVIS_SUCCESS] = self.state[K_N_INVIS_SUCCESS] / (self.state[K_N_INVIS] + 1e-28)
        self.state[K_RATE_VIS] = self.state[K_N_VIS] / (self.state[K_N_SEG] + 1e-28)

        self.state[K_EMD] = self.state[K_EMD] + past_results.state[K_EMD]
        self.state[K_VIS_EMD] = self.state[K_VIS_EMD] + past_results.state[K_VIS_EMD]
        self.state[K_INVIS_EMD] = self.state[K_INVIS_EMD] + past_results.state[K_INVIS_EMD]
        self.state[K_AVG_EMD] = self.state[K_EMD] / (self.state[K_N_SEG] + 1e-28)
        self.state[K_VIS_AVG_EMD] = self.state[K_VIS_EMD] / (self.state[K_N_VIS] + 1e-28)
        self.state[K_INVIS_AVG_EMD] = self.state[K_INVIS_EMD] / (self.state[K_N_INVIS] + 1e-28)

        self.state[K_SEMD] = self.state[K_SEMD] + past_results.state[K_SEMD]
        self.state[K_VIS_SEMD] = self.state[K_VIS_SEMD] + past_results.state[K_VIS_SEMD]
        self.state[K_INVIS_SEMD] = self.state[K_INVIS_SEMD] + past_results.state[K_INVIS_SEMD]
        self.state[K_AVG_SEMD] = self.state[K_SEMD] / (self.state[K_N_SEG] + 1e-28)
        self.state[K_VIS_AVG_SEMD] = self.state[K_VIS_SEMD] / (self.state[K_N_VIS] + 1e-28)
        self.state[K_INVIS_AVG_SEMD] = self.state[K_INVIS_SEMD] / (self.state[K_N_INVIS] + 1e-28)

        self.state[K_NEMD] = self.state[K_NEMD] + past_results.state[K_NEMD]
        self.state[K_VIS_NEMD] = self.state[K_VIS_NEMD] + past_results.state[K_VIS_NEMD]
        self.state[K_INVIS_NEMD] = self.state[K_INVIS_NEMD] + past_results.state[K_INVIS_NEMD]
        self.state[K_AVG_NEMD] = self.state[K_NEMD] / (self.state[K_N_SEG] + 1e-28)
        self.state[K_VIS_AVG_NEMD] = self.state[K_VIS_NEMD] / (self.state[K_N_VIS] + 1e-28)
        self.state[K_INVIS_AVG_NEMD] = self.state[K_INVIS_NEMD] / (self.state[K_N_INVIS] + 1e-28)

        self.metastate_distances += past_results.metastate_distances

        self.instr_len_emd = self.instr_len_emd + past_results.instr_len_emd
        self.instr_len_succ = self.instr_len_succ + past_results.instr_len_succ

        return self

    def get_dict(self):
        self.state[K_MEDIAN_DIST] = np.median(np.asarray(self.metastate_distances[1:])) if len(self.metastate_distances) > 1 else 0.0
        #self.state[K_ALL_DIST] = self.metastate_distances
        return self.state

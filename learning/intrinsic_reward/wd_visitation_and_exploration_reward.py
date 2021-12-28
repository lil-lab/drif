import torch
import ot
import numpy as np
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward
import learning.datasets.top_down_dataset as tdd

import transformations


class WDVisitationAndExplorationReward(AbstractIntrinsicReward):
    def __init__(self, world_size_m, world_size_px, params):
        super(WDVisitationAndExplorationReward, self).__init__()
        self.world_size_m = world_size_m
        self.world_size_px = world_size_px
        self.params = params

        self.visit_alpha = params.get("visit_alpha", 0.3)
        self.stop_alpha = params.get("stop_alpha", 0.5)
        self.stop_p_alpha = params.get("stop_p_alpha", 0.5)
        self.exploration_alpha = params.get("exploration_alpha", 3.0)
        self.stop_oob_alpha = params.get("stop_oob_alpha", 1.0)

        self.distance_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        coords_vec = np.linspace(0, self.world_size_px-1, self.world_size_px)
        coords_grid = np.asarray(np.meshgrid(coords_vec, coords_vec))
        coords_grid = coords_grid[[1, 0], :, :]
        coords_grid = coords_grid.transpose((1, 2, 0))
        coords_grid_flat = np.reshape(coords_grid, [-1, 2])
        pt_mat_a = np.tile(coords_grid_flat[:, np.newaxis, :], [1, coords_grid_flat.shape[0], 1])
        pt_mat_b = pt_mat_a.transpose((1, 0, 2))
        distances = np.linalg.norm(pt_mat_a - pt_mat_b, axis=2)
        return distances

    def wasserstein_distance(self, dist_a, dist_b):
        #start = time.time()
        p_flat = dist_a.reshape([-1])
        q_flat = dist_b.reshape([-1])

        q_flat = q_flat.astype(np.float64)
        p_flat = p_flat.astype(np.float64)

        q_flat /= q_flat.sum()
        p_flat /= p_flat.sum()

        ot_plan, log = ot.emd(p_flat, q_flat, self.distance_matrix, log=True, numItermax=10000)
        wd = log["cost"]
        #end=time.time()
        #print(f"Time taken: {end - start}")
        return wd

    def empty_distribution(self):
        return np.zeros([self.world_size_px, self.world_size_px])

    def get_reward(self, model_state, v_dist_w, cam_pos, action, done, first):
        # Prepare things
        pos_in_map_m = cam_pos[0:1, 0:2]# * self.world_size_px / self.
        pos_in_map_px = torch.from_numpy(transformations.pos_m_to_px(pos_in_map_m.detach().cpu().numpy(),
                                                                     self.world_size_px,
                                                                     self.world_size_m,
                                                                     self.world_size_px))[0]
        prev_pos = model_state.get("prev_pos", pos_in_map_px)

        # Empirical distribution
        visited_dist = model_state.get("visited_dist", None)
        if visited_dist is None:
            visited_dist = self.empty_distribution()
        visited_dist = tdd.plot_path_on_img(visited_dist, [prev_pos, pos_in_map_px])
        model_state.put("visited_dist", visited_dist)

        # Predicted (desired) distribution
        visit_dist = v_dist_w.inner_distribution[0, 0, :, :]
        visit_dist = visit_dist.detach().cpu().numpy()
        goal_unobserved_prob = v_dist_w.outer_prob_mass[0,1].item()
        goal_observed_prob = 1 - goal_unobserved_prob

        # -----------------------------------------------------------------------
        # Calculate exploration reward, using probability that goal is observed as a potential function

        # Don't ever reduce the potential - only increase it
        exploration_potential = goal_observed_prob
        prev_exploration_potential = model_state.get("prev_exploration_potential", exploration_potential)
        exploration_potential = max(exploration_potential, prev_exploration_potential)
        exploration_reward = (exploration_potential - prev_exploration_potential) * self.exploration_alpha
        model_state.put("prev_exploration_potential", exploration_potential)

        # -----------------------------------------------------------------------
        # Calculate visitation reward (potential shaped by visitation probability)
        # Give reward for visiting the high-probability states at next timestep
        visit_potential = -self.wasserstein_distance(visited_dist, visit_dist)
        prev_visit_potential = model_state.get("prev_visit_potential", visit_potential)
        visit_reward = (visit_potential - prev_visit_potential) * self.visit_alpha
        model_state.put("prev_visit_potential", visit_potential)

        # -----------------------------------------------------------------------
        # Calculate stop reward consisting of EMD(stop,goal), P(stop=goal), and -P(stop_oob)
        if action[3] > 0.5 or done:
            partial_stop_dist = v_dist_w.inner_distribution[0, 1, :, :].detach().cpu().numpy()
            stopped_dist = tdd.plot_path_on_img(self.empty_distribution(), [pos_in_map_px, pos_in_map_px])
            stop_wd = self.wasserstein_distance(partial_stop_dist, stopped_dist)
            stop_reward = -stop_wd * self.stop_alpha

            # Calculate reward proportional to P(p_g = p_stop)
            pos_in_map_m = cam_pos[0:1, 0:2]  # * self.world_size_px / self.
            pos_in_map_px = torch.from_numpy(transformations.pos_m_to_px(pos_in_map_m.detach().cpu().numpy(),
                                                                         self.world_size_px,
                                                                         self.world_size_m,
                                                                         self.world_size_px))
            pos_x = int(pos_in_map_px[0, 0].item() + 0.5)
            pos_y = int(pos_in_map_px[0, 1].item() + 0.5)
            pos_x = min(max(pos_x, 0), partial_stop_dist.shape[0] - 1)
            pos_y = min(max(pos_y, 0), partial_stop_dist.shape[1] - 1)
            stop_prob_at_pos = partial_stop_dist[pos_x, pos_y].item()
            stop_prob_prop = stop_prob_at_pos * goal_observed_prob / (partial_stop_dist.max() + 1e-10)
            stop_p_reward = stop_prob_prop * self.stop_p_alpha

            # Add negative reward for stopping when P(goal oob) is high
            stop_oob_reward = -self.stop_oob_alpha * goal_unobserved_prob
        else:
            stop_reward = 0.0
            stop_p_reward = 0.0
            stop_oob_reward = 0.0

        # -----------------------------------------------------------------------
        model_state.put("prev_pos", pos_in_map_px)

        return visit_reward, stop_reward, exploration_reward, stop_oob_reward, stop_p_reward

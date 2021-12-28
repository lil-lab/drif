import torch
import numpy as np

from data_io.model_io import load_pytorch_model
from learning.models.model_pvn_stage1_bidomain import PVN_Stage1_Bidomain
from learning.models.model_rpn_fewshot_stage1_bidomain import RPN_FewShot_Stage1_Bidomain
from learning.models.model_pvn_stage2_keyboard import PVN_Stage2_Keyboard
from learning.models.model_pvn_stage2_actor_critic import PVN_Stage2_ActorCritic
from learning.modules.map_transformer import MapTransformer
from learning.modules.structured_map_layers import StructuredMapLayers
from learning.intrinsic_reward.wd_visitation_and_exploration_reward import WDVisitationAndExplorationReward
from learning.intrinsic_reward.action_oob_reward import ActionOutOfBoundsReward
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.models.navigation_model_component_base import NavigationModelComponentBase
from learning.modules.generic_model_state import GenericModelState

import learning.datasets.aux_data_providers as aup

try:
    from drones.aero_interface.rviz import RvizInterface
except Exception as e:
    print("NO ROS")

from utils.simple_profiler import SimpleProfiler

import parameters.parameter_server as P

from visualization import Presenter

ACTPROF = False


class FS_PVN_Wrapper_Bidomain(NavigationModelComponentBase):

    def __init__(self, run_name="", model_instance_name="only", oracle_stage1=False):
        super(FS_PVN_Wrapper_Bidomain, self).__init__(run_name, domain="sim", model_name="rpn_fspvn_full")
        self.instance_name = model_instance_name

        self.s1_params = P.get_current_parameters()["ModelPVN"]["Stage1"]
        self.s2_params = P.get_current_parameters()["ModelPVN"]["Stage2"]
        self.wrapper_params = P.get_current_parameters()["PVNWrapper"]
        self.oracle_stage1 = oracle_stage1

        self.real_drone = P.get_current_parameters()["Setup"]["real_drone"]
        self.rviz = None
        if self.real_drone and P.get_current_parameters()["Setup"].get("use_rviz", False):
            self.rviz = RvizInterface(
                base_name="/pvn/",
                map_topics=["semantic_map", "visitation_dist"],
                markerarray_topics=["instruction"])

        s1_model_type = self.wrapper_params.get("stage1_model_type")
        if s1_model_type == "PVNv2":
            self.stage1_visitation_prediction : NavigationModelComponentBase = PVN_Stage1_Bidomain(run_name, model_instance_name)
        elif s1_model_type == "RPN_FSPVN":
            self.stage1_visitation_prediction : NavigationModelComponentBase = RPN_FewShot_Stage1_Bidomain(run_name, model_instance_name)
        else:
            raise ValueError(f"Unsupported Stage 1 model type: {s1_model_type}")

        self.keyboard = self.wrapper_params.get("keyboard")

        if self.keyboard:
            self.stage2_action_generation = PVN_Stage2_Keyboard(run_name, model_instance_name)
        else:
            # Wrap Stage 1 to hide it from PyTorch
            self.stage2_action_generation = PVN_Stage2_ActorCritic(run_name, model_instance_name)

        self.wd_visitation_and_exploration_reward = WDVisitationAndExplorationReward(
            world_size_px=self.s1_params["world_size_px"],
            world_size_m=self.s1_params["world_size_m"],
            params=self.wrapper_params["wd_reward"])
        self.action_oob_reward = ActionOutOfBoundsReward()

        # Create a single model state and share it with submodels
        self.initialize_empty_model_state()
        self.structured_layers = StructuredMapLayers(self.s1_params["global_map_size"])

        self.load_models_from_file()

        self.map_transformer_w_to_r = MapTransformer(
            source_map_size=self.s1_params["global_map_size"],
            dest_map_size=self.s1_params["local_map_size"],
            world_size_m=self.s1_params["world_size_m"],
            world_size_px=self.s1_params["world_size_px"]
        )

        self.presenter = Presenter()

        self.actprof = SimpleProfiler(print=ACTPROF, torch_sync=ACTPROF)

    def make_picklable(self):
        super().make_picklable()
        self.stage1_visitation_prediction.make_picklable()
        self.stage2_action_generation.make_picklable()

    def load_models_from_file(self, stage1_file=None, stage2_file=None):
        stage1_file = stage1_file or self.wrapper_params.get("stage1_file")
        stage2_file = stage2_file or self.wrapper_params.get("stage2_file")
        if stage1_file:
            print("PVNWrapper: Loading Stage 1")
            try:
                load_pytorch_model(self.stage1_visitation_prediction, stage1_file)
            except RuntimeError as e:
                print(f"Couldn't load Stage1 without namespace: {e}")
                load_pytorch_model(self.stage1_visitation_prediction, stage1_file, namespace="stage1_visitation_prediction")
        if stage2_file:
            print("PVNWrapper: Loading Stage 2")
            try:
                load_pytorch_model(self.stage2_action_generation, stage2_file)
            except RuntimeError as e:
                print(f"Couldn't load Stage2 without namespace: {e}")
                load_pytorch_model(self.stage2_action_generation, stage2_file, namespace="stage2_action_generation")

    # Policy state is whatever needs to be updated during RL training.
    # Right now we only update the stage 2 weights.
    def get_policy_state(self):
        return self.stage2_action_generation.state_dict()

    def set_policy_state(self, state):
        self.stage2_action_generation.load_state_dict(state)

    def get_static_state(self):
        return self.stage1_visitation_prediction.state_dict()

    def set_static_state(self, state):
        print("Setting Stage1 Policy State")
        self.stage1_visitation_prediction.load_state_dict(state)

    def init_weights(self):
        super().init_weights()
        self.stage1_visitation_prediction.init_weights()
        self.stage2_action_generation.init_weights()
        self.load_models_from_file()

    def calc_intrinsic_rewards(self, next_state, action, done, first):
        if self.model_state.get("v_dist_w", None) is None or self.model_state.get("map_uncoverage_w", None) is None:
            raise ValueError("Computing intrinsic reward prior to any rollouts!")
        else:
            states_np = next_state.state[np.newaxis, :]
            states = torch.from_numpy(states_np)
            cam_pos = states[:, 0:12]
            v_dist_w = self.model_state.get("v_dist_w")

            visitation_reward, stop_reward, exploration_reward, stop_oob_reward, stop_p_reward = (
                self.wd_visitation_and_exploration_reward(self.model_state, v_dist_w, cam_pos, action, done, first))

            if self.wrapper_params.get("explore_reward_only"):
                visitation_reward = 0.0
                stop_reward = 0.0
            negative_per_step_reward = -self.wrapper_params["wd_reward"]["step_alpha"]
            action_oob_reward = self.action_oob_reward.get_reward(action)
            return {"visitation_reward": visitation_reward,
                    "stop_reward": stop_reward,
                    "exploration_reward": exploration_reward,
                    "negative_per_step_reward": negative_per_step_reward,
                    "action_oob_reward": action_oob_reward,
                    "stop_oob_reward": stop_oob_reward,
                    "stop_p_reward": stop_p_reward}

    def extract_viz_data(self):
        # TODO: Delete this function - obsolete
        last_pre_outputs = self.stage1_visitation_prediction.last_pre_outputs
        ts = self.stage1_visitation_prediction.tensor_store
        print("ding")
        viz_data = {}
        viz_data["object_database"] = self.model_state.get("object_database")
        viz_data["anon_instruction_nl"] = last_pre_outputs["anon_instr_nl"]
        viz_data["obj_ref"] = last_pre_outputs["obj_ref"]
        viz_data["noun_chunks"] = last_pre_outputs["noun_chunks"]
        viz_data["similarity_matrix"] = last_pre_outputs["similarity_matrix"]
        viz_data["image"] = ts.get_latest_input("fpv").cpu()
        viz_data["visual_similarity_matrix"] = ts.get_latest_input("visual_similarity_matrix").cpu()
        viz_data["grounding_matrix"] = ts.get_latest_input("grounding_matrix").cpu()
        viz_data["region_crops"] = ts.get_latest_input("region_crops").cpu()
        viz_data["full_masks_fpv"] = ts.get_latest_input("full_masks_fpv").cpu()
        viz_data["accum_full_masks_w"] = ts.get_latest_input("accum_full_masks_w")[0].cpu()
        viz_data["log_v_dist_w"] = ts.get_latest_input("log_v_dist_w")[0].cpu()
        return viz_data

    def initialize_empty_model_state(self):
        new_model_state = GenericModelState()
        self.set_model_state(new_model_state)
        self.stage1_visitation_prediction.set_model_state(new_model_state)
        self.stage2_action_generation.set_model_state(new_model_state)

    def start_rollout(self, env_id, set_idx, seg_idx):
        super()
        self.initialize_empty_model_state()
        self.model_state.put("env_id", env_id)
        self.model_state.put("seg_idx", seg_idx)
        self.model_state.put("set_idx", set_idx)
        self.stage1_visitation_prediction.start_rollout(env_id, set_idx, seg_idx)

    def get_action(self, state, instruction_str, sample=False, rl_rollout=False):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction_str: RAW instruction given the corpus
        :param sample: If true, sample action from action distribution. If False, take most likely action.
        #TODO: Absorb corpus within model
        :return:
        """
        self.actprof.tick("out")
        device = next(self.parameters()).device
        loop_profiler = SimpleProfiler(torch_sync=False, print=False)
        assert isinstance(instruction_str, str), "From now on, raw string instructions are fed into models!"
        self.eval()

        with torch.no_grad():
            states, images_fpv = self.states_to_torch(state)

            first_step = self.model_state.get("first_step", True)
            self.model_state.put("first_step", False)
            if first_step:
                reset_mask = [True]
                self.model_state.put("start_cam_poses", self.cam_poses_from_states(states).to(device))
                if self.rviz is not None:
                    self.rviz.publish_instruction_text("instruction", instruction_str)
            else:
                reset_mask = [False]

            self.model_state.put("seq_step", self.model_state.get("seq_step", 0) + 1)

            images_fpv = images_fpv.to(device)
            states = states.to(device)
            self.actprof.tick("start")

            start_cam_poses = self.model_state.get("start_cam_poses")
            log_v_dist_w, v_dist_w_poses, rl_outputs = self.stage1_visitation_prediction(
                images_fpv, states, instruction_str, reset_mask,
                noisy_start_poses=start_cam_poses, start_poses=start_cam_poses, rl=True)
            v_dist_w = log_v_dist_w.softmax()
            self.actprof.tick("stage1")

            map_uncoverage_w = rl_outputs["map_uncoverage_w"]
            self.model_state.put("v_dist_w", v_dist_w)
            self.model_state.put("map_uncoverage_w", map_uncoverage_w)

            # TODO: Fix
            if self.rviz: #a.k.a False
                v_dist_w_np = v_dist_w.inner_distribution[0].data.cpu().numpy().transpose(1, 2, 0)
                # expand to 0-1 range
                v_dist_w_np[:, :, 0] /= (np.max(v_dist_w_np[:, :, 0]) + 1e-10)
                v_dist_w_np[:, :, 1] /= (np.max(v_dist_w_np[:, :, 1]) + 1e-10)
                self.rviz.publish_map("visitation_dist", v_dist_w_np,
                                      self.s1_params["world_size_m"])

            # Transform to robot reference frame
            drn_poses = self.poses_from_states(states)
            x = v_dist_w.inner_distribution
            xr, r_poses = self.map_transformer_w_to_r(x, None, drn_poses)
            v_dist_r = Partial2DDistribution(xr, v_dist_w.outer_prob_mass)

            # Add map structured info (simulating laserscans)
            structured_map_info_r, map_info_w = self.structured_layers.build(map_uncoverage_w,
                                                                             drn_poses,
                                                                             self.map_transformer_w_to_r,
                                                                             use_map_boundary=self.s2_params["use_map_boundary"])

            if self.oracle_stage1:
                # Ground truth visitation distributions (in start and global frames)
                env_id, set_idx, seg_idx = self.model_state.get(["env_id", "set_idx", "seg_idx"])
                v_dist_w_gt = aup.resolve_and_get_ground_truth_static_global(env_id, set_idx, seg_idx,
                                                                             self.s1_params["global_map_size"],
                                                                             self.s1_params["world_size_px"]).to(images_fpv.device)
                v_dist_r_ground_truth_select, poses_r = self.map_transformer_w_to_r(v_dist_w_gt, None, drn_poses)
                map_uncoverage_r = structured_map_info_r[:, 0, :, :]
                # PVNv2: Mask the visitation distributions according to observability thus far:
                if self.s1_params["clip_observability"]:
                    v_dist_r_gt_masked = Partial2DDistribution.from_distribution_and_mask(v_dist_r_ground_truth_select, 1 - map_uncoverage_r)
                # PVNv1: Have P(oob)=0, and use unmasked ground-truth visitation distributions
                else:
                    v_dist_r_gt_masked = Partial2DDistribution(v_dist_r_ground_truth_select, torch.zeros_like(v_dist_r_ground_truth_select[:,:,0,0]))
                v_dist_r = v_dist_r_gt_masked

            if True:
                Presenter().show_image(structured_map_info_r, "map_struct", scale=1, waitkey=1)
                v_dist_r.show("v_dist_r", scale=4, waitkey=1)

            # Run stage2 action generation
            self.actprof.tick("pipes")
            # If RL, stage 2 outputs distributions over actions (following torch.distributions API)
            xvel_dist, yawrate_dist, stop_dist, value = self.stage2_action_generation(v_dist_r, structured_map_info_r, eval=True)

            self.actprof.tick("stage2")
            if sample:
                xvel, yawrate, stop = self.stage2_action_generation.sample_action(xvel_dist, yawrate_dist, stop_dist)
            else:
                xvel, yawrate, stop = self.stage2_action_generation.mode_action(xvel_dist, yawrate_dist, stop_dist)

            self.actprof.tick("sample")
            xvel_logprob, yawrate_logprob, stop_logprob = self.stage2_action_generation.action_logprob(xvel_dist, yawrate_dist, stop_dist, xvel, yawrate, stop)

            xvel = xvel.detach().cpu().numpy()
            yawrate = yawrate.detach().cpu().numpy()
            stop = stop.detach().cpu().numpy()
            xvel_logprob = xvel_logprob.detach()
            yawrate_logprob = yawrate_logprob.detach()
            stop_logprob = stop_logprob.detach()
            if value is not None:
                value = value.detach()

            # Add an empty column for sideways velocity
            act = np.concatenate([xvel, np.zeros(xvel.shape), yawrate, stop])

            # Keep all the info we will need later for A2C / PPO training
            # TODO: We assume independence between velocity and stop distributions. Not true, but what ya gonna do?
            rl_data = {
                "stage1_iter": self.stage1_visitation_prediction.get_iter(),
                "policy_input": v_dist_r.detach().cpu(),
                "policy_input_b": structured_map_info_r[0].detach().cpu(),
                "v_dist_w": v_dist_w.inner_distribution[0].detach().cpu(),
                "value_pred": value[0].detach().cpu() if value else None,
                "xvel": xvel,
                "yawrate": yawrate,
                "stop": stop,
                "xvel_logprob": xvel_logprob.detach().cpu(),
                "yawrate_logprob": yawrate_logprob.detach().cpu(),
                "stop_logprob": stop_logprob.detach().cpu(),
                "action_logprob": (xvel_logprob + stop_logprob + yawrate_logprob).detach().cpu()
            }

            loop_profiler.tick("get_action")
            loop_profiler.loop()
            loop_profiler.print_stats(1)
            self.actprof.tick("end")
            self.actprof.loop()
            self.actprof.print_stats(1)
            if rl_rollout:
                return act, rl_data
            elif self.wrapper_params["add_model_state_to_rollout_data"]:
                return act, {"model_state": self.model_state}
            else:
                return act, {}


    def unbatch(self, batch):
        # Inputs
        images = self.to_model_device(batch["images"][0])
        states = self.to_model_device(batch["states"][0])
        actions = self.to_model_device(batch["actions"][0])

        # Ground truth visitation distributions (in start and global frames)
        v_dist_w_ground_truth = self.to_model_device(batch["traj_ground_truth"][0])
        poses = self.poses_from_states(states)
        v_dist_r_ground_truth, poses_r = self.map_transformer_w_to_r(v_dist_w_ground_truth, None, poses)

        return images, states, actions, v_dist_r_ground_truth, poses

    def sup_loss_on_batch(self, batch, eval, halfway=False):
        self.initialize_empty_model_state()

        images, states, actions_gt, v_dist_r_ground_truth, poses = self.unbatch(batch)
        batch_size = images.shape[0]

        # ----------------------------------------------------------------------------
        with torch.no_grad():
            accum_obs_mask_w = self.stage1_visitation_prediction.compute_observability_masks_for_stage2(images, states)
            map_uncoverage_w = 1 - accum_obs_mask_w

        # ----------------------------------------------------------------------------
        poses = self.poses_from_states(states)
        structured_map_layers_r, structured_map_layers_w = self.structured_layers.build(
            map_uncoverage_w, poses, self.map_transformer_w_to_r, use_map_boundary=self.s2_params["use_map_boundary"])
        map_uncoverage_r = structured_map_layers_r[:, 0, :, :]

        v_dist_r_gt_masked = Partial2DDistribution.from_distribution_and_mask(v_dist_r_ground_truth, 1 - map_uncoverage_r)

        xvel_dist, yawrate_dist, stop_dist, value_pred = self.stage2_action_generation(v_dist_r_gt_masked, structured_map_layers_r, eval=False)
        xvel_logprob = xvel_dist.log_probs(actions_gt[:, 0])
        yawrate_logprob = yawrate_dist.log_probs(actions_gt[:, 2])
        # TODO: Figure out why this doesn't already sum
        stop_logprob = stop_dist.log_probs(actions_gt[:, 3]).sum()
        total_logprob = xvel_logprob + yawrate_logprob + stop_logprob

        avg_logprob = total_logprob / batch_size
        avg_xvel_logprob = xvel_logprob / batch_size
        avg_yawrate_logprob = yawrate_logprob / batch_size
        avg_stop_logprob = stop_logprob / batch_size

        squared_xvel_dst = ((xvel_dist.mean - actions_gt[:, 0]) ** 2).mean()
        squared_yawrate_dst = ((yawrate_dist.mean - actions_gt[:, 2]) ** 2).mean()

        action_loss = -avg_stop_logprob + squared_xvel_dst + squared_yawrate_dst

        prefix = self.stage2_action_generation.model_name + ("/eval" if eval else "/train")
        self.stage2_action_generation.writer.add_scalar(prefix + "/action_loss", action_loss.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/avg_logprob", avg_logprob.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/avg_xvel_logprob", avg_xvel_logprob.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/avg_yawrate_logprob", avg_yawrate_logprob.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/avg_stop_logprob", avg_stop_logprob.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/squared_xvel_dst", squared_xvel_dst.data.cpu().item(), self.stage2_action_generation.get_iter())
        self.stage2_action_generation.writer.add_scalar(prefix + "/squared_yawrate_dst", squared_yawrate_dst.data.cpu().item(), self.stage2_action_generation.get_iter())

        #self.writer.add_dict(prefix, get_current_meters(), self.get_iter())
        self.stage2_action_generation.inc_iter()
        return action_loss, None

    def get_dataset(self, *args, **kwargs):
        return self.stage2_action_generation.get_dataset(*args, **kwargs)

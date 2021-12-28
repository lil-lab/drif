import torch
import numpy as np
import torch.nn as nn

from transforms3d.euler import euler2quat

from data_io.models import load_pytorch_model
from data_io.env import load_env_img
from data_io.instructions import debug_untokenize_instruction
from learning.models.model_pvn_stage1_bidomain import PVN_Stage1_Bidomain
from learning.models.model_pvn_stage2_bidomain import PVN_Stage2_Bidomain
from learning.models.model_pvn_stage2_keyboard import PVN_Stage2_Keyboard
from learning.models.model_pvn_stage2_actor_critic import PVN_Stage2_ActorCritic
from learning.modules.map_transformer import MapTransformer
from learning.inputs.pose import Pose
from learning.inputs.vision import standardize_images, standardize_image
from learning.intrinsic_reward.visitation_reward import VisitationReward
from learning.intrinsic_reward.wd_visitation_and_exploration_reward import WDVisitationAndExplorationReward
from learning.intrinsic_reward.map_coverage_reward import MapCoverageReward
from learning.intrinsic_reward.action_oob_reward import ActionOutOfBoundsReward
from learning.intrinsic_reward.visitation_and_exploration_reward import VisitationAndExplorationReward
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.modules.structured_map_layers import StructuredMapLayers
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


class PVN_Wrapper_Bidomain(NavigationModelComponentBase):

    def __init__(self, run_name="", model_instance_name="only", oracle_stage1=False):
        super(PVN_Wrapper_Bidomain, self).__init__(run_name, domain=model_instance_name, model_name="pvnv2")
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

        self.stage1_visitation_prediction = PVN_Stage1_Bidomain(run_name, model_instance_name)

        self.keyboard = self.wrapper_params.get("keyboard")

        if self.keyboard:
            self.stage2_action_generation = PVN_Stage2_Keyboard(run_name, model_instance_name)
        else:
            self.stage2_action_generation = PVN_Stage2_ActorCritic(run_name, model_instance_name)

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

        self.visitation_reward = VisitationReward(world_size_px=self.s1_params["world_size_px"],
                                                  world_size_m=self.s1_params["world_size_m"])
        self.visitation_and_exploration_reward = VisitationAndExplorationReward(world_size_px=self.s1_params["world_size_px"],
                                                  world_size_m=self.s1_params["world_size_m"])
        self.wd_visitation_and_exploration_reward = WDVisitationAndExplorationReward(world_size_px=self.s1_params["world_size_px"],
                                                  world_size_m=self.s1_params["world_size_m"], params=self.wrapper_params["wd_reward"])

        self.map_coverage_reward = MapCoverageReward()
        self.action_oob_reward = ActionOutOfBoundsReward()

        self.map_boundary = self.make_map_boundary()

        #self.prev_instruction = None
        #self.start_cam_poses = None
        #self.seq_step = 0
        #self.log_v_dist_w = None
        #self.v_dist_w = None
        #self.map_uncoverage_w = None
        #self.current_segment = None
        self.presenter = Presenter()
        self.needs_tokenized_str = True

        self.actprof = SimpleProfiler(print=ACTPROF, torch_sync=ACTPROF)

    def make_picklable(self):
        super().make_picklable()
        self.stage1_visitation_prediction.make_picklable()
        self.stage2_action_generation.make_picklable()

    def load_models_from_file(self, stage1_file=None, stage2_file=None):
        #if self.real_drone:
        #    stage1_file = stage1_file or self.wrapper_params.get("stage1_file_real")
        #else:
        #    stage1_file = stage1_file or self.wrapper_params.get("stage1_file_sim")
        #if not stage1_file:
        #    stage1_file = self.wrapper_params.get("stage1_file", None)
        #stage2_file = stage2_file or self.wrapper_params.get("stage2_file")
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

    def parameters(self, recurse=True):
        #print("WARNING: RETURNING STAGE 2 PARAMS AS WRAPPER PARAMS")
        return self.stage2_action_generation.parameters(recurse)

    # Policy state is whatever needs to be updated during RL training.
    # Right now we only update the stage 2 weights.
    def get_policy_state(self):
        return self.stage2_action_generation.state_dict()

    def set_policy_state(self, state):
        self.stage2_action_generation.load_state_dict(state)

    def set_static_state(self, state):
        self.stage1_visitation_prediction.load_state_dict(state)

    def get_static_state(self):
        return self.stage1_visitation_prediction.state_dict()

    def init_weights(self):
        super().init_weights()
        self.stage1_visitation_prediction.init_weights()
        self.stage2_action_generation.init_weights()
        self.load_models_from_file()

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

    def make_map_boundary(self):
        mapsize = self.s1_params["global_map_size"]
        boundary = torch.zeros([1, 1, mapsize, mapsize])
        boundary[:,:,0,:] = 1.0
        boundary[:,:,mapsize-1,:] = 1.0
        boundary[:,:,:,0] = 1.0
        boundary[:,:,:,mapsize-1] = 1.0
        return boundary

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

    def build_map_structured_input(self, map_uncoverage_w, cam_poses):
        map_uncoverage_r, _ = self.map_transformer_w_to_r(map_uncoverage_w, None, cam_poses)
        if self.s2_params["use_map_boundary"]:
            # Change device if necessary
            self.map_boundary = self.map_boundary.to(map_uncoverage_r.device)
            batch_size = map_uncoverage_w.shape[0]
            map_boundary_r, _ = self.map_transformer_w_to_r(self.map_boundary.repeat([batch_size, 1, 1, 1]), None, cam_poses)
            structured_map_info_r = torch.cat([map_uncoverage_r, map_boundary_r], dim=1)
        else:
            structured_map_info_r = map_uncoverage_r

        batch_size = map_uncoverage_w.shape[0]
        struct_info_w = torch.cat([map_uncoverage_w, self.map_boundary.repeat([batch_size, 1, 1, 1])])
        return structured_map_info_r, struct_info_w

    def get_action(self, state, instruction, sample=False, rl_rollout=False):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        :param sample: (Only applies if self.rl): If true, sample action from action distribution. If False, take most likely action.
        #TODO: Absorb corpus within model
        :return:
        """
        device = next(self.parameters()).device
        self.eval()

        with torch.no_grad():

            states, images_fpv = self.states_to_torch(state)

            first_step = self.model_state.get("first_step", True)
            self.model_state.put("first_step", False)
            if first_step:
                reset_mask = [True]
                self.model_state.put("start_cam_poses", self.cam_poses_from_states(states).to(device))
            else:
                reset_mask = [False]

            self.model_state.put("seq_step", self.model_state.get("seq_step", 0) + 1)

            instr_len = [len(instruction)] if instruction is not None else None
            instructions = torch.LongTensor(instruction).unsqueeze(0)

            start_cam_poses = self.model_state.get("start_cam_poses")

            # Run stage1 visitation prediction
            # TODO: There's a bug here where we ignore images between planning timesteps. That's why must plan every timestep
            device = next(self.parameters()).device
            images_fpv = images_fpv.to(device)
            states = states.to(device)
            instructions = instructions.to(device)
            start_cam_poses = start_cam_poses.cuda(device)

            self.actprof.tick("start")
            #print("Planning for: " + debug_untokenize_instruction(list(instructions[0].detach().cpu().numpy())))
            log_v_dist_w, v_dist_w_poses, rl_outputs = self.stage1_visitation_prediction(
                images_fpv, states, instructions, instr_len,
                plan=[True], firstseg=[first_step],
                noisy_start_poses=start_cam_poses,
                start_poses=start_cam_poses,
                select_only=True,
                rl=True,
                noshow=True
            )
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
            # Log-distributions CANNOT be transformed - the transformer fills empty space with zeroes, which makes sense for
            # probability distributions, but makes no sense for likelihood scores
            x = v_dist_w.inner_distribution
            xr, r_poses = self.map_transformer_w_to_r(x, None, drn_poses)
            v_dist_r = Partial2DDistribution(xr, v_dist_w.outer_prob_mass)

            # Add map structured info (simulating laserscans)
            structured_map_info_r, map_info_w = self.structured_layers.build(map_uncoverage_w,
                                                                             drn_poses,
                                                                             self.map_transformer_w_to_r,
                                                                             use_map_boundary=self.s2_params[
                                                                                 "use_map_boundary"])

            if self.oracle_stage1:
                # Ground truth visitation distributions (in start and global frames)
                env_id, set_idx, seg_idx = self.model_state.get(["env_id", "set_idx", "seg_idx"])
                v_dist_w_gt = aup.resolve_and_get_ground_truth_static_global(env_id, set_idx, seg_idx, self.s1_params["global_map_size"], self.s1_params["world_size_px"]).to(images_fpv.device)
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
                Presenter().show_image(structured_map_info_r, "map_struct", scale=4, waitkey=1)
                v_dist_r.show("v_dist_r", scale=4, waitkey=1)

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
                value = value.detach()#.cpu().numpy()

            # Add an empty column for sideways velocity
            act = np.concatenate([xvel, np.zeros(xvel.shape), yawrate, stop])
            # This will be needed to compute rollout statistics later on

            # Keep all the info we will need later for A2C / PPO training
            # TODO: We assume independence between velocity and stop distributions. Not true, but what ya gonna do?
            rl_data = {
                "policy_input": v_dist_r.detach(),
                "policy_input_b": structured_map_info_r[0].detach(),
                "v_dist_w": v_dist_w.inner_distribution[0].detach(),
                "value_pred": value[0] if value else None,
                "xvel": xvel,
                "yawrate": yawrate,
                "stop": stop,
                "xvel_logprob": xvel_logprob,
                "yawrate_logprob": yawrate_logprob,
                "stop_logprob": stop_logprob,
                "action_logprob": xvel_logprob + stop_logprob + yawrate_logprob
            }
            self.actprof.tick("end")
            self.actprof.loop()
            self.actprof.print_stats(1)
            if rl_rollout:
                return act, rl_data
            else:
                viz_data = {}
                viz_data["v_dist_r_inner"] = v_dist_r.inner_distribution[0].detach().cpu().numpy()
                viz_data["v_dist_r_outer"] = v_dist_r.outer_prob_mass[0].detach().cpu().numpy()
                viz_data["v_dist_w_inner"] = v_dist_w.inner_distribution[0].detach().cpu().numpy()
                viz_data["v_dist_w_outer"] = v_dist_w.outer_prob_mass[0].detach().cpu().numpy()
                viz_data["map_struct"] = structured_map_info_r[0].detach().cpu().numpy()
                viz_data["BM_W"] = map_info_w[0].detach().cpu().numpy()
                return act, viz_data

    def unbatch(self, batch):
        # Inputs
        states = self.stage2_action_generation.cuda_var(batch["states"][0])
        seq_len = len(states)
        firstseg_mask = batch["firstseg_mask"][0]          # True for every timestep that is a new instruction segment
        plan_mask = batch["plan_mask"][0]                  # True for every timestep that we do visitation prediction
        actions = self.stage2_action_generation.cuda_var(batch["actions"][0])

        actions_select = self.stage2_action_generation.batch_select.one(actions, plan_mask, actions.device)

        # Ground truth visitation distributions (in start and global frames)
        v_dist_w_ground_truth_select = self.stage2_action_generation.cuda_var(batch["traj_ground_truth"][0])
        poses = self.poses_from_states(states)
        poses_select = self.stage2_action_generation.batch_select.one(poses, plan_mask, actions.device)
        v_dist_r_ground_truth_select, poses_r = self.map_transformer_w_to_r(v_dist_w_ground_truth_select, None, poses_select)

        #Presenter().show_image(v_dist_w_ground_truth_select.detach().cpu()[0,0], "v_dist_w_ground_truth_select", waitkey=1, scale=4)
        #Presenter().show_image(v_dist_r_ground_truth_select.detach().cpu()[0,0], "v_dist_r_ground_truth_select", waitkey=1, scale=4)

        return states, actions_select, v_dist_r_ground_truth_select, poses_select, plan_mask, firstseg_mask


    # Forward pass for training (with batch optimizations
    def sup_loss_on_batch(self, batch, eval, halfway=False):
        self.reset()
        states, actions_gt_select, v_dist_r_ground_truth_select, poses_select, plan_mask, firstseg_mask = self.unbatch(batch)
        images, states, instructions, instr_len, plan_mask, firstseg_mask, \
         start_poses, noisy_start_poses, metadata = self.stage1_visitation_prediction.unbatch(batch, halfway=halfway)
        batch_size = images.shape[0]

        # ----------------------------------------------------------------------------
        with torch.no_grad():
            map_uncoverage_w = self.stage1_visitation_prediction(images, states, instructions, instr_len,
                                                plan=plan_mask, firstseg=firstseg_mask,
                                                noisy_start_poses=start_poses,
                                                start_poses=start_poses,
                                                select_only=True,
                                                halfway="observability")

        # ----------------------------------------------------------------------------
        poses = self.poses_from_states(states)
        structured_map_info_r = self.build_map_structured_input(map_uncoverage_w, poses)
        map_uncoverage_r = structured_map_info_r[:, 0, :, :]

        v_dist_r_gt_masked = Partial2DDistribution.from_distribution_and_mask(v_dist_r_ground_truth_select, 1 - map_uncoverage_r)

        if False:
            for i in range(batch_size):
                v_dist_r_gt_masked.show("v_dist_r_masked", scale=4, waitkey=True, idx=i)

        xvel_dist, yawrate_dist, stop_dist, value_pred = self.stage2_action_generation(v_dist_r_gt_masked, structured_map_info_r, eval=False)
        xvel_logprob = xvel_dist.log_probs(actions_gt_select[:,0])
        yawrate_logprob = yawrate_dist.log_probs(actions_gt_select[:,2])
        # TODO: Figure out why this doesn't already sum
        stop_logprob = stop_dist.log_probs(actions_gt_select[:,3]).sum()
        total_logprob = xvel_logprob + yawrate_logprob + stop_logprob

        avg_logprob = total_logprob / batch_size
        avg_xvel_logprob = xvel_logprob / batch_size
        avg_yawrate_logprob = yawrate_logprob / batch_size
        avg_stop_logprob = stop_logprob / batch_size

        squared_xvel_dst = ((xvel_dist.mean - actions_gt_select[:,0]) ** 2).mean()
        squared_yawrate_dst = ((yawrate_dist.mean - actions_gt_select[:,2]) ** 2).mean()

        #action_loss = -avg_stop_logprob + squared_xvel_dst + squared_yawrate_dst
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

    def get_dataset(self, data=None, envs=None, domain=None, dataset_names=None, dataset_prefix=None, eval=False):
        return self.stage1_visitation_prediction.get_dataset(data=data, envs=envs, domain=domain, dataset_names=dataset_names, dataset_prefix=dataset_prefix, eval=eval)

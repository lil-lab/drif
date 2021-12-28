import torch
import torch.nn as nn
import torch.nn.functional as F

from data_io.paths import get_logging_dir

from learning.training.fixed_distributions import FixedBernoulli, FixedBeta, BoundedNormal, FixedCategorical, FixedNormal

from learning.inputs.pose import Pose
from learning.modules.map_to_map.map_batch_fill_missing import MapBatchFillMissing
from learning.modules.map_transformer import MapTransformer
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.map_to_map.map_batch_select import MapBatchSelect
from learning.modules.pvn.pvn_stage2_rlbase import PVN_Stage2_RLBase
from learning.modules.pvn.pvn_stage2_actionhead import PVN_Stage2_ActionHead
from learning.modules.pvn.pvn_stage2_valuehead import PVN_Stage2_ValueHead
from learning.inputs.partial_2d_distribution import Partial2DDistribution
from learning.models.navigation_model_component_base import NavigationModelComponentBase
from learning.modules.structured_map_layers import StructuredMapLayers

import learning.datasets.aux_data_providers as aup
from learning.datasets.segment_dataset_simple import SegmentDataset
from learning.modules.generic_model_state import GenericModelState

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.dummy_summary_writer import DummySummaryWriter

from learning.meters_and_metrics.meter_server import get_current_meters
import transformations

from parameters.parameter_server import get_current_parameters
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

#ACT_DIST = "categorical"
ACT_DIST = "normal"

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...


class PVN_Stage2_ActorCritic(NavigationModelComponentBase):

    def __init__(self, run_name="", model_instance_name="only"):

        super(PVN_Stage2_ActorCritic, self).__init__(run_name, domain=model_instance_name, model_name="pvn_stage2_ac")

        self.params_s1 = get_current_parameters()["ModelPVN"]["Stage1"]
        self.params_s2 = get_current_parameters()["ModelPVN"]["Stage2"]
        self.params = get_current_parameters()["ModelPVN"]["ActorCritic"]

        self.oob = self.params_s1.get("clip_observability")
        self.ignore_struct = self.params.get("ignore_structured_input", False)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        # Common
        # --------------------------------------------------------------------------------------------------------------
        # Wrap the Stage1 model so that it becomes invisible to PyTorch, get_parameters etc, and doesn't get optimized

        self.map_transform_w_to_r = MapTransformer(source_map_size=self.params_s1["global_map_size"],
                                                       dest_map_size=self.params_s1["local_map_size"],
                                                       world_size_px=self.params_s1["world_size_px"],
                                                       world_size_m=self.params_s1["world_size_m"])

        self.action_base = PVN_Stage2_RLBase(map_channels=self.params_s2["map_to_act_channels"],
                                             map_struct_channels=self.params_s2["map_structure_channels"],
                                             crop_size=self.params_s2["crop_size"],
                                             map_size=self.params_s1["local_map_size"],
                                             h1=self.params["h1"], h2=self.params["h2"],
                                             structure_h1=self.params["structure_h1"],
                                             obs_dim=self.params["obs_dim"],
                                             name="action")

        self.value_base = PVN_Stage2_RLBase(map_channels=self.params_s2["map_to_act_channels"],
                                            map_struct_channels=self.params_s2["map_structure_channels"],
                                            crop_size=self.params_s2["crop_size"],
                                            map_size=self.params_s1["local_map_size"],
                                            h1=self.params["h1"], h2=self.params["h2"],
                                            structure_h1=self.params["structure_h1"],
                                            obs_dim=self.params["obs_dim"],
                                            name="value")

        #self.structured_layers = StructuredMapLayers(self.s1_params["global_map_size"])

        self.action_head = PVN_Stage2_ActionHead(h2=self.params["h2"])
        self.value_head = PVN_Stage2_ValueHead(h2=self.params["h2"])

        self.spatialsoftmax = SpatialSoftmax2d()
        self.batch_select = MapBatchSelect()

        # PPO interface:
        self.is_recurrent = False

    def make_picklable(self):
        self.writer = DummySummaryWriter()

    def init_weights(self):
        self.action_base.init_weights()
        self.value_base.init_weights()
        self.action_head.init_weights()
        self.value_head.init_weights()

    def forward(self, v_dist_r, map_structure_r, eval=False, summarize=False):

        if isinstance(v_dist_r, list):
            inners = torch.cat([m.inner_distribution for m in v_dist_r], dim=0)
            outers = torch.cat([m.outer_prob_mass for m in v_dist_r], dim=0)
            v_dist_r = Partial2DDistribution(inners, outers)
            map_structure_r = torch.stack(map_structure_r, dim=0)

        if self.ignore_struct:
            map_structure_r = torch.zeros_like(map_structure_r)

        avec = self.action_base(v_dist_r, map_structure_r)
        vvec = self.value_base(v_dist_r, map_structure_r)
        action_scores = self.action_head(avec)
        value_pred = self.value_head(vvec)

        xvel_mean = action_scores[:, 0]
        xvel_std = F.softplus(action_scores[:, 2])
        xvel_dist = FixedNormal(xvel_mean, xvel_std)

        yawrate_mean = action_scores[:, 3]
        yawrate_std = F.softplus(action_scores[:, 5])
        yawrate_dist = FixedNormal(yawrate_mean, yawrate_std)

        # Skew it towards not stopping in the beginning
        stop_logits = action_scores[:, 6]
        stop_dist = FixedBernoulli(logits=stop_logits)

        return xvel_dist, yawrate_dist, stop_dist, value_pred

    def sample_action(self, xvel_dist, yawrate_dist, stop_dist):
        # Sample action from the predicted distributions
        xvel_sample = xvel_dist.sample()
        yawrate_sample = yawrate_dist.sample()
        stop = stop_dist.sample()
        return xvel_sample, yawrate_sample, stop

    def mode_action(self, xvel_dist, yawrate_dist, stop_dist):
        xvel_sample = xvel_dist.mode()
        yawrate_sample = yawrate_dist.mode()
        stop = stop_dist.mean
        return xvel_sample, yawrate_sample, stop

    def action_logprob(self, xvel_dist, yawrate_dist, stop_dist, xvel, yawrate, stop):
        xvel_logprob = xvel_dist.log_prob(xvel)
        yawrate_logprob = yawrate_dist.log_prob(yawrate)
        stop_logprob = stop_dist.log_prob(stop)
        return xvel_logprob, yawrate_logprob, stop_logprob

    def evaluate_actions(self, maps_r_batch, map_cov_r_batch, hidden_states_batch, masks_batch, actions_batch, global_step):
        xvel_dist, yawrate_dist, stop_dist, value_pred = self(maps_r_batch, map_cov_r_batch, summarize=True)

        # X-vel and yaw rate. Drop sideways velocity
        x_vel = actions_batch[:, 0]
        yawrate = actions_batch[:, 2]
        stops = actions_batch[:, 3]

        # Get action log-probabilities
        x_vel_log_probs, yawrate_log_probs, stop_log_probs = \
            self.action_logprob(xvel_dist, yawrate_dist, stop_dist, x_vel, yawrate, stops)

        x_vel_entropy = xvel_dist.entropy().mean()
        yawrate_entropy = yawrate_dist.entropy().mean()
        stop_entropy = stop_dist.entropy().mean()

        entropy = x_vel_entropy + yawrate_entropy + stop_entropy
        log_probs = x_vel_log_probs + yawrate_log_probs + stop_log_probs

        i = self.get_iter()
        if i % 17 == 0:
            stop_probs = stop_dist.probs.detach().cpu().mean()
            self.writer.add_scalar(f"{self.model_name}/stopprob", stop_probs.item(), i)

            if ACT_DIST == "normal":
                x_vel_mean = xvel_dist.mean.detach().cpu().mean(dim=0)
                x_vel_std = xvel_dist.stddev.detach().cpu().mean(dim=0)
                yawrate_mean = yawrate_dist.mean.detach().cpu().mean(dim=0)
                yawrate_std = yawrate_dist.stddev.detach().cpu().mean(dim=0)
                self.writer.add_scalar(f"{self.model_name}/x_vel_mean", x_vel_mean.item(), i)
                self.writer.add_scalar(f"{self.model_name}/x_vel_std", x_vel_std.item(), i)
                self.writer.add_scalar(f"{self.model_name}/yawrate_mean", yawrate_mean.item(), i)
                self.writer.add_scalar(f"{self.model_name}/yawrate_std", yawrate_std.item(), i)

            elif ACT_DIST == "beta":
                x_vel_alpha = xvel_dist.concentration0.detach().cpu().mean(dim=0)
                x_vel_beta = xvel_dist.concentration1.detach().cpu().mean(dim=0)
                yawrate_alpha = yawrate_dist.concentration0.detach().cpu().mean(dim=0)
                yawrate_beta = yawrate_dist.concentration1.detach().cpu().mean(dim=0)
                self.writer.add_scalar(f"{self.model_name}/vel_x_alpha", x_vel_alpha.item(), i)
                self.writer.add_scalar(f"{self.model_name}/vel_x_beta", x_vel_beta.item(), i)
                self.writer.add_scalar(f"{self.model_name}/yaw_rate_alpha", yawrate_alpha.item(), i)
                self.writer.add_scalar(f"{self.model_name}/yaw_rate_beta", yawrate_beta.item(), i)

        self.inc_iter()
        return value_pred, log_probs, entropy, None

    def unbatch(self, batch):
        # Inputs
        images = self.to_model_device(batch["images"][0])
        states = self.to_model_device(batch["states"][0])
        actions = self.to_model_device(batch["actions"][0])

        # Ground truth visitation distributions (in start and global frames)
        v_dist_w_ground_truth = self.to_model_device(batch["traj_ground_truth"][0])
        metadata = batch["md"][0][0]
        env_id = metadata["env_id"]

        self.model_state.tensor_store.keep_inputs("v_dist_w_ground_truth", v_dist_w_ground_truth)
        self.model_state.tensor_store.set_flag("env_id", env_id)

        return images, states, v_dist_w_ground_truth, actions, metadata

    # TODO: Delete this
    def sup_loss_on_batch(self, batch, eval):
        self.prof.tick("out")
        self.model_state = GenericModelState()

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.model_state.tensor_store

        images, states, v_dist_w_ground_truth, actions, metadata = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        traj_len = states.shape[0]
        reset_mask = [True] + [False] * (traj_len - 1)

        # TODO: Call the model and apply maximum likelihood

        # Transform to robot reference frame
        drn_poses = self.poses_from_states(states)
        x = v_dist_w_ground_truth.inner_distribution
        xr, r_poses = self.map_transformer_w_to_r(x, None, drn_poses)
        v_dist_r = Partial2DDistribution(xr, v_dist_w_ground_truth.outer_prob_mass)

        # Add map structured info (simulating laserscans)
        structured_map_info_r, map_info_w = self.structured_layers.build(map_uncoverage_w, drn_poses)

    def get_dataset(self, data=None, envs=None, domain=None, dataset_names=None, dataset_prefix=None, eval=False, halfway_only=False):
        # TODO: Maybe use eval here
        data_sources = []
        # If we're running auxiliary objectives, we need to include the data sources for the auxiliary labels
        data_sources.append(aup.PROVIDER_GOAL_POS)
        # Adding these in this order will compute poses with added noise and compute trajectory ground truth
        # in the reference frame of these noisy poses
        data_sources.append(aup.PROVIDER_START_POSES)
        data_sources.append(aup.PROVIDER_NOISY_POSES)
        data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)

        if not eval:
            index_by_env = get_current_parameters()["Data"].get("index_supervised_data_by_env", False)
        else:
            index_by_env = False

        return SegmentDataset(data=data,
                              env_list=envs,
                              domain=domain,
                              dataset_names=dataset_names,
                              dataset_prefix=dataset_prefix,
                              aux_provider_names=data_sources,
                              index_by_env=index_by_env,
                              segment_level=True)
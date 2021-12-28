import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io.paths import get_logging_dir
from torch.autograd import Variable

from learning.training.fixed_distributions import FixedBernoulli, FixedNormal
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.pose import Pose
from learning.datasets.aux_data_providers import get_top_down_ground_truth_static_global
from learning.modules.action_loss import ActionLoss
from learning.modules.map_to_action.cropped_map_to_action_triplet import CroppedMapToActionTriplet
from learning.modules.map_transformer import MapTransformer
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.map_to_map.map_batch_select import MapBatchSelect
from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.dummy_summary_writer import DummySummaryWriter

from learning.meters_and_metrics.meter_server import get_current_meters

from parameters.parameter_server import get_current_parameters
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

class PVN_Stage2_Bidomain(nn.Module):

    def __init__(self, run_name="", model_instance_name="only"):
        super(PVN_Stage2_Bidomain, self).__init__()
        self.model_name = "pvn_stage2"
        self.run_name = run_name
        self.instance_name = model_instance_name
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.instance_name}")

        self.params_s1 = get_current_parameters()["ModelPVN"]["Stage1"]
        self.params = get_current_parameters()["ModelPVN"]["Stage2"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.tensor_store = KeyTensorStore()

        # Common
        # --------------------------------------------------------------------------------------------------------------
        self.map_transform_w_to_r = MapTransformer(source_map_size=self.params_s1["global_map_size"],
                                                       dest_map_size=self.params_s1["local_map_size"],
                                                       world_size_px=self.params_s1["world_size_px"],
                                                       world_size_m=self.params_s1["world_size_m"])

        self.map_to_action = CroppedMapToActionTriplet(
            map_channels=self.params["map_to_act_channels"],
            map_size=self.params_s1["local_map_size"],
            manual=False,
            path_only=self.params["action_in_path_only"],
            recurrence=self.params["action_recurrence"])

        self.batch_select = MapBatchSelect()

        self.action_loss = ActionLoss()

        self.env_id = None
        self.seg_idx = None
        self.prev_instruction = None
        self.seq_step = 0
        self.get_act_start_pose = None
        self.gt_labels = None

    def steal_cross_domain_modules(self, other_self):
        self.map_to_action = other_self.map_to_action

    def both_domain_parameters(self, other_self):
        # This function iterates and yields parameters from this module and the other module, but does not yield
        # shared parameters twice.
        # Since all the parameters are shared, it's fine to just iterate over this module's parameters
        for p in self.parameters():
            yield p
        return

    def make_picklable(self):
        self.writer = DummySummaryWriter()

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.map_to_action.init_weights()

    def reset(self):
        self.tensor_store.reset()

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def start_rollout(self):
        import rollout.run_metadata as md
        m_size = self.params["local_map_size"]
        w_size = self.params["world_size_px"]
        self.gt_labels = get_top_down_ground_truth_static_global(
            md.ENV_ID, md.START_IDX, md.END_IDX, m_size, m_size, w_size, w_size)
        self.seg_idx = md.SEG_IDX
        self.gt_labels = self.maybe_cuda(self.gt_labels)
        if self.params["clear_history"]:
            self.start_sequence()

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def forward(self, visit_dist_r, structured_map_info_r, firstseg=None, eval=False):
        structured_map_info_r = None # not used in CoRL 2018 model

        action_scores = self.map_to_action(visit_dist_r, None, fistseg_mask=firstseg)

        self.prof.tick("map_to_action")
        xvel_mean = action_scores[:, 0]
        xvel_std = F.softplus(action_scores[:, 2])
        xvel_dist = FixedNormal(xvel_mean, xvel_std)

        yawrate_mean = action_scores[:, 3]
        if eval and self.params.get("test_time_amplifier"):
            yawrate_mean = yawrate_mean * self.params["test_time_amplifier"]

        yawrate_std = F.softplus(action_scores[:, 5])
        yawrate_dist = FixedNormal(yawrate_mean, yawrate_std)

        # Skew it towards not stopping in the beginning
        stop_logits = action_scores[:, 6]
        stop_dist = FixedBernoulli(logits=stop_logits)

        # TODO: This PVNv1 CoRL 2018 head is incompatible with Actor-critic  training for now
        value = None
        return xvel_dist, yawrate_dist, stop_dist, value

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

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch):
        # Inputs
        states = self.cuda_var(batch["states"][0])
        seq_len = len(states)
        firstseg_mask = batch["firstseg_mask"][0]          # True for every timestep that is a new instruction segment
        plan_mask = batch["plan_mask"][0]                  # True for every timestep that we do visitation prediction
        actions = self.cuda_var(batch["actions"][0])

        actions_select = self.batch_select.one(actions, plan_mask, actions.device)

        # Ground truth visitation distributions (in start and global frames)
        v_dist_w_ground_truth_select = self.cuda_var(batch["traj_ground_truth"][0])
        cam_poses = self.cam_poses_from_states(states)
        cam_poses_select = self.batch_select.one(cam_poses, plan_mask, actions.device)
        v_dist_r_ground_truth_select, poses_r = self.map_transform_w_to_r(v_dist_w_ground_truth_select, None, cam_poses_select)
        self.tensor_store.keep_inputs("v_dist_w_ground_truth_select", v_dist_w_ground_truth_select)
        self.tensor_store.keep_inputs("v_dist_r_ground_truth_select", v_dist_r_ground_truth_select)

        Presenter().show_image(v_dist_w_ground_truth_select.detach().cpu()[0,0], "v_dist_w_ground_truth_select", waitkey=1, scale=4)
        Presenter().show_image(v_dist_r_ground_truth_select.detach().cpu()[0,0], "v_dist_r_ground_truth_select", waitkey=1, scale=4)

        return states, actions_select, v_dist_r_ground_truth_select, cam_poses_select, plan_mask, firstseg_mask

    # TODO: This code heavily overlaps with that in ModelPvnWrapperBidomain.
    # PVN Wrapper should be used for training Stage 2 and not this
    # Forward pass for training Stage 2 only (with batch optimizations)
    def sup_loss_on_batch(self, batch, eval, halfway=False):
        raise ValueError("This code still works but is deprecated. Train Stage2 using PVNWrapper instead (it can compute observability of maps etc)")
        self.prof.tick("out")
        action_loss_total = self.cuda_var(torch.zeros([1]))
        if batch is None:
            print("Skipping None Batch")
            return action_loss_total

        self.reset()
        states, actions_gt_select, v_dist_r_ground_truth_select, cam_poses_select, plan_mask, firstseg_mask = self.unbatch(batch)
        count = 0
        self.prof.tick("inputs")

        batch_size = actions_gt_select.shape[0]

        # ----------------------------------------------------------------------------
        xvel_dist, yawrate_dist, stop_dist, _ = self(v_dist_r_ground_truth_select, firstseg_mask)

        stop_logprob = stop_dist.log_probs(actions_gt_select[:,3]).sum()
        avg_stop_logprob = stop_logprob / batch_size
        squared_xvel_dst = ((xvel_dist.mean - actions_gt_select[:,0]) ** 2).sum()
        squared_yawrate_dst = ((yawrate_dist.mean - actions_gt_select[:,2]) ** 2).sum()
        action_loss = -stop_logprob + squared_xvel_dst + squared_yawrate_dst

        self.prof.tick("loss")

        prefix = self.model_name + ("/eval" if eval else "/train")
        self.writer.add_scalar(prefix + "/action_loss", action_loss.data.cpu().item(), self.get_iter())
        self.writer.add_scalar(prefix + "/x_sqrdst", squared_xvel_dst.data.cpu().item(), self.get_iter())
        self.writer.add_scalar(prefix + "/yaw_sqrdst", (squared_yawrate_dst / batch_size).data.cpu().item(), self.get_iter())
        self.writer.add_scalar(prefix + "/stop_logprob", (avg_stop_logprob / batch_size).data.cpu().item(), self.get_iter())
        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return action_loss, self.tensor_store

    def get_dataset(self, data=None, envs=None, domain=None, dataset_names=None, dataset_prefix=None, eval=False):
        # TODO: Maybe use eval here
        data_sources = []
        data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
        return SegmentDataset(data=data, env_list=envs, domain=domain, dataset_names=dataset_names, dataset_prefix=dataset_prefix, aux_provider_names=data_sources, segment_level=True)

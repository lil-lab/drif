import torch
import torch.nn as nn
import numpy as np

from data_io.paths import get_logging_dir
from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose
from learning.datasets.aux_data_providers import get_top_down_ground_truth_static_global
from learning.modules.action_loss import ActionLoss
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.keyboard import KeyTeleop

from parameters.parameter_server import get_current_parameters
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...


class PVN_Stage2_Keyboard(nn.Module):

    def __init__(self, run_name="", model_instance_name="only"):

        super(PVN_Stage2_Keyboard, self).__init__()
        self.model_name = "pvn_stage2"
        self.run_name = run_name
        self.instance_name = model_instance_name
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.instance_name}")

        self.params_s1 = get_current_parameters()["ModelPVN"]["Stage1"]
        self.params = get_current_parameters()["ModelPVN"]["Stage2"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.spatialsoftmax = SpatialSoftmax2d()
        self.action_loss = ActionLoss()

        self.teleoper = KeyTeleop()

        self.env_id = None
        self.seg_idx = None
        self.prev_instruction = None
        self.seq_step = 0
        self.get_act_start_pose = None
        self.gt_labels = None

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        pass

    def reset(self):
        self.teleoper.reset()

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def start_segment_rollout(self):
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

    def forward(self, visit_dist_r, map_uncoverage, firstseg=None, eval=False):
        action = self.teleoper.get_command()

        inner_goal_dist = visit_dist_r.inner_distribution

        prob_goal_inside = inner_goal_dist[0, 1].sum().detach().item()
        rectangle = np.zeros([100, 20, 3])
        fill_until = int(100 * prob_goal_inside)
        rectangle[fill_until:, :, 0] = 1.0
        Presenter().show_image(rectangle, "P(outside)", scale=4, waitkey=1)

        # Normalize channels for viewing
        inner_goal_dist[0, 0] /= (inner_goal_dist[0, 0].max() + 1e-10)
        inner_goal_dist[0, 1] /= (inner_goal_dist[0, 1].max() + 1e-10)

        Presenter().show_image(inner_goal_dist[0].detach(), "visit_dist", scale=8, waitkey=1)
        Presenter().show_image(map_uncoverage[0].detach(), "unobserved", scale=8, waitkey=1)

        action_t = torch.Tensor(action)
        return action_t

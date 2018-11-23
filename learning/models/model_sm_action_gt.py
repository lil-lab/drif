import torch
import torch.nn as nn
from torch.autograd import Variable

from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.weights import enable_weight_saving
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup
from learning.datasets.masking import get_obs_mask_every_n_and_segstart, get_obs_mask_segstart
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose
from learning.inputs.sequence import len_until_nones
from learning.datasets.aux_data_providers import get_top_down_ground_truth_static_global
from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.action_loss import ActionLoss
from learning.modules.map_to_action.ego_map_to_action_triplet import EgoMapToActionTriplet
from learning.modules.map_to_action.cropped_map_to_action_triplet import CroppedMapToActionTriplet
from learning.modules.map_to_map.map_batch_fill_missing import MapBatchFillMissing
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.meters.meter_server import get_current_meters

from parameters.parameter_server import get_current_parameters

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...


class ModelTrajectoryToAction(ModuleWithAuxiliaries):

    def __init__(self, run_name=""):

        super(ModelTrajectoryToAction, self).__init__()
        self.model_name = "lsvd_action"
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir="runs/" + run_name)

        self.params = get_current_parameters()["ModelPVN"]
        self.aux_weights = get_current_parameters()["AuxWeights"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Common
        # --------------------------------------------------------------------------------------------------------------
        self.map_transform_w_to_s = MapTransformerBase(source_map_size=self.params["global_map_size"],
                                                       dest_map_size=self.params["local_map_size"],
                                                       world_size=self.params["world_size_px"])

        self.map_transform_r_to_w = MapTransformerBase(source_map_size=self.params["local_map_size"],
                                                       dest_map_size=self.params["global_map_size"],
                                                       world_size=self.params["world_size_px"])

        # Output an action given the global semantic map
        if self.params["map_to_action"] == "downsample2":
            self.map_to_action = EgoMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"],
                other_features_size=self.params["emb_size"])

        elif self.params["map_to_action"] == "cropped":
            self.map_to_action = CroppedMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"],
                manual=self.params["manual_rule"],
                path_only=self.params["action_in_path_only"],
                recurrence=self.params["action_recurrence"])

        self.spatialsoftmax = SpatialSoftmax2d()
        self.gt_fill_missing = MapBatchFillMissing(self.params["local_map_size"], self.params["world_size_px"])

        # Don't freeze the trajectory to action weights, because it will be pre-trained during path-prediction training
        # and finetuned on all timesteps end-to-end
        enable_weight_saving(self.map_to_action, "map_to_action", alwaysfreeze=False, neverfreeze=True)

        self.action_loss = ActionLoss()

        self.env_id = None
        self.seg_idx = None
        self.prev_instruction = None
        self.seq_step = 0
        self.get_act_start_pose = None
        self.gt_labels = None

    # TODO: Try to hide these in a superclass or something. They take up a lot of space:
    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.map_to_action.cuda(device)
        self.action_loss.cuda(device)
        self.map_transform_w_to_s.cuda(device)
        self.map_transform_r_to_w.cuda(device)
        self.gt_fill_missing.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.map_to_action.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        super(ModelTrajectoryToAction, self).reset()
        self.map_transform_w_to_s.reset()
        self.map_transform_r_to_w.reset()
        self.gt_fill_missing.reset()

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

    def get_action(self, state, instruction):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        #TODO: Absorb corpus within model
        :return:
        """
        prof = SimpleProfiler(print=True)
        prof.tick(".")
        # TODO: Simplify this
        self.eval()
        images_np_pure = state.image
        state_np = state.state
        state = Variable(none_padded_seq_to_tensor([state_np]))

        #print("Act: " + debug_untokenize_instruction(instruction))

        # Add the batch dimension

        first_step = True
        if instruction == self.prev_instruction:
            first_step = False
        self.prev_instruction = instruction
        if first_step:
            self.get_act_start_pose = self.cam_poses_from_states(state[0:1])

        self.seq_step += 1

        # This is for training the policy to mimic the ground-truth state distribution with oracle actions
        # b_traj_gt_w_select = b_traj_ground_truth[b_plan_mask_t[:, np.newaxis, np.newaxis, np.newaxis].expand_as(b_traj_ground_truth)].view([-1] + gtsize)
        traj_gt_w = Variable(self.gt_labels)
        b_poses = self.cam_poses_from_states(state)
        # TODO: These source and dest should go as arguments to get_maps (in forward pass not params)
        transformer = MapTransformerBase(
            source_map_size=self.params["global_map_size"],
            world_size=self.params["world_size_px"],
            dest_map_size=self.params["local_map_size"])
        self.maybe_cuda(transformer)
        transformer.set_maps(traj_gt_w, None)
        traj_gt_r, _ = transformer.get_maps(b_poses)
        self.clear_inputs("traj_gt_r_select")
        self.clear_inputs("traj_gt_w_select")
        self.keep_inputs("traj_gt_r_select", traj_gt_r)
        self.keep_inputs("traj_gt_w_select", traj_gt_w)

        action = self(traj_gt_r, firstseg=[self.seq_step == 1])

        output_action = action.squeeze().data.cpu().numpy()

        stop_prob = output_action[3]
        output_stop = 1 if stop_prob > self.params["stop_threshold"] else 0
        output_action[3] = output_stop

        return output_action

    def deterministic_action(self, action_mean, action_std, stop_prob):
        batch_size = action_mean.size(0)
        action = Variable(empty_float_tensor((batch_size, 4), self.is_cuda, self.cuda_device))
        action[:, 0:3] = action_mean[:, 0:3]
        action[:, 3] = stop_prob
        return action

    def sample_action(self, action_mean, action_std, stop_prob):
        action = torch.normal(action_mean, action_std)
        stop = torch.bernoulli(stop_prob)
        return action, stop

    # This is called before beginning an execution sequence
    def start_sequence(self):
        self.seq_step = 0
        self.reset()
        print("RESETTED!")
        return

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def save(self, epoch):
        filename = self.params["map_to_action_file"] + "_" + self.run_name + "_" + str(epoch)
        save_pytorch_model(self.map_to_action, filename)
        print("Saved action model to " + filename)

    def forward(self, traj_gt_r, firstseg=None):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param instructions: BxM LongTensor where M is the maximum length of any instruction
        :param instr_lengths: list of len B of integers, indicating length of each instruction
        :param has_obs: list of booleans of length B indicating whether the given element in the sequence has an observation
        :param yield_semantic_maps: If true, will not compute actions (full model), but return the semantic maps that
            were built along the way in response to the images. This is ugly, but allows code reuse
        :return:
        """
        action_pred = self.map_to_action(traj_gt_r, None, fistseg_mask=firstseg)
        out_action = self.deterministic_action(action_pred[:, 0:3], None, action_pred[:, 3])
        self.keep_inputs("action", out_action)
        self.prof.tick("map_to_action")

        return out_action

    def maybe_cuda(self, tensor):
        if self.is_cuda:
            if False:
                if type(tensor) is Variable:
                    tensor.data.pin_memory()
                elif type(tensor) is Pose:
                    pass
                elif type(tensor) is torch.FloatTensor:
                    tensor.pin_memory()
            return tensor.cuda()
        else:
            return tensor

    def cuda_var(self, tensor):
        return cuda_var(tensor, self.is_cuda, self.cuda_device)

    # Forward pass for training (with batch optimizations
    def sup_loss_on_batch(self, batch, eval):
        self.prof.tick("out")

        action_loss_total = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))

        if batch is None:
            print("Skipping None Batch")
            return action_loss_total

        actions = self.maybe_cuda(batch["actions"])
        states = self.maybe_cuda(batch["states"])

        firstseg_mask = batch["firstseg_mask"]

        # Auxiliary labels
        traj_ground_truth_select = self.maybe_cuda(batch["traj_ground_truth"])
        # stops = self.maybe_cuda(batch["stops"])
        metadata = batch["md"]
        batch_size = actions.size(0)
        count = 0

        # Loop thru batch
        for b in range(batch_size):
            seg_idx = -1

            self.reset()

            self.prof.tick("out")
            b_seq_len = len_until_nones(metadata[b])

            # TODO: Generalize this
            # Slice the data according to the sequence length
            b_metadata = metadata[b][:b_seq_len]
            b_actions = actions[b][:b_seq_len]
            b_traj_ground_truth_select = traj_ground_truth_select[b]
            b_states = states[b][:b_seq_len]

            self.keep_inputs("traj_gt_global_select", b_traj_ground_truth_select)

            #b_firstseg = get_obs_mask_segstart(b_metadata)
            b_firstseg = firstseg_mask[b][:b_seq_len]

            # ----------------------------------------------------------------------------
            # Optional Auxiliary Inputs
            # ----------------------------------------------------------------------------
            gtsize = list(b_traj_ground_truth_select.size())[1:]
            b_poses = self.cam_poses_from_states(b_states)
            # TODO: These source and dest should go as arguments to get_maps (in forward pass not params)
            transformer = MapTransformerBase(
                source_map_size=self.params["global_map_size"],
                world_size=self.params["world_size_px"],
                dest_map_size=self.params["local_map_size"])
            self.maybe_cuda(transformer)
            transformer.set_maps(b_traj_ground_truth_select, None)
            traj_gt_local_select, _ = transformer.get_maps(b_poses)
            self.keep_inputs("traj_gt_r_select", traj_gt_local_select)
            self.keep_inputs("traj_gt_w_select", b_traj_ground_truth_select)

            # ----------------------------------------------------------------------------

            self.prof.tick("inputs")

            actions = self(traj_gt_local_select, firstseg=b_firstseg)
            action_losses, _ = self.action_loss(b_actions, actions, batchreduce=False)
            action_losses = self.action_loss.batch_reduce_loss(action_losses)
            action_loss = self.action_loss.reduce_loss(action_losses)
            action_loss_total = action_loss
            count += b_seq_len

            self.prof.tick("loss")

        action_loss_avg = action_loss_total / (count + 1e-9)
        prefix = self.model_name + ("/eval" if eval else "/train")
        self.writer.add_scalar(prefix + "/action_loss", action_loss_avg.data.cpu()[0], self.get_iter())

        self.prof.tick("out")

        prefix = self.model_name + ("/eval" if eval else "/train")
        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return action_loss_avg

    def get_dataset(self, data=None, envs=None, dataset_name=None, eval=False):
        # TODO: Maybe use eval here
        data_sources = []
        data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
        return SegmentDataset(data=data, env_list=envs, dataset_name=dataset_name, aux_provider_names=data_sources, segment_level=True)
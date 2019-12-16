import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup


from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones
from learning.inputs.vision import standardize_image
#from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from learning.modules.auxiliaries.class_auxiliary import ClassAuxiliary
from learning.modules.gs_fpv.recurrent_embedding import RecurrentEmbedding
from learning.modules.action_loss import ActionLoss
from learning.modules.img_to_map.fpv_to_fpv import FPVToFPVMap
from learning.modules.blocks import DenseMlpBlock2
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter
from learning.modules.rss.map_lang_spatial_filter import MapLangSpatialFilter
from learning.modules.downsample_map.downsample_res import DownsampleResidual
from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.meters_and_metrics.meter_server import get_current_meters

from parameters.parameter_server import get_current_parameters

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False
RESNET_FACTOR = 4

MODEL_RSS = "rss"
MODEL_GS_FPV_MEM = "gs_fpv_mem"
MODEL_GS_FPV = "gs_fpv"

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...


class ModelGSFPV(nn.Module):

    def __init__(self, run_name="",
                 aux_class_features=False, aux_grounding_features=False, aux_lang=False, recurrence=False):

        super(ModelGSFPV, self).__init__()
        self.model_name = "gs_fpv" + "_mem" if recurrence else ""
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir="runs/" + run_name)

        self.params = get_current_parameters()["Model"]
        self.aux_weights = get_current_parameters()["AuxWeights"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Auxiliary Objectives
        self.use_aux_class_features = aux_class_features
        self.use_aux_grounding_features = aux_grounding_features
        self.use_aux_lang = aux_lang
        self.use_recurrence = recurrence

        self.img_to_features_w = FPVToFPVMap(self.params["img_w"], self.params["img_h"],
                                             self.params["resnet_channels"], self.params["feature_channels"])

        self.lang_filter_gnd = MapLangSemanticFilter(self.params["emb_size"], self.params["feature_channels"], self.params["relevance_channels"])

        self.lang_filter_goal = MapLangSpatialFilter(self.params["emb_size"], self.params["relevance_channels"], self.params["goal_channels"])

        self.map_downsample = DownsampleResidual(self.params["map_to_act_channels"], 2)

        self.recurrence = RecurrentEmbedding(self.params["gs_fpv_feature_map_size"], self.params["gs_fpv_recurrence_size"])

        # Sentence Embedding
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"])

        in_features_size = self.params["gs_fpv_feature_map_size"] + self.params["emb_size"]
        if self.use_recurrence:
            in_features_size += self.params["gs_fpv_recurrence_size"]

        self.features_to_action = DenseMlpBlock2(in_features_size, self.params["mlp_hidden"], 4)

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        self.add_auxiliary(ClassAuxiliary2D("aux_class", None,  self.params["feature_channels"], self.params["num_landmarks"],
                                                "fpv_features", "lm_pos_fpv", "lm_indices"))
        self.add_auxiliary(ClassAuxiliary2D("aux_ground", None, self.params["relevance_channels"], 2,
                                                "fpv_features_g", "lm_pos_fpv", "lm_mentioned"))
        if self.params["templates"]:
            self.add_auxiliary(ClassAuxiliary("aux_lang_lm", self.params["emb_size"], self.params["num_landmarks"], 1,
                                              "sentence_embed", "lm_mentioned_tplt"))
            self.add_auxiliary(ClassAuxiliary("aux_lang_side", self.params["emb_size"], self.params["num_sides"], 1,
                                              "sentence_embed", "side_mentioned_tplt"))
        else:
            self.add_auxiliary(ClassAuxiliary("aux_lang_lm_nl", self.params["emb_size"], 2, self.params["num_landmarks"],
                                                "sentence_embed", "lang_lm_mentioned"))

        self.action_loss = ActionLoss()

        self.env_id = None
        self.prev_instruction = None
        self.seq_step = 0

    # TODO: Try to hide these in a superclass or something. They take up a lot of space:
    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.sentence_embedding.cuda(device)
        self.img_to_features_w.cuda(device)
        self.lang_filter_gnd.cuda(device)
        self.lang_filter_goal.cuda(device)
        self.action_loss.cuda(device)
        self.recurrence.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.img_to_features_w.init_weights()
        self.lang_filter_gnd.init_weights()
        self.lang_filter_goal.init_weights()
        self.sentence_embedding.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        super(ModelGSFPV, self).reset()
        self.sentence_embedding.reset()
        self.img_to_features_w.reset()
        self.recurrence.reset()
        self.prev_instruction = None
        print("GS_FPV_MEM_RESET")

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def start_segment_rollout(self, *args):
        self.reset()

    def get_action(self, state, instruction):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        #TODO: Absorb corpus within model
        :return:
        """
        # TODO: Simplify this
        self.eval()
        images_np_pure = state.image
        state_np = state.state

        #print("Act: " + debug_untokenize_instruction(instruction))

        images_np = standardize_image(images_np_pure)
        image_fpv = Variable(none_padded_seq_to_tensor([images_np]))
        state = Variable(none_padded_seq_to_tensor([state_np]))
        self.prev_instruction = instruction

        img_in_t = image_fpv
        img_in_t.volatile = True

        instr_len = [len(instruction)] if instruction is not None else None
        instruction = torch.LongTensor(instruction).unsqueeze(0)
        instruction = cuda_var(instruction, self.is_cuda, self.cuda_device)

        state.volatile = True

        if self.is_cuda:
            img_in_t = img_in_t.cuda(self.cuda_device)
            state = state.cuda(self.cuda_device)

        self.seq_step += 1

        action = self(img_in_t, state, instruction, instr_len)

        output_action = action.squeeze().data.cpu().numpy()
        print("action: ", output_action)

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

    # TODO: Move this somewhere and standardize
    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def forward(self, images, states, instructions, instr_lengths):
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
        cam_poses = self.cam_poses_from_states(states)
        self.prof.tick("out")

        #print("Trn: " + debug_untokenize_instruction(instructions[0].data[:instr_lengths[0]]))

        # Calculate the instruction embedding
        if instructions is not None:
            # TODO: Take batch of instructions and their lengths, return batch of embeddings. Store the last one as internal state
            sent_embeddings = self.sentence_embedding(instructions, instr_lengths)
            self.keep_inputs("sentence_embed", sent_embeddings)
        else:
            sent_embeddings = self.sentence_embedding.get()

        self.prof.tick("embed")

        seq_size = len(images)

        # Extract and project features onto the egocentric frame for each image
        fpv_features = self.img_to_features_w(images, cam_poses, sent_embeddings, self, show="")

        self.keep_inputs("fpv_features", fpv_features)
        self.prof.tick("img_to_map_frame")

        self.lang_filter_gnd.precompute_conv_weights(sent_embeddings)
        self.lang_filter_goal.precompute_conv_weights(sent_embeddings)

        gnd_features = self.lang_filter_gnd(fpv_features)
        goal_features = self.lang_filter_goal(gnd_features)

        self.keep_inputs("fpv_features_g", gnd_features)
        visual_features = torch.cat([gnd_features, goal_features], dim=1)

        lstm_in_features = visual_features.view([seq_size, 1, -1])

        catlist = [lstm_in_features.view([seq_size, -1]), sent_embeddings]

        if self.use_recurrence:
            memory_features = self.recurrence(lstm_in_features)
            catlist.append(memory_features[:, 0, :])

        action_features = torch.cat(catlist, dim=1)

        # Output the final action given the processed map
        action_pred = self.features_to_action(action_features)
        action_pred[:, 3] = torch.sigmoid(action_pred[:, 3])
        out_action = self.deterministic_action(action_pred[:, 0:3], None, action_pred[:, 3])
        self.prof.tick("map_to_action")

        return out_action

    def maybe_cuda(self, tensor):
        if self.is_cuda:
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

        images = self.maybe_cuda(batch["images"])

        instructions = self.maybe_cuda(batch["instr"])
        instr_lengths = batch["instr_len"]
        states = self.maybe_cuda(batch["states"])
        actions = self.maybe_cuda(batch["actions"])

        # Auxiliary labels
        lm_pos_fpv = batch["lm_pos_fpv"]
        lm_indices = batch["lm_indices"]
        lm_mentioned = batch["lm_mentioned"]
        lang_lm_mentioned = batch["lang_lm_mentioned"]

        templates = get_current_parameters()["Environment"]["Templates"]
        if templates:
            lm_mentioned_tplt = batch["lm_mentioned_tplt"]
            side_mentioned_tplt = batch["side_mentioned_tplt"]

        # stops = self.maybe_cuda(batch["stops"])
        masks = self.maybe_cuda(batch["masks"])
        metadata = batch["md"]

        seq_len = images.size(1)
        batch_size = images.size(0)
        count = 0
        correct_goal_count = 0
        goal_count = 0

        # Loop thru batch
        for b in range(batch_size):
            seg_idx = -1

            self.reset()

            self.prof.tick("out")
            b_seq_len = len_until_nones(metadata[b])

            # TODO: Generalize this
            # Slice the data according to the sequence length
            b_metadata = metadata[b][:b_seq_len]
            b_images = images[b][:b_seq_len]
            b_instructions = instructions[b][:b_seq_len]
            b_instr_len = instr_lengths[b][:b_seq_len]
            b_states = states[b][:b_seq_len]
            b_actions = actions[b][:b_seq_len]
            b_lm_pos_fpv = lm_pos_fpv[b][:b_seq_len]
            b_lm_indices = lm_indices[b][:b_seq_len]
            b_lm_mentioned = lm_mentioned[b][:b_seq_len]

            b_lm_pos_fpv = [self.cuda_var((s / RESNET_FACTOR).long()) if s is not None else None for s in b_lm_pos_fpv]
            b_lm_indices = [self.cuda_var(s) if s is not None else None for s in b_lm_indices]
            b_lm_mentioned = [self.cuda_var(s) if s is not None else None for s in b_lm_mentioned]

            # TODO: Figure out how to keep these properly. Perhaps as a whole batch is best
            # TODO: Introduce a key-value store (encapsulate instead of inherit)
            self.keep_inputs("lm_pos_fpv", b_lm_pos_fpv)
            self.keep_inputs("lm_indices", b_lm_indices)
            self.keep_inputs("lm_mentioned", b_lm_mentioned)

            # TODO: Abstract all of these if-elses in a modular way once we know which ones are necessary
            if templates:
                b_lm_mentioned_tplt = lm_mentioned_tplt[b][:b_seq_len]
                b_side_mentioned_tplt = side_mentioned_tplt[b][:b_seq_len]
                b_side_mentioned_tplt = self.cuda_var(b_side_mentioned_tplt)
                b_lm_mentioned_tplt = self.cuda_var(b_lm_mentioned_tplt)
                self.keep_inputs("lm_mentioned_tplt", b_lm_mentioned_tplt)
                self.keep_inputs("side_mentioned_tplt", b_side_mentioned_tplt)
            else:
                b_lang_lm_mentioned = self.cuda_var(lang_lm_mentioned[b][:b_seq_len])
                self.keep_inputs("lang_lm_mentioned", b_lang_lm_mentioned)


            # ----------------------------------------------------------------------------

            self.prof.tick("inputs")

            actions = self(b_images, b_states, b_instructions, b_instr_len)

            action_losses, _ = self.action_loss(b_actions, actions, batchreduce=False)

            self.prof.tick("call")
            action_losses = self.action_loss.batch_reduce_loss(action_losses)
            action_loss = self.action_loss.reduce_loss(action_losses)
            action_loss_total = action_loss
            count += b_seq_len

            self.prof.tick("loss")

        action_loss_avg = action_loss_total / (count + 1e-9)

        self.prof.tick("out")

        # Doing this in the end (outside of se
        aux_losses = self.calculate_aux_loss(reduce_average=True)
        aux_loss = self.combine_aux_losses(aux_losses, self.aux_weights)

        prefix = self.model_name + ("/eval" if eval else "/train")

        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())
        self.writer.add_dict(prefix, aux_losses, self.get_iter())
        self.writer.add_scalar(prefix + "/action_loss", action_loss_avg.data.cpu()[0], self.get_iter())

        self.prof.tick("auxiliaries")

        total_loss = action_loss_avg + aux_loss

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return total_loss

    def get_dataset(self, data=None, envs=None, dataset_names=None, dataset_prefix=None, eval=False):
        # TODO: Maybe use eval here
        #if self.fpv:
        data_sources = []
        data_sources.append(aup.PROVIDER_LM_POS_DATA)
        data_sources.append(aup.PROVIDER_LANDMARKS_MENTIONED)

        templates = get_current_parameters()["Environment"]["Templates"]
        if templates:
            data_sources.append(aup.PROVIDER_LANG_TEMPLATE)

        return SegmentDataset(data=data, env_list=envs, dataset_names=dataset_names, dataset_prefix=dataset_prefix, aux_provider_names=data_sources, segment_level=True)
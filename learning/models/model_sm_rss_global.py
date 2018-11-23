import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.misc import imsave

from data_io.weights import enable_weight_saving
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose, get_noisy_poses_torch
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones
from learning.inputs.vision import standardize_image
from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from learning.modules.auxiliaries.goal_auxiliary import GoalAuxiliary2D
from learning.modules.auxiliaries.path_auxiliary import PathAuxiliary2D
from learning.modules.auxiliaries.class_auxiliary import ClassAuxiliary
from learning.modules.auxiliaries.feature_reg_auxiliary import FeatureRegularizationAuxiliary2D
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.modules.action_loss import ActionLoss
from learning.modules.img_to_map.fpv_to_global_map import FPVToGlobalMap
from learning.modules.map_to_action.ego_map_to_action_triplet import EgoMapToActionTriplet
from learning.modules.map_to_action.cropped_map_to_action_triplet import CroppedMapToActionTriplet
from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegratorGlobalMap
from learning.modules.map_to_map.identity_map_to_map import IdentityMapProcessor
from learning.modules.map_to_map.lang_filter_map_to_map import LangFilterMapProcessor
from learning.modules.map_to_map.map_batch_fill_missing import MapBatchFillMissing
from learning.modules.map_to_map.map_batch_select import MapBatchSelect
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.map_transformer_base import MapTransformerBase
from utils.simple_profiler import SimpleProfiler
from visualization import Presenter
from utils.logging_summary_writer import LoggingSummaryWriter
from tensorboardX import SummaryWriter

from learning.meters.moving_average import MovingAverageMeter
from learning.meters.meter_server import get_current_meters

from learning.utils import save_tensor_as_img, get_viz_dir

from parameters.parameter_server import get_current_parameters
from rollout import run_metadata as run_metadata

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False
RESNET_FACTOR = 4

MODEL_RSS = "rss"

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...


class ModelTrajectoryTopDown(ModuleWithAuxiliaries):

    def __init__(self, run_name="", model_class=MODEL_RSS,
                 aux_class_features=False, aux_grounding_features=False,
                 aux_class_map=False, aux_grounding_map=False, aux_goal_map=False,
                 aux_lang=False, aux_traj=False, rot_noise=False, pos_noise=False):

        super(ModelTrajectoryTopDown, self).__init__()
        self.model_name = "sm_trajectory" + str(model_class)
        self.model_class = model_class
        print("Init model of type: ", str(model_class))
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir="runs/" + run_name)

        self.params = get_current_parameters()["Model"]
        self.aux_weights = get_current_parameters()["AuxWeights"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Auxiliary Objectives
        self.use_aux_class_features = aux_class_features
        self.use_aux_grounding_features = aux_grounding_features
        self.use_aux_class_on_map = aux_class_map
        self.use_aux_grounding_on_map = aux_grounding_map
        self.use_aux_goal_on_map = aux_goal_map
        self.use_aux_lang = aux_lang
        self.use_aux_traj_on_map = aux_traj
        self.use_aux_reg_map = self.aux_weights["regularize_map"]

        self.use_rot_noise = rot_noise
        self.use_pos_noise = pos_noise


        # Path-pred FPV model definition
        # --------------------------------------------------------------------------------------------------------------

        self.img_to_features_w = FPVToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"], world_size=self.params["world_size_m"],
            res_channels=self.params["resnet_channels"], map_channels=self.params["feature_channels"],
            img_w=self.params["img_w"], img_h=self.params["img_h"], img_dbg=IMG_DBG)

        self.map_accumulator_w = LeakyIntegratorGlobalMap(source_map_size=self.params["global_map_size"], world_in_map_size=self.params["world_size_px"])

        # Pre-process the accumulated map to do language grounding if necessary - in the world reference frame
        if self.use_aux_grounding_on_map and not self.use_aux_grounding_features:
            self.map_processor_a_w = LangFilterMapProcessor(
                source_map_size=self.params["global_map_size"],
                world_size=self.params["world_size_px"],
                embed_size=self.params["emb_size"],
                in_channels=self.params["feature_channels"],
                out_channels=self.params["relevance_channels"],
                spatial=False, cat_out=True)
        else:
            self.map_processor_a_w = IdentityMapProcessor(source_map_size=self.params["global_map_size"], world_size=self.params["world_size_px"])

        if self.use_aux_goal_on_map:
            self.map_processor_b_r = LangFilterMapProcessor(source_map_size=self.params["local_map_size"],
                                                            world_size=self.params["world_size_px"],
                                                            embed_size=self.params["emb_size"],
                                                            in_channels=self.params["relevance_channels"],
                                                            out_channels=self.params["goal_channels"],
                                                            spatial=True, cat_out=True)
        else:
            self.map_processor_b_r = IdentityMapProcessor(source_map_size=self.params["local_map_size"],
                                                          world_size=self.params["world_size_px"])

        pred_channels = self.params["goal_channels"] + self.params["relevance_channels"]

        # Common
        # --------------------------------------------------------------------------------------------------------------

        # Sentence Embedding
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"])

        self.map_transform_w_to_r = MapTransformerBase(source_map_size=self.params["global_map_size"],
                                                       dest_map_size=self.params["local_map_size"],
                                                       world_size=self.params["world_size_px"])
        self.map_transform_r_to_w = MapTransformerBase(source_map_size=self.params["local_map_size"],
                                                       dest_map_size=self.params["global_map_size"],
                                                       world_size=self.params["world_size_px"])

        # Batch select is used to drop and forget semantic maps at those timestaps that we're not planning in
        self.batch_select = MapBatchSelect()
        # Since we only have path predictions for some timesteps (the ones not dropped above), we use this to fill
        # in the missing pieces by reorienting the past trajectory prediction into the frame of the current timestep
        self.map_batch_fill_missing = MapBatchFillMissing(self.params["local_map_size"], self.params["world_size_px"])

        # Passing true to freeze will freeze these weights regardless of whether they've been explicitly reloaded or not
        enable_weight_saving(self.sentence_embedding, "sentence_embedding", alwaysfreeze=False)

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
                other_features_size=self.params["emb_size"]
            )

        # Don't freeze the trajectory to action weights, because it will be pre-trained during path-prediction training
        # and finetuned on all timesteps end-to-end
        enable_weight_saving(self.map_to_action, "map_to_action", alwaysfreeze=False, neverfreeze=True)

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        # We add all auxiliaries that are necessary. The first argument is the auxiliary name, followed by parameters,
        # followed by variable number of names of inputs. ModuleWithAuxiliaries will automatically collect these inputs
        # that have been saved with keep_auxiliary_input() during execution
        if aux_class_features:
            self.add_auxiliary(ClassAuxiliary2D("aux_class", None,  self.params["feature_channels"], self.params["num_landmarks"], self.params["dropout"],
                                                "fpv_features", "lm_pos_fpv", "lm_indices"))
        if aux_grounding_features:
            self.add_auxiliary(ClassAuxiliary2D("aux_ground", None, self.params["relevance_channels"], 2, self.params["dropout"],
                                                "fpv_features_g", "lm_pos_fpv", "lm_mentioned"))
        if aux_class_map:
            self.add_auxiliary(ClassAuxiliary2D("aux_class_map", self.params["world_size_px"], self.params["feature_channels"], self.params["num_landmarks"], self.params["dropout"],
                                                "map_s_w_select", "lm_pos_map_select", "lm_indices_select"))
        if aux_grounding_map:
            self.add_auxiliary(ClassAuxiliary2D("aux_grounding_map", self.params["world_size_px"], self.params["relevance_channels"], 2, self.params["dropout"],
                                                "map_a_w_select", "lm_pos_map_select", "lm_mentioned_select"))
        if aux_goal_map:
            self.add_auxiliary(GoalAuxiliary2D("aux_goal_map", self.params["goal_channels"], self.params["world_size_px"],
                                               "map_b_w", "goal_pos_map"))
        # RSS model uses templated data for landmark and side prediction
        if self.use_aux_lang and self.params["templates"]:
            self.add_auxiliary(ClassAuxiliary("aux_lang_lm", self.params["emb_size"], self.params["num_landmarks"], 1,
                                                "sentence_embed", "lm_mentioned_tplt"))
            self.add_auxiliary(ClassAuxiliary("aux_lang_side", self.params["emb_size"], self.params["num_sides"], 1,
                                                "sentence_embed", "side_mentioned_tplt"))
        # CoRL model uses alignment-model groundings
        elif self.use_aux_lang:
            # one output for each landmark, 2 classes per output. This is for finetuning, so use the embedding that's gonna be fine tuned
            self.add_auxiliary(ClassAuxiliary("aux_lang_lm_nl", self.params["emb_size"], 2, self.params["num_landmarks"],
                                                "sentence_embed", "lang_lm_mentioned"))
        if self.use_aux_traj_on_map:
            self.add_auxiliary(PathAuxiliary2D("aux_path", "map_b_r_select", "traj_gt_r_select"))

        if self.use_aux_reg_map:
            self.add_auxiliary(FeatureRegularizationAuxiliary2D("aux_regularize_features", None, "l1",
                                                                "map_s_w_select", "lm_pos_map_select"))

        self.goal_good_criterion = GoalPredictionGoodCriterion(ok_distance=3.2)
        self.goal_acc_meter = MovingAverageMeter(10)

        self.print_auxiliary_info()

        self.action_loss = ActionLoss()

        self.env_id = None
        self.prev_instruction = None
        self.seq_step = 0

    # TODO: Try to hide these in a superclass or something. They take up a lot of space:
    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.sentence_embedding.cuda(device)
        self.map_accumulator_w.cuda(device)
        self.map_processor_a_w.cuda(device)
        self.map_processor_b_r.cuda(device)
        self.img_to_features_w.cuda(device)
        self.map_to_action.cuda(device)
        self.action_loss.cuda(device)
        self.map_batch_fill_missing.cuda(device)
        self.map_transform_w_to_r.cuda(device)
        self.map_transform_r_to_w.cuda(device)
        self.batch_select.cuda(device)
        self.map_batch_fill_missing.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.img_to_features_w.init_weights()
        self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()
        self.map_to_action.init_weights()
        self.map_processor_a_w.init_weights()
        self.map_processor_b_r.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        super(ModelTrajectoryTopDown, self).reset()
        self.sentence_embedding.reset()
        self.img_to_features_w.reset()
        self.map_accumulator_w.reset()
        self.map_processor_a_w.reset()
        self.map_processor_b_r.reset()
        self.map_transform_w_to_r.reset()
        self.map_transform_r_to_w.reset()
        self.map_batch_fill_missing.reset()
        self.prev_instruction = None

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def save_viz(self, images_in):
        imsave(get_viz_dir() + "fpv_" + str(self.seq_step) + ".png", images_in)
        features_cam = self.get_inputs_batch("fpv_features")[-1, 0, 0:3]
        save_tensor_as_img(features_cam, "F_c", self.env_id)
        feature_map_torch = self.get_inputs_batch("f_w")[-1, 0, 0:3]
        save_tensor_as_img(feature_map_torch, "F_w", self.env_id)
        coverage_map_torch = self.get_inputs_batch("m_w")[-1, 0, 0:3]
        save_tensor_as_img(coverage_map_torch, "M_w", self.env_id)
        semantic_map_torch = self.get_inputs_batch("map_s_w_select")[-1, 0, 0:3]
        save_tensor_as_img(semantic_map_torch, "S_w", self.env_id)
        relmap_torch = self.get_inputs_batch("map_a_w_select")[-1, 0, 0:3]
        save_tensor_as_img(relmap_torch, "R_w", self.env_id)
        relmap_r_torch = self.get_inputs_batch("map_a_r_select")[-1, 0, 0:3]
        save_tensor_as_img(relmap_r_torch, "R_r", self.env_id)
        goalmap_torch = self.get_inputs_batch("map_b_w_select")[-1, 0, 0:3]
        save_tensor_as_img(goalmap_torch, "G_w", self.env_id)
        goalmap_r_torch = self.get_inputs_batch("map_b_r_select")[-1, 0, 0:3]
        save_tensor_as_img(goalmap_r_torch, "G_r", self.env_id)

        action = self.get_inputs_batch("action")[-1].data.cpu().squeeze().numpy()
        action_fname = self.get_viz_dir() + "action_" + str(self.seq_step) + ".png"
        Presenter().save_action(action, action_fname, "")

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
        # Add the batch dimension

        first_step = True
        if instruction == self.prev_instruction:
            first_step = False
        self.prev_instruction = instruction

        img_in_t = image_fpv
        img_in_t.volatile = True

        instr_len = [len(instruction)] if instruction is not None else None
        instruction = torch.LongTensor(instruction).unsqueeze(0)
        instruction = cuda_var(instruction, self.is_cuda, self.cuda_device)

        state.volatile = True

        if self.is_cuda:
            if img_in_t is not None:
                img_in_t = img_in_t.cuda(self.cuda_device)
            state = state.cuda(self.cuda_device)

        step_enc = None
        plan_now = None

        self.seq_step += 1

        action = self(img_in_t, state, instruction, instr_len, plan=plan_now, pos_enc=step_enc)

        # Save materials for paper and presentation
        if False:
            self.save_viz(images_np_pure)

        output_action = action.squeeze().data.cpu().numpy()
        stop_prob = output_action[3]
        output_stop = 1 if stop_prob > 0.5 else 0
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

        pos_variance = 0
        rot_variance = 0
        if self.use_pos_noise:
            pos_variance = self.params["noisy_pos_variance"]
        if self.use_rot_noise:
            rot_variance = self.params["noisy_rot_variance"]

        pose = Pose(cam_pos, cam_rot)
        if self.use_pos_noise or self.use_rot_noise:
            pose = get_noisy_poses_torch(pose, pos_variance, rot_variance, cuda=self.is_cuda, cuda_device=self.cuda_device)
        return pose

    def forward(self, images, states, instructions, instr_lengths, has_obs=None, plan=None, save_maps_only=False, pos_enc=None, noisy_poses=None):
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
        g_poses = None#[None for pose in cam_poses]
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

        # Extract and project features onto the egocentric frame for each image
        features_w, coverages_w = self.img_to_features_w(images, cam_poses, sent_embeddings, self, show="")
        self.prof.tick("img_to_map_frame")
        self.keep_inputs("f_w", features_w)
        self.keep_inputs("m_w", coverages_w)

        # Accumulate the egocentric features in a global map
        maps_w = self.map_accumulator_w(features_w, coverages_w, add_mask=has_obs, show="acc" if IMG_DBG else "")
        map_poses_w = g_poses

        # TODO: Maybe keep maps_w if necessary
        #self.keep_inputs("map_sm_local", maps_m)
        self.prof.tick("map_accumulate")

        # Throw away those timesteps that don't correspond to planning timesteps
        maps_w_select, map_poses_w_select, cam_poses_select, noisy_poses_select, _, sent_embeddings_select, pos_enc = \
            self.batch_select(maps_w, map_poses_w, cam_poses, noisy_poses, None, sent_embeddings, pos_enc, plan)

        # Only process the maps on planning timesteps
        if len(maps_w_select) > 0:
            self.keep_inputs("map_s_w_select", maps_w_select)
            self.prof.tick("batch_select")

            # Process the map via the two map_procesors
            # Do grounding of objects in the map chosen to do so
            maps_w_select, map_poses_w_select = self.map_processor_a_w(maps_w_select, sent_embeddings_select, map_poses_w_select, show="")
            self.keep_inputs("map_a_w_select", maps_w_select)

            self.prof.tick("map_proc_gnd")

            self.map_transform_w_to_r.set_maps(maps_w_select, map_poses_w_select)
            maps_m_select, map_poses_m_select = self.map_transform_w_to_r.get_maps(cam_poses_select)

            self.keep_inputs("map_a_r_select", maps_w_select)
            self.prof.tick("transform_w_to_r")

            self.keep_inputs("map_a_r_perturbed_select", maps_m_select)

            self.prof.tick("map_perturb")

            # Include positional encoding for path prediction
            if pos_enc is not None:
                sent_embeddings_pp = torch.cat([sent_embeddings_select, pos_enc.unsqueeze(1)], dim=1)
            else:
                sent_embeddings_pp = sent_embeddings_select

            # Process the map via the two map_procesors (e.g. predict the trajectory that we'll be taking)
            maps_m_select, map_poses_m_select = self.map_processor_b_r(maps_m_select, sent_embeddings_pp, map_poses_m_select)

            self.keep_inputs("map_b_r_select", maps_m_select)

            if True:
                self.map_transform_r_to_w.set_maps(maps_m_select, map_poses_m_select)
                maps_b_w_select, _ = self.map_transform_r_to_w.get_maps(None)
                self.keep_inputs("map_b_w_select", maps_b_w_select)

            self.prof.tick("map_proc_b")

        else:
            maps_m_select = None

        maps_m, map_poses_m = self.map_batch_fill_missing(maps_m_select, cam_poses, plan, show="")
        self.keep_inputs("map_b_r", maps_m)
        self.prof.tick("map_fill_missing")

        # Keep global maps for auxiliary objectives if necessary
        if self.input_required("map_b_w"):
            maps_b, _ = self.map_processor_b_r.get_maps(g_poses)
            self.keep_inputs("map_b_w", maps_b)

        self.prof.tick("keep_global_maps")

        if run_metadata.IS_ROLLOUT:
            pass
            #Presenter().show_image(maps_m.data[0, 0:3], "plan_map_now", torch=True, scale=4, waitkey=1)
            #Presenter().show_image(maps_w.data[0, 0:3], "sm_map_now", torch=True, scale=4, waitkey=1)
        self.prof.tick("viz")

        # Output the final action given the processed map
        action_pred = self.map_to_action(maps_m, sent_embeddings)
        out_action = self.deterministic_action(action_pred[:, 0:3], None, action_pred[:, 3])

        self.keep_inputs("action", out_action)
        self.prof.tick("map_to_action")

        return out_action

    # TODO: The below two methods seem to do the same thing
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
        lm_pos_map = batch["lm_pos_map"]
        lm_indices = batch["lm_indices"]
        goal_pos_map = batch["goal_loc"]

        TEMPLATES = True
        if TEMPLATES:
            lm_mentioned_tplt = batch["lm_mentioned_tplt"]
            side_mentioned_tplt = batch["side_mentioned_tplt"]
        else:
            lm_mentioned = batch["lm_mentioned"]
            lang_lm_mentioned = batch["lang_lm_mentioned"]

        # stops = self.maybe_cuda(batch["stops"])
        masks = self.maybe_cuda(batch["masks"])
        # This is the first-timestep metadata
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
            b_lm_pos_map = lm_pos_map[b][:b_seq_len]
            b_lm_indices = lm_indices[b][:b_seq_len]
            b_goal_pos = goal_pos_map[b][:b_seq_len]
            if not TEMPLATES:
                b_lang_lm_mentioned = lang_lm_mentioned[b][:b_seq_len]
                b_lm_mentioned = lm_mentioned[b][:b_seq_len]

            b_lm_pos_map = [self.cuda_var(s.long()) if s is not None else None for s in b_lm_pos_map]
            b_lm_pos_fpv = [self.cuda_var((s / RESNET_FACTOR).long()) if s is not None else None for s in b_lm_pos_fpv]
            b_lm_indices = [self.cuda_var(s) if s is not None else None for s in b_lm_indices]
            b_goal_pos = self.cuda_var(b_goal_pos)
            if not TEMPLATES:
                b_lang_lm_mentioned = self.cuda_var(b_lang_lm_mentioned)
                b_lm_mentioned = [self.cuda_var(s) if s is not None else None for s in b_lm_mentioned]

            # TODO: Figure out how to keep these properly. Perhaps as a whole batch is best
            # TODO: Introduce a key-value store (encapsulate instead of inherit)
            self.keep_inputs("lm_pos_fpv", b_lm_pos_fpv)
            self.keep_inputs("lm_pos_map", b_lm_pos_map)
            self.keep_inputs("lm_indices", b_lm_indices)
            self.keep_inputs("goal_pos_map", b_goal_pos)
            if not TEMPLATES:
                self.keep_inputs("lang_lm_mentioned", b_lang_lm_mentioned)
                self.keep_inputs("lm_mentioned", b_lm_mentioned)

            # TODO: Abstract all of these if-elses in a modular way once we know which ones are necessary
            if TEMPLATES:
                b_lm_mentioned_tplt = lm_mentioned_tplt[b][:b_seq_len]
                b_side_mentioned_tplt = side_mentioned_tplt[b][:b_seq_len]
                b_side_mentioned_tplt = self.cuda_var(b_side_mentioned_tplt)
                b_lm_mentioned_tplt = self.cuda_var(b_lm_mentioned_tplt)
                self.keep_inputs("lm_mentioned_tplt", b_lm_mentioned_tplt)
                self.keep_inputs("side_mentioned_tplt", b_side_mentioned_tplt)

                b_lm_mentioned = b_lm_mentioned_tplt


            b_obs_mask = [True for _ in range(b_seq_len)]
            b_plan_mask = [True for _ in range(b_seq_len)]
            b_plan_mask_t_cpu = torch.Tensor(b_plan_mask) == True
            b_plan_mask_t = self.maybe_cuda(b_plan_mask_t_cpu)
            b_pos_enc = None

            # ----------------------------------------------------------------------------
            # Optional Auxiliary Inputs
            # ----------------------------------------------------------------------------
            if self.input_required("lm_pos_map_select"):
                b_lm_pos_map_select = [lm_pos for i,lm_pos in enumerate(b_lm_pos_map) if b_plan_mask[i]]
                self.keep_inputs("lm_pos_map_select", b_lm_pos_map_select)
            if self.input_required("lm_indices_select"):
                b_lm_indices_select = [lm_idx for i,lm_idx in enumerate(b_lm_indices) if b_plan_mask[i]]
                self.keep_inputs("lm_indices_select", b_lm_indices_select)
            if self.input_required("lm_mentioned_select"):
                b_lm_mentioned_select = [lm_m for i,lm_m in enumerate(b_lm_mentioned) if b_plan_mask[i]]
                self.keep_inputs("lm_mentioned_select", b_lm_mentioned_select)

            # ----------------------------------------------------------------------------

            self.prof.tick("inputs")

            actions = self(b_images, b_states, b_instructions, b_instr_len,
                           has_obs=b_obs_mask, plan=b_plan_mask, pos_enc=b_pos_enc)

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
        # TODO: Log value here
        self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), self.get_iter())

        self.prof.tick("auxiliaries")

        total_loss = action_loss_avg + aux_loss

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return total_loss

    def get_dataset(self, data=None, envs=None, dataset_name=None, eval=False):
        # TODO: Maybe use eval here
        #if self.fpv:
        data_sources = []
        # If we're running auxiliary objectives, we need to include the data sources for the auxiliary labels
        #if self.use_aux_class_features or self.use_aux_class_on_map or self.use_aux_grounding_features or self.use_aux_grounding_on_map:
        #if self.use_aux_goal_on_map:
        data_sources.append(aup.PROVIDER_LM_POS_DATA)
        data_sources.append(aup.PROVIDER_GOAL_POS)
        #data_sources.append(aup.PROVIDER_LANDMARKS_MENTIONED)
        data_sources.append(aup.PROVIDER_LANG_TEMPLATE)

        #if self.use_rot_noise or self.use_pos_noise:
        #    data_sources.append(aup.PROVIDER_POSE_NOISE)

        return SegmentDataset(data=data, env_list=envs, dataset_name=dataset_name, aux_provider_names=data_sources, segment_level=True)
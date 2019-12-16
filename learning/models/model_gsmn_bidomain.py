import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from imageio import imsave
import os

from data_io.model_io import load_pytorch_model, find_state_subdict
from data_io.instructions import debug_untokenize_instruction
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose, get_noisy_poses_torch
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones
from learning.inputs.vision import standardize_image
#from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
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
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.map_transformer_base import MapTransformerBase

from learning.modules.key_tensor_store import KeyTensorStore, save_tensors_as_images
from learning.modules.auxiliary_losses import AuxiliaryLosses

from utils.simple_profiler import SimpleProfiler
from visualization import Presenter
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.text2speech import say

try:
    from drones.aero_interface.rviz import RvizInterface
except ModuleNotFoundError:
    print("Not loading ROS")

from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import get_current_meters

from learning.utils import save_tensor_as_img_during_rollout, get_viz_dir_for_rollout

from parameters.parameter_server import get_current_parameters
from rollout import run_metadata as run_metadata

import transformations

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

# TODO:Currently this treats the sequence as a batch. Technically it should take inputs of size BxSx.... where B is
# the actual batch size and S is the sequence length. Currently everything is in Sx...

class ModelGSMNBiDomain(nn.Module):

    def __init__(self, run_name="", model_instance_name=""):

        super(ModelGSMNBiDomain, self).__init__()
        self.model_name = "gsmn_bidomain"
        self.run_name = run_name
        self.name = model_instance_name
        if not self.name:
            self.name = ""
        self.writer = LoggingSummaryWriter(log_dir=f"runs/{run_name}/{self.name}")

        self.params = get_current_parameters()["Model"]
        self.aux_weights = get_current_parameters()["AuxWeights"]
        self.use_aux = self.params["UseAuxiliaries"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.tensor_store = KeyTensorStore()
        self.aux_losses = AuxiliaryLosses()

        self.rviz = None
        if self.params.get("rviz"):
            self.rviz = RvizInterface(
                base_name="/gsmn/",
                map_topics=["semantic_map", "grounding_map", "goal_map"],
                markerarray_topics = ["instruction"])


        # Path-pred FPV model definition
        # --------------------------------------------------------------------------------------------------------------

        self.img_to_features_w = FPVToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"],
            res_channels=self.params["resnet_channels"], map_channels=self.params["feature_channels"],
            img_w=self.params["img_w"], img_h=self.params["img_h"], cam_h_fov=self.params["cam_h_fov"],img_dbg=IMG_DBG)

        self.map_accumulator_w = LeakyIntegratorGlobalMap(source_map_size=self.params["global_map_size"],
                                                          world_size_px=self.params["world_size_px"],
                                                          world_size_m=self.params["world_size_m"])

        # Pre-process the accumulated map to do language grounding if necessary - in the world reference frame
        if self.use_aux["grounding_map"] and not self.use_aux["grounding_features"]:
            self.map_processor_a_w = LangFilterMapProcessor(
                embed_size=self.params["emb_size"],
                in_channels=self.params["feature_channels"],
                out_channels=self.params["relevance_channels"],
                spatial=False, cat_out=True)
        else:
            self.map_processor_a_w = IdentityMapProcessor(
                source_map_size=self.params["global_map_size"],
                world_size_px=self.params["world_size_px"],
                world_size_m=self.params["world_size_m"])

        if self.use_aux["goal_map"]:
            self.map_processor_b_r = LangFilterMapProcessor(embed_size=self.params["emb_size"],
                                                            in_channels=self.params["relevance_channels"],
                                                            out_channels=self.params["goal_channels"],
                                                            spatial=self.params["spatial_goal_filter"],
                                                            cat_out=self.params["cat_rel_and_goal"])
        else:
            self.map_processor_b_r = IdentityMapProcessor(source_map_size=self.params["local_map_size"],
                                                          world_size_px=self.params["world_size_px"],
                                                          world_size_m=self.params["world_size_m"])

        # Common
        # --------------------------------------------------------------------------------------------------------------

        # Sentence Embedding
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"], dropout=0.0)

        self.map_transform_w_to_r = MapTransformerBase(source_map_size=self.params["global_map_size"],
                                                       dest_map_size=self.params["local_map_size"],
                                                       world_size_px=self.params["world_size_px"],
                                                       world_size_m=self.params["world_size_m"])
        self.map_transform_r_to_w = MapTransformerBase(source_map_size=self.params["local_map_size"],
                                                       dest_map_size=self.params["global_map_size"],
                                                       world_size_px=self.params["world_size_px"],
                                                       world_size_m=self.params["world_size_m"])

        # Output an action given the global semantic map
        if self.params["map_to_action"] == "downsample2":
            self.map_to_action = EgoMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"],
                other_features_size=self.params["emb_size"])

        elif self.params["map_to_action"] == "cropped":
            self.map_to_action = CroppedMapToActionTriplet(
                map_channels=self.params["map_to_act_channels"],
                map_size=self.params["local_map_size"]
            )

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        # We add all auxiliaries that are necessary. The first argument is the auxiliary name, followed by parameters,
        # followed by variable number of names of inputs. ModuleWithAuxiliaries will automatically collect these inputs
        # that have been saved with keep_auxiliary_input() during execution
        if self.use_aux["class_features"]:
            self.aux_losses.add_auxiliary(ClassAuxiliary2D("aux_class",  self.params["feature_channels"], self.params["num_landmarks"], self.params["dropout"],
                                                "fpv_features", "lm_pos_fpv_features", "lm_indices", "tensor_store"))
        if self.use_aux["grounding_features"]:
            self.aux_losses.add_auxiliary(ClassAuxiliary2D("aux_ground", self.params["relevance_channels"], 2, self.params["dropout"],
                                                "fpv_features_g", "lm_pos_fpv_features", "lm_mentioned", "tensor_store"))
        if self.use_aux["class_map"]:
            self.aux_losses.add_auxiliary(ClassAuxiliary2D("aux_class_map", self.params["feature_channels"], self.params["num_landmarks"], self.params["dropout"],
                                                "map_S_W", "lm_pos_map", "lm_indices", "tensor_store"))
        if self.use_aux["grounding_map"]:
            self.aux_losses.add_auxiliary(ClassAuxiliary2D("aux_grounding_map", self.params["relevance_channels"], 2, self.params["dropout"],
                                                "map_R_W", "lm_pos_map", "lm_mentioned", "tensor_store"))
        if self.use_aux["goal_map"]:
            self.aux_losses.add_auxiliary(GoalAuxiliary2D("aux_goal_map", self.params["goal_channels"], self.params["global_map_size"], "map_G_W", "goal_pos_map"))
        # RSS model uses templated data for landmark and side prediction
        if self.use_aux["language"] and self.params["templates"]:
            self.aux_losses.add_auxiliary(ClassAuxiliary("aux_lang_lm", self.params["emb_size"], self.params["num_landmarks"], 1,
                                                "sentence_embed", "lm_mentioned_tplt"))
            self.aux_losses.add_auxiliary(ClassAuxiliary("aux_lang_side", self.params["emb_size"], self.params["num_sides"], 1,
                                                "sentence_embed", "side_mentioned_tplt"))
        # CoRL model uses alignment-model groundings
        elif self.use_aux["language"]:
            # one output for each landmark, 2 classes per output. This is for finetuning, so use the embedding that's gonna be fine tuned
            self.aux_losses.add_auxiliary(ClassAuxiliary("aux_lang_lm_nl", self.params["emb_size"], 2, self.params["num_landmarks"],
                                                "sentence_embed", "lang_lm_mentioned"))
        if self.use_aux["l1_regularization"]:
            self.aux_losses.add_auxiliary(FeatureRegularizationAuxiliary2D("aux_regularize_features", "l1", "map_S_W"))
            self.aux_losses.add_auxiliary(FeatureRegularizationAuxiliary2D("aux_regularize_features", "l1", "map_R_W"))

        self.goal_acc_meter = MovingAverageMeter(10)

        self.aux_losses.print_auxiliary_info()

        self.action_loss = ActionLoss()

        self.env_id = None
        self.prev_instruction = None
        self.seq_step = 0

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        self.aux_losses.cuda(device)
        self.sentence_embedding.cuda(device)
        self.map_accumulator_w.cuda(device)
        self.map_processor_a_w.cuda(device)
        self.map_processor_b_r.cuda(device)
        self.img_to_features_w.cuda(device)
        self.map_to_action.cuda(device)
        self.action_loss.cuda(device)
        self.map_transform_w_to_r.cuda(device)
        self.map_transform_r_to_w.cuda(device)
        return self

    def steal_cross_domain_modules(self, other_self):
        # TODO: Consider whether to share auxiliary losses, and if so, all of them?
        self.aux_losses = other_self.aux_losses
        self.action_loss = other_self.action_loss

        # TODO: Make sure that none of these things are stateful, or that there are resets after every forward pass
        self.sentence_embedding = other_self.sentence_embedding
        self.map_accumulator_w = other_self.map_accumulator_w
        self.map_processor_a_w = other_self.map_processor_a_w
        self.map_processor_b_r = other_self.map_processor_b_r
        self.map_to_action = other_self.map_to_action

        # We'll have a separate one of these for each domain
        #self.img_to_features_w = other_self.img_to_features_w

        # TODO: Check that statefulness is not an issue in sharing modules
        # These have no parameters so no point sharing
        #self.map_transform_w_to_r = other_self.map_transform_w_to_r
        #self.map_transform_r_to_w = other_self.map_transform_r_to_w

    def both_domain_parameters(self, other_self):
        # This function iterates and yields parameters from this module and the other module, but does not yield
        # shared parameters twice.
        # First yield all of the other module's parameters
        for p in other_self.parameters():
            yield p
        # Then yield all the parameters from the this module that are not shared with the other one
        for p in self.img_to_features_w.parameters():
            yield p
        return

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def load_img_feature_weights(self):
        if self.params.get("load_feature_net"):
            filename = self.params.get("feature_net_filename")
            weights = load_pytorch_model(None, filename)
            prefix = self.params.get("feature_net_tensor_name")
            if prefix:
                weights = find_state_subdict(weights, prefix)
            # TODO: This breaks OOP conventions
            self.img_to_features_w.img_to_features.load_state_dict(weights)
            print(f"Loaded pretrained weights from file {filename} with prefix {prefix}")

    def init_weights(self):
        self.img_to_features_w.init_weights()
        self.load_img_feature_weights()
        self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()
        self.map_to_action.init_weights()
        self.map_processor_a_w.init_weights()
        self.map_processor_b_r.init_weights()

    def reset(self):
        self.tensor_store.reset()
        self.sentence_embedding.reset()
        self.img_to_features_w.reset()
        self.map_accumulator_w.reset()
        self.map_transform_w_to_r.reset()
        self.map_transform_r_to_w.reset()
        self.load_img_feature_weights()
        self.prev_instruction = None

    def set_env_context(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def save_viz(self, images_in, instruction):
        # Save incoming images
        imsave(os.path.join(get_viz_dir_for_rollout(), "fpv_" + str(self.seq_step) + ".png"), images_in)
        #self.tensor_store.keep_input("fpv_img", images_in)
        # Save all of these tensors from the tensor store as images
        save_tensors_as_images(self.tensor_store, [
            "images_w",
            "fpv_img",
            "fpv_features",
            "map_F_W",
            "map_M_W",
            "map_S_W",
            "map_R_W",
            "map_R_R",
            "map_G_R",
            "map_G_W"
        ], str(self.seq_step))

        # Save action as image
        action = self.tensor_store.get_inputs_batch("action")[-1].data.cpu().squeeze().numpy()
        action_fname = get_viz_dir_for_rollout() + "action_" + str(self.seq_step) + ".png"
        Presenter().save_action(action, action_fname, "")

        instruction_fname = get_viz_dir_for_rollout() + "instruction.txt"
        with open(instruction_fname, "w") as fp:
            fp.write(instruction)

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
        instruction_str = debug_untokenize_instruction(instruction)

        # TODO: Move this to PomdpInterface (for now it's here because this is already visualizing the maps)
        if first_step:
            if self.rviz is not None:
                self.rviz.publish_instruction_text("instruction", debug_untokenize_instruction(instruction))

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

        passive_mode_debug_projections = True
        if passive_mode_debug_projections:
            self.show_landmark_locations(loop=False, states=state)
            self.reset()

        # Run auxiliary objectives for debugging purposes (e.g. to compute classification predictions)
        if self.params.get("run_auxiliaries_at_test_time"):
            _, _ = self.aux_losses.calculate_aux_loss(self.tensor_store, reduce_average=True)
            overlaid = self.get_overlaid_classification_results(whole_batch=False)

        # Save materials for analysis and presentation
        if self.params["write_figures"]:
            self.save_viz(images_np_pure, instruction_str)


        output_action = action.squeeze().data.cpu().numpy()
        stop_prob = output_action[3]
        output_stop = 1 if stop_prob > self.params["stop_p"] else 0
        output_action[3] = output_stop

        return output_action

    def get_overlaid_classification_results(self, map_not_features=False):
        if map_not_features:
            predictions_name = "aux_class_map_predictions"
        else:
            predictions_name = "aux_class_predictions"
        predictions = self.tensor_store.get_latest_input(predictions_name)
        if predictions is None:
            return None
        predictions = predictions[0].detach()
        # Get the 3 channels corresponding to no landmark, banana and gorilla
        predictions = predictions[[0, 3, 24], :, :]
        images = self.tensor_store.get_latest_input("images")[0].detach()
        overlaid = Presenter().overlaid_image(images, predictions, gray_bg=True)
        return overlaid

    def deterministic_action(self, action_mean, action_std, stop_prob):
        batch_size = action_mean.size(0)
        action = Variable(empty_float_tensor((batch_size, 4), self.is_cuda, self.cuda_device))
        action[:, 0:3] = action_mean[:, 0:3]
        action[:, 3] = stop_prob
        return action

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
        if self.params.get("use_pos_noise"):
            pos_variance = self.params["noisy_pos_variance"]
        if self.params.get("use_rot_noise"):
            rot_variance = self.params["noisy_rot_variance"]

        pose = Pose(cam_pos, cam_rot)
        if self.params.get("use_pos_noise") or self.params.get("use_rot_noise"):
            pose = get_noisy_poses_torch(pose, pos_variance, rot_variance, cuda=self.is_cuda, cuda_device=self.cuda_device)
        return pose

    def forward(self, images, states, instructions, instr_lengths, has_obs=None, plan=None, save_maps_only=False, pos_enc=None, noisy_poses=None, halfway=False):
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

        #str_instr = debug_untokenize_instruction(instructions[0].data[:instr_lengths[0]])
        #print("Trn: " + str_instr)

        # Calculate the instruction embedding
        if instructions is not None:
            # TODO: Take batch of instructions and their lengths, return batch of embeddings. Store the last one as internal state
            sent_embeddings = self.sentence_embedding(instructions, instr_lengths)
            self.tensor_store.keep_inputs("sentence_embed", sent_embeddings)
        else:
            sent_embeddings = self.sentence_embedding.get()

        self.prof.tick("embed")

        # Extract and project features onto the egocentric frame for each image
        features_w, coverages_w = self.img_to_features_w(images, cam_poses, sent_embeddings, self.tensor_store, show="")

        # If we're running the model halway, return now. This is to compute enough features for the wasserstein critic, but no more
        if halfway:
            return None

        # Don't back-prop into resnet if we're freezing these features (TODO: instead set requires grad to false)
        if self.params.get("freeze_feature_net"):
            features_w = features_w.detach()

        self.prof.tick("img_to_map_frame")
        self.tensor_store.keep_inputs("images", images)
        self.tensor_store.keep_inputs("map_F_w", features_w)
        self.tensor_store.keep_inputs("map_M_w", coverages_w)

        if run_metadata.IS_ROLLOUT:
            Presenter().show_image(features_w.data[0, 0:3], "F", torch=True, scale=8, waitkey=1)

        # Accumulate the egocentric features in a global map
        maps_s_w = self.map_accumulator_w(features_w, coverages_w, add_mask=has_obs, show="acc" if IMG_DBG else "")
        map_poses_w = g_poses
        self.tensor_store.keep_inputs("map_S_W", maps_s_w)
        self.prof.tick("map_accumulate")

        Presenter().show_image(maps_s_w.data[0], f"{self.name}_S_map_W", torch=True, scale=4, waitkey=1)

        # Do grounding of objects in the map chosen to do so
        maps_r_w, map_poses_r_w = self.map_processor_a_w(maps_s_w, sent_embeddings, map_poses_w, show="")
        self.tensor_store.keep_inputs("map_R_W", maps_r_w)
        Presenter().show_image(maps_r_w.data[0], f"{self.name}_R_map_W", torch=True, scale=4, waitkey=1)
        self.prof.tick("map_proc_gnd")

        # Transform to drone's reference frame
        self.map_transform_w_to_r.set_maps(maps_r_w, map_poses_r_w)
        maps_r_r, map_poses_r_r = self.map_transform_w_to_r.get_maps(cam_poses)
        self.tensor_store.keep_inputs("map_R_R", maps_r_r)
        self.prof.tick("transform_w_to_r")

        # Predict goal location
        maps_g_r, map_poses_g_r = self.map_processor_b_r(maps_r_r, sent_embeddings, map_poses_r_r)
        self.tensor_store.keep_inputs("map_G_R", maps_g_r)

        # Transform back to map frame
        self.map_transform_r_to_w.set_maps(maps_g_r, map_poses_g_r)
        maps_g_w, _ = self.map_transform_r_to_w.get_maps(None)
        self.tensor_store.keep_inputs("map_G_W", maps_g_w)
        self.prof.tick("map_proc_b")

        # Show and publish to RVIZ
        Presenter().show_image(maps_g_w.data[0], f"{self.name}_G_map_W", torch=True, scale=8, waitkey=1)
        if self.rviz:
            self.rviz.publish_map("goal_map", maps_g_w[0].data.cpu().numpy().transpose(1,2,0), self.params["world_size_m"])

        # Output the final action given the processed map
        action_pred = self.map_to_action(maps_g_r, sent_embeddings)
        out_action = self.deterministic_action(action_pred[:, 0:3], None, action_pred[:, 3])
        self.tensor_store.keep_inputs("action", out_action)
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

    def unbatch(self, batch):
        # TODO: Carefully consider this line. This is necessary to reset state between batches (e.g. delete all tensors in the tensor store)
        self.reset()
        # Get rid of the batch dimension for everything
        images = self.maybe_cuda(batch["images"])[0]
        seq_len = images.shape[0]
        instructions = self.maybe_cuda(batch["instr"])[0][:seq_len]
        instr_lengths = batch["instr_len"][0]
        states = self.maybe_cuda(batch["states"])[0]
        actions = self.maybe_cuda(batch["actions"])[0]

        # Auxiliary labels
        lm_pos_fpv = batch["lm_pos_fpv"][0]
        lm_pos_map = batch["lm_pos_map"][0]
        lm_indices = batch["lm_indices"][0]
        goal_pos_map = batch["goal_loc"][0]

        # TODO: Get rid of this. We will have lm_mentioned booleans and lm_mentioned_idx integers and that's it.
        TEMPLATES = True
        if TEMPLATES:
            lm_mentioned_tplt = batch["lm_mentioned_tplt"][0]
            side_mentioned_tplt = batch["side_mentioned_tplt"][0]
            side_mentioned_tplt = self.cuda_var(side_mentioned_tplt)
            lm_mentioned_tplt = self.cuda_var(lm_mentioned_tplt)
            lang_lm_mentioned = None
        else:
            lm_mentioned_tplt = None
            side_mentioned_tplt = None
            lang_lm_mentioned = batch["lang_lm_mentioned"][0]
        lm_mentioned = batch["lm_mentioned"][0]
        # This is the first-timestep metadata
        metadata = batch["md"][0]

        lm_pos_map = [torch.from_numpy(transformations.pos_m_to_px(p.numpy(),
                                                   self.params["global_map_size"],
                                                   self.params["world_size_m"],
                                                   self.params["world_size_px"]))
                        if p is not None else None for p in lm_pos_map]

        goal_pos_map = torch.from_numpy(transformations.pos_m_to_px(goal_pos_map.numpy(),
                                                           self.params["global_map_size"],
                                                           self.params["world_size_m"],
                                                           self.params["world_size_px"]))

        lm_pos_map = [self.cuda_var(s.long()) if s is not None else None for s in lm_pos_map]
        lm_pos_fpv_features = [self.cuda_var((s / self.img_to_features_w.img_to_features.get_downscale_factor()).long()) if s is not None else None for s in lm_pos_fpv]
        lm_pos_fpv_img = [self.cuda_var(s.long()) if s is not None else None for s in lm_pos_fpv]
        lm_indices = [self.cuda_var(s) if s is not None else None for s in lm_indices]
        goal_pos_map = self.cuda_var(goal_pos_map)
        if not TEMPLATES:
            lang_lm_mentioned = self.cuda_var(lang_lm_mentioned)
        lm_mentioned = [self.cuda_var(s) if s is not None else None for s in lm_mentioned]

        obs_mask = [True for _ in range(seq_len)]
        plan_mask = [True for _ in range(seq_len)]
        pos_enc = None

        # TODO: Figure out how to keep these properly. Perhaps as a whole batch is best
        self.tensor_store.keep_inputs("lm_pos_fpv_img", lm_pos_fpv_img)
        self.tensor_store.keep_inputs("lm_pos_fpv_features", lm_pos_fpv_features)
        self.tensor_store.keep_inputs("lm_pos_map", lm_pos_map)
        self.tensor_store.keep_inputs("lm_indices", lm_indices)
        self.tensor_store.keep_inputs("goal_pos_map", goal_pos_map)
        if not TEMPLATES:
            self.tensor_store.keep_inputs("lang_lm_mentioned", lang_lm_mentioned)
        else:
            self.tensor_store.keep_inputs("lm_mentioned_tplt", lm_mentioned_tplt)
            self.tensor_store.keep_inputs("side_mentioned_tplt", side_mentioned_tplt)
        self.tensor_store.keep_inputs("lm_mentioned", lm_mentioned)

        # ----------------------------------------------------------------------------
        # Optional Auxiliary Inputs
        # ----------------------------------------------------------------------------
        #if self.aux_losses.input_required("lm_pos_map"):
        self.tensor_store.keep_inputs("lm_pos_map", lm_pos_map)
        #if self.aux_losses.input_required("lm_indices"):
        self.tensor_store.keep_inputs("lm_indices", lm_indices)
        #if self.aux_losses.input_required("lm_mentioned"):
        self.tensor_store.keep_inputs("lm_mentioned", lm_mentioned)

        return images, instructions, instr_lengths, states, actions, \
               lm_pos_fpv_img, lm_pos_fpv_features, lm_pos_map, lm_indices, goal_pos_map, \
               lm_mentioned, lm_mentioned_tplt, side_mentioned_tplt, lang_lm_mentioned, \
               metadata, obs_mask, plan_mask, pos_enc

    def show_landmark_locations(self, loop=True, states=None):
        # Show landmark locations in first-person images
        img_all = self.tensor_store.get("images")
        img_w_all = self.tensor_store.get("images_w")
        import rollout.run_metadata as md
        if md.IS_ROLLOUT:
            # TODO: Discard this and move this to PomdpInterface or something
            # (it's got nothing to do with the model)
            # load landmark positions from configs
            from data_io.env import load_env_config
            from learning.datasets.aux_data_providers import get_landmark_locations_airsim
            from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjection
            projector = PinholeCameraProjection(
                map_size_px=self.params["global_map_size"],
                world_size_px=self.params["world_size_px"],
                world_size_m=self.params["world_size_m"],
                img_x=self.params["img_w"],
                img_y=self.params["img_h"],
                cam_fov=self.params["cam_h_fov"],
                #TODO: Handle correctly
                domain="sim",
                use_depth=False
            )
            conf_json = load_env_config(md.ENV_ID)
            landmark_names, landmark_indices, landmark_pos = get_landmark_locations_airsim(conf_json)
            cam_poses = self.cam_poses_from_states(states)
            cam_pos = cam_poses.position[0]
            cam_rot = cam_poses.orientation[0]
            lm_pos_map_all = []
            lm_pos_img_all = []
            for i, landmark_in_world in enumerate(landmark_pos):
                lm_pos_img, landmark_in_cam, status = projector.world_point_to_image(cam_pos, cam_rot, landmark_in_world)
                lm_pos_map = torch.from_numpy(transformations.pos_m_to_px(landmark_in_world[np.newaxis, :],
                                                                           self.params["global_map_size"],
                                                                           self.params["world_size_m"],
                                                                           self.params["world_size_px"]))
                lm_pos_map_all += [lm_pos_map[0]]
                if lm_pos_img is not None:
                    lm_pos_img_all += [lm_pos_img]

            lm_pos_img_all = [lm_pos_img_all]
            lm_pos_map_all = [lm_pos_map_all]

        else:
            lm_pos_img_all = self.tensor_store.get("lm_pos_fpv_img")
            lm_pos_map_all = self.tensor_store.get("lm_pos_map")

        print("Plotting landmark points")

        for i in range(len(img_all)):
            p = Presenter()
            overlay_fpv = p.overlay_pts_on_image(img_all[i][0], lm_pos_img_all[i])
            overlay_map = p.overlay_pts_on_image(img_w_all[i][0], lm_pos_map_all[i])
            p.show_image(overlay_fpv, "landmarks_on_fpv_img", scale=8)
            p.show_image(overlay_map, "landmarks_on_map", scale=20)

            if not loop:
                break

    def calc_tensor_statistics(self, prefix, tensor):
        stats = {}
        stats[f"{prefix}_mean"] = torch.mean(tensor).item()
        stats[f"{prefix}_l2"] = torch.norm(tensor).item()
        stats[f"{prefix}_stddev"] = torch.std(tensor).item()
        return stats

    def get_activation_statistics(self, keys):
        stats = {}
        from utils.dict_tools import dict_merge
        for key in keys:
            t = self.tensor_store.get_inputs_batch(key)
            t_stats = self.calc_tensor_statistics(key, t)
            stats = dict_merge(stats, t_stats)
        return stats

    # Forward pass for training (with batch optimizations
    def sup_loss_on_batch(self, batch, eval, halfway=False):
        self.prof.tick("out")

        action_loss_total = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))

        if batch is None:
            print("Skipping None Batch")
            return action_loss_total

        images, instructions, instr_lengths, states, action_labels, \
        lm_pos_fpv_img, lm_pos_fpv_features, lm_pos_map, lm_indices, goal_pos_map, \
        lm_mentioned, lm_mentioned_tplt, side_mentioned_tplt, lang_lm_mentioned, \
        metadata, obs_mask, plan_mask, pos_enc = self.unbatch(batch)

        # ----------------------------------------------------------------------------
        self.prof.tick("inputs")

        pred_actions = self(images, states, instructions, instr_lengths,
                       has_obs=obs_mask, plan=plan_mask, pos_enc=pos_enc, halfway=halfway)

        # Debugging landmark locations
        if False:
            self.show_landmark_locations()

        # Don't compute any losses - those will not be used. All we care about are the intermediate activations
        if halfway:
            return None, self.tensor_store

        action_losses, _ = self.action_loss(action_labels, pred_actions, batchreduce=False)

        self.prof.tick("call")

        action_losses = self.action_loss.batch_reduce_loss(action_losses)
        action_loss = self.action_loss.reduce_loss(action_losses)

        action_loss_total = action_loss

        self.prof.tick("loss")

        aux_losses, aux_metrics = self.aux_losses.calculate_aux_loss(self.tensor_store, reduce_average=True)
        aux_loss = self.aux_losses.combine_losses(aux_losses, self.aux_weights)

        #overlaid = self.get_overlaid_classification_results()
        #Presenter().show_image(overlaid, "classification", scale=2)

        prefix = f"{self.model_name}/{'eval' if eval else 'train'}"
        act_prefix = f"{self.model_name}_activations/{'eval' if eval else 'train'}"

        # Mean, stddev, norm of maps
        act_stats = self.get_activation_statistics(["map_S_W", "map_R_W", "map_G_W"])
        self.writer.add_dict(act_prefix, act_stats, self.get_iter())

        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())
        self.writer.add_dict(prefix, aux_losses, self.get_iter())
        self.writer.add_dict(prefix, aux_metrics, self.get_iter())
        self.writer.add_scalar(prefix + "/action_loss", action_loss_total.data.cpu().item(), self.get_iter())
        # TODO: Log value here
        self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), self.get_iter())

        self.prof.tick("auxiliaries")

        total_loss = action_loss_total + aux_loss

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return total_loss, self.tensor_store

    def get_dataset(self, data=None, envs=None, dataset_names=None, dataset_prefix=None, eval=False):
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

        return SegmentDataset(data=data, env_list=envs, dataset_names=dataset_names, dataset_prefix=dataset_prefix, aux_provider_names=data_sources, segment_level=True)
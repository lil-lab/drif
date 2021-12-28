import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from bert.bert_tools import OBJ_REF_TOK
from data_io.instructions import get_all_instructions, get_word_to_token_map
from data_io.instructions import tokenize_instruction
from learning.datasets.dynamic_object_database import DynamicObjectDatabase
from learning.datasets.static_object_database import StaticObjectDatabase
from learning.datasets.nav_around_novel_objects_dataset import NavAroundNovelObjectsDataset
from learning.datasets.object_reference_dataset import ObjectReferenceDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.pose import Pose, get_noisy_poses_torch
from learning.modules.auxiliaries.path_auxiliary import PathAuxiliary2D
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.modules.img_to_map.project_to_global_map import ProjectToGlobalMap
from learning.modules.map_to_map.lingunet_path_predictor import LingunetPathPredictor
from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegrator
from learning.modules.map_transformer import MapTransformer
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.visitation_softmax import VisitationSoftmax

from learning.modules.generic_model_state import GenericModelState

from learning.inputs.partial_2d_distribution import Partial2DDistribution
from learning.modules.add_drone_pos_to_coverage_mask import AddDroneInitPosToCoverage
from learning.modules.grounding.object_reference_tagger import ObjectReferenceTagger
from learning.modules.grounding.language_conditioned_segmentation import LanguageConditionedSegmentation
from learning.modules.grounding.context_grounding_map import ContextGroundingMap
from learning.modules.grounding.segmentation_mask_accumulator import SegmentationMaskAccumulator

from learning.modules.losses.distributed_visitation_discriminator import DistributedVisitationDiscriminator
from learning.modules.losses.visitation_discriminator import VisitationDiscriminator

import learning.models.visualization.viz_html_rpn_fs_stage1_bidomain as viz

from learning.models.navigation_model_component_base import NavigationModelComponentBase

from utils.simple_profiler import SimpleProfiler
from utils.assertion import assert_shape

from learning.utils import draw_drone_poses

from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import get_current_meters

from utils.dict_tools import dict_merge, dict_merge_deep
from utils import dict_tools

import parameters.parameter_server as P

from parameters.parameter_server import get_current_parameters
import transformations
from visualization import Presenter

#from memory_profiler import profile

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class RPN_FewShot_Stage1_Bidomain(NavigationModelComponentBase):

    #@profile
    def __init__(self, run_name="", domain="sim", nowriter=False):
        super(RPN_FewShot_Stage1_Bidomain, self).__init__(run_name, domain, model_name="fspvn_stage1", nowriter=nowriter)
        self.root_params = P.get("ModelPVN")
        self.params = P.get("ModelPVN::Stage1")
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.query_scale = 32

        self.losses = AuxiliaryLosses()

        self.do_perturb_maps = self.params["perturb_maps"]
        print("Perturbing maps: ", self.do_perturb_maps)

        # --------------------------------------------------------------------------------------------------------------
        # Ablations
        self.deaf = self.params.get("deaf", False)
        self.blind = self.params.get("blind", False)
        self.noling = self.params.get("noLing", False)
        if self.blind:
            print("RUNNING BLIND ABLATION")
        if self.deaf:
            print("RUNNING DEAF ABLATION")
        if self.noling:
            print("RUNNING WITHOUT LINGUNET LANGUAGE CONDITIONING")


        # --------------------------------------------------------------------------------------------------------------
        # Novel Object Specifics
        # --------------------------------------------------------------------------------------------------------------
        # TODO: Figure this out
        real_drone = get_current_parameters()["Setup"]["real_drone"]
        if real_drone:
            nod_path_train = get_current_parameters()["Data"]["object_database_real"]
        else:
            nod_path_train = get_current_parameters()["Data"]["object_database_sim"]
        static_nod = self.params.get("static_nod")
        NodClass = StaticObjectDatabase if static_nod else DynamicObjectDatabase
        print(f"Using {'STATIC' if static_nod else 'DYNAMIC'} object database!")
        self.object_database_dataset = NodClass(nod_path_train, self.query_scale)
        # --------------------------------------------------------------------------------------------------------------
        # Instruction processing!
        # --------------------------------------------------------------------------------------------------------------
        self.ord = ObjectReferenceDataset()

        self.token_embedding = nn.Embedding(num_embeddings=self.params["vocab_size"],
                                            embedding_dim=self.params["word_emb_size"])
        self.anon_instruction_lstm = nn.LSTM(input_size=self.params["word_emb_size"],
                                             hidden_size=self.params["context_hidden_size"],
                                             num_layers=self.params["context_num_layers"],
                                             bias=True,
                                             batch_first=True,
                                             bidirectional=self.params["context_bidirectional"])

        self.full_instruction_lstm = nn.LSTM(input_size=self.params["word_emb_size"],
                                             hidden_size = self.params["full_hidden_size"],
                                             num_layers=self.params["full_num_layers"],
                                             bias=True,
                                             batch_first=True,
                                             bidirectional=self.params["full_bidirectional"])

        assert self.params["context_hidden_size"] * (2 if self.params["context_bidirectional"] else 1) == self.params["context_embedding_size"], (
            "Context embedding size must match the LSTM output based on hidden sie and directionality")
        #assert self.params["full_hidden_size"] * (2 if self.params["full_bidirectional"] else 1) == self.params["full_embedding_size"], (
        #    "Instruction embedding size must match the LSTM output based on hidden size and directionality")

        # -----------------------------------------------------------------
        # Instance Recognition Pipeline
        # -----------------------------------------------------------------

        self.object_reference_tagger = ObjectReferenceTagger(run_name=run_name, domain=domain)
        self.language_conditioned_segmentation = LanguageConditionedSegmentation(run_name=run_name, domain=domain)

        self.context_grounding_map = ContextGroundingMap()

        # -----------------------------------------------------------------

        self.project_c_to_w = ProjectToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"], world_size_m=self.params["world_size_m"],
            img_w=self.params["img_w"], img_h=self.params["img_h"], cam_h_fov=self.params["cam_h_fov"],
            domain=domain, img_dbg=IMG_DBG
        )

        self.map_accumulator_w = SegmentationMaskAccumulator(name="mask_accumulator")

        self.add_init_pos_to_coverage = AddDroneInitPosToCoverage(
            world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"],
            map_size_px=self.params["local_map_size"])

        # Process the global accumulated map
        # +1 for map boundaries, +1 for generic object layer.
        # TODO: Finish generic object layer
        MAP_BOUNDARY_LAYERS = 4
        COVERAGE_LAYERS = 1
        GENERIC_OBJECT_LAYERS = 1
        lingunet_in_channels = self.params["context_embedding_size"] + MAP_BOUNDARY_LAYERS + COVERAGE_LAYERS + GENERIC_OBJECT_LAYERS
        self.path_predictor_lingunet = LingunetPathPredictor(
            self.params["lingunet"],
            posterior_channels_in=lingunet_in_channels,
            oob=self.params["clip_observability"],
            noling=self.noling
        )

        # Draw map boundaries - have 4 channels for the 4 sides
        wspx = self.params["world_size_px"]
        self.boundary_mask = torch.zeros([1, 4, wspx, wspx], dtype=torch.float32)
        self.boundary_mask[:, 0, :, 0] = 1.0
        self.boundary_mask[:, 1, :, wspx-1] = 1.0
        self.boundary_mask[:, 2, 0, :] = 1.0
        self.boundary_mask[:, 3, wspx-1, :] = 1.0

        self.second_transform = self.do_perturb_maps or self.params["predict_in_start_frame"]

        # Transformations
        # --------------------------------------------------------------------------------------------------------------

        self.map_transform_local_to_local = MapTransformer(source_map_size=self.params["local_map_size"],
                                                           dest_map_size=self.params["local_map_size"],
                                                           world_size_px=self.params["world_size_px"],
                                                           world_size_m=self.params["world_size_m"])

        self.map_transform_global_to_local = MapTransformer(source_map_size=self.params["global_map_size"],
                                                       dest_map_size=self.params["local_map_size"],
                                                       world_size_px=self.params["world_size_px"],
                                                       world_size_m=self.params["world_size_m"])

        self.map_transform_local_to_global = MapTransformer(source_map_size=self.params["local_map_size"],
                                                       dest_map_size=self.params["global_map_size"],
                                                       world_size_px=self.params["world_size_px"],
                                                       world_size_m=self.params["world_size_m"])

        self.map_transform_s_to_p = self.map_transform_local_to_local
        self.map_transform_w_to_s = self.map_transform_global_to_local
        self.map_transform_w_to_r = self.map_transform_global_to_local
        self.map_transform_r_to_s = self.map_transform_local_to_local
        self.map_transform_r_to_w = self.map_transform_local_to_global
        self.map_transform_p_to_w = self.map_transform_local_to_global
        self.map_transform_p_to_r = self.map_transform_local_to_local

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        # We add all auxiliaries that are necessary. The first argument is the auxiliary name, followed by parameters,
        # followed by variable number of names of inputs. ModuleWithAuxiliaries will automatically collect these inputs
        # that have been saved with keep_auxiliary_input() during execution
        lossfunc = self.params["path_loss_function"]
        self.losses.add_auxiliary(PathAuxiliary2D("visitation_dist", lossfunc, self.params["clip_observability"],
                                                  "log_v_dist_s", "v_dist_s_ground_truth", "accum_obs_masks_s"))
        self.loss_weights = {"visitation_dist": 1.0}

        if self.params.get("use_visitation_discriminator"):
            self.discriminator_model = VisitationDiscriminator()
            # Freeze this copy of the discriminator model
            for param in self.discriminator_model.parameters():
                param.requires_grad = False
            if self.params.get("distributed_training_of_visitation_discriminator"):
                self.distributed_visitation_discriminator = DistributedVisitationDiscriminator(self.discriminator_model)
            else:
                self.distributed_visitation_discriminator = self.discriminator_model

        # Misc
        # --------------------------------------------------------------------------------------------------------------

        self.grounding_map_norm = nn.InstanceNorm2d(128)
        self.semantic_map_norm = nn.InstanceNorm2d(32)

        self.spatialsoftmax = SpatialSoftmax2d()
        self.visitation_softmax = VisitationSoftmax()

        self.goal_good_criterion = GoalPredictionGoodCriterion(ok_distance=self.params["world_size_px"]*0.1)
        self.goal_acc_meter = MovingAverageMeter(10)
        self.visible_goal_acc_meter = MovingAverageMeter(10)
        self.invisible_goal_acc_meter = MovingAverageMeter(10)
        self.visible_goal_frac_meter = MovingAverageMeter(10)

        self.losses.print_auxiliary_info()

        # Construct word2token and token2word maps, and add in the additional object reference token
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)
        max_idx = max(self.word2token.values())
        self.obj_ref_tok_idx = max_idx + 1
        self.token2word[self.obj_ref_tok_idx] = OBJ_REF_TOK
        self.word2token[OBJ_REF_TOK] = self.obj_ref_tok_idx

        self.total_goals = 0
        self.correct_goals = 0

        self.should_save_path_overlays = False

        # This model loads the pre-trained modules from disk on the first forward call.
        self.loaded_pretrained_modules = False
        self.loaded_pref_pretrained_modules = False

        self.model_state = GenericModelState()

    def enable_logging(self):
        super().enable_logging()
        self.language_conditioned_segmentation.enable_logging()
        self.object_reference_tagger.enable_logging()

    def make_picklable(self):
        super().make_picklable()
        self.language_conditioned_segmentation.make_picklable()
        self.object_reference_tagger.make_picklable()

    def steal_cross_domain_modules(self, other_self):
        super().steal_cross_domain_modules(other_self)
        self.losses = other_self.losses
        self.full_instruction_lstm = other_self.full_instruction_lstm
        self.anon_instruction_lstm = other_self.anon_instruction_lstm
        self.token_embedding = other_self.token_embedding
        self.map_accumulator_w = other_self.map_accumulator_w
        self.path_predictor_lingunet = other_self.path_predictor_lingunet

    def load_pref_pretrained_modules(self):
        if not self.loaded_pref_pretrained_modules:
            super().load_pref_pretrained_modules()
            print("FSPVN Stage1: Loading Pre-Forward Pretrained Modules")
            obj_recognizer_model_name = self.root_params.get("object_recognizer_model_name")
            print(f"Loading object reference recognizer: {obj_recognizer_model_name}")
            self.object_reference_tagger.load_object_reference_recognizer(obj_recognizer_model_name)
            real_drone = get_current_parameters()["Setup"]["real_drone"]
            segmentation_params = self.params.get("SegmentationModel")
            self.language_conditioned_segmentation.load_pref_pretrained_modules(segmentation_params, real_drone)
            self.loaded_pref_pretrained_modules = True

    def load_pretrained_modules(self):
        if not self.loaded_pretrained_modules:
            super().load_pretrained_modules()
            print("FSPVN Stage1: Loading Post-Forward Pretrained Modules")
            real_drone = get_current_parameters()["Setup"]["real_drone"]
            segmentation_params = self.params.get("SegmentationModel")
            self.language_conditioned_segmentation.load_pretrained_modules(segmentation_params, real_drone)
            self.loaded_pretrained_modules = True

    def _lstm_init(self, lstm):
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def init_weights(self):
        super().init_weights()
        self.path_predictor_lingunet.init_weights()
        self.token_embedding.weight.data.normal_(0, 1)
        self._lstm_init(self.anon_instruction_lstm)
        self._lstm_init(self.full_instruction_lstm)

    def goal_visible(self, masks, goal_pos):
        goal_mask = masks.detach()[0, 0, :, :]
        goal_pos = goal_pos[0].long().detach()
        visible = bool((goal_mask[goal_pos[0], goal_pos[1]] > 0.5).detach().cpu().item())
        return visible

    def _get_boundary_mask(self, reference_tensor):
        self.boundary_mask = self.boundary_mask.to(reference_tensor.device)
        bmask_out = self.boundary_mask.repeat([reference_tensor.shape[0], 1, 1, 1])
        return bmask_out

    def start_rollout(self, env_id, set_idx, seg_idx):
        super()
        # Load novel object data
        device = next(self.parameters()).device
        object_database = self.object_database_dataset.build_for_env(env_id, device=device)
        self.model_state.put("object_database", object_database)

    def print_tensor_stats(self, name, tensor):
        if tensor.contiguous().view([-1]).shape[0] > 0:
            prefix = f"{self.model_name}_stats_{self.domain}/{name}/"
            iteration = self.get_iter()
            self.writer.add_scalar(prefix + "min", tensor.min().item(), iteration)
            self.writer.add_scalar(prefix + "max", tensor.max().item(), iteration)
            self.writer.add_scalar(prefix + "mean", tensor.mean().item(), iteration)
            self.writer.add_scalar(prefix + "std", tensor.std().item(), iteration)

    def preprocess_batch(self, batch):
        """
        :param batch:
        :return:
        """
        batch = super().preprocess_batch(batch)
        string_instructions_nl = [b[0] for b in batch["instr_nl"]]
        add_to_batch = self.pre_forward(string_instructions_nl[0], batch["object_database"][0])
        batch = dict_tools.dict_merge_deep(batch, add_to_batch)
        return batch

    def forward(self, images, states, instruction, reset_mask, noisy_start_poses=None, start_poses=None, rl=False):
        """
        This function is only called at test-time. During training, the pre_forward and post_forward functions are
        called separately. pre_forward is called form within the dataset, post_forward is called from the learning loop.
        :param images:
        :param states:
        :param instruction:
        :param reset_mask:
        :param noisy_start_poses:
        :param start_poses:
        :param rl:
        :return:
        """
        self.full_instruction_lstm.flatten_parameters()
        self.anon_instruction_lstm.flatten_parameters()
        object_database = self.model_state.get("object_database")
        # Compute these inputs only on the FIRST timestep - they only depend on novel object dataset and instruction
        first_step_done = self.model_state.get("computed_first_step", False)
        if not first_step_done:
            add_to_batch = self.pre_forward(instruction, object_database)
            object_database = dict_merge_deep(object_database, add_to_batch["object_database"][0])

            anon_instr_tokenized = add_to_batch["anon_instr"]
            anon_instr_tok_t = torch.tensor(anon_instr_tokenized, device=images.device, dtype=torch.long)

            # Add things that we will definitely need later to the model state
            self.model_state.put("object_database", object_database)
            self.model_state.put("anon_instr", anon_instr_tok_t)
            self.model_state.put("anon_instr_ref_tok_indices", add_to_batch["anon_instr_ref_tok_indices"][0])
            self.model_state.put("text_similarity_matrix", add_to_batch["text_similarity_matrix"][0].to(images.device))

            # Add these to model state for later visualization - these are not actually needed otherwise
            self.model_state.put("anon_instr_nl", add_to_batch["anon_instr_nl"])
            self.model_state.put("object_references", add_to_batch["object_references"])
            self.model_state.put("noun_chunks", add_to_batch["noun_chunks"])

            # Mark that we don't have to run text processing again - that's only needed on the first instruction step
            self.model_state.put("computed_first_step", True)

        anon_instruction = self.model_state.get("anon_instr")
        anon_instr_ref_tok_indices = self.model_state.get("anon_instr_ref_tok_indices")
        text_similarity_matrix = self.model_state.get("text_similarity_matrix")

        returns = self.post_forward(images, states,
                                    anon_instruction,
                                    anon_instr_ref_tok_indices,
                                    object_database,
                                    text_similarity_matrix=text_similarity_matrix,
                                    reset_mask=reset_mask,
                                    noisy_start_poses=noisy_start_poses, start_poses=start_poses, rl=rl)
        return returns

    def pre_forward(self, instruction_string, object_database):
        # Make sure all pre-trained modules are loaded
        self.load_pref_pretrained_modules()

        # If deaf, empty out the instruction and replace it with an empty string
        if self.deaf:
            instruction_string = ""

        noun_chunks = self.object_reference_tagger.tag_chunks_only(instruction_string)
        chunk_affinity_scores = self.language_conditioned_segmentation.chunk_affinity_scores(noun_chunks, object_database)
        anonymized_instruction, object_references, object_reference_embeddings = self.object_reference_tagger.filter_chunks(instruction_string, noun_chunks, chunk_affinity_scores)

        anon_instr_tokenized = tokenize_instruction(anonymized_instruction, word2token=self.word2token)
        obj_ref_indices = [i for i, t in enumerate(anon_instr_tokenized) if t == self.obj_ref_tok_idx]

        # Pre-compute text similarity matrix
        text_similarity_matrix, add_to_database = self.language_conditioned_segmentation.pre_forward(
            object_references, object_database)

        # Add pre-computed stuff to the data batch
        # TODO: make sure dimensions work out correctly
        add_to_batch = {}
        add_to_batch["anon_instr_ref_tok_indices"] = [obj_ref_indices]
        add_to_batch["anon_instr_nl"] = [anonymized_instruction]
        add_to_batch["anon_instr"] = [anon_instr_tokenized]
        add_to_batch["noun_chunks"] = [noun_chunks]
        add_to_batch["object_references"] = [object_references]
        add_to_batch["text_similarity_matrix"] = [text_similarity_matrix]
        add_to_batch["object_database"] = [add_to_database]

        return add_to_batch

    def compute_observability_masks_for_stage2(self, images, states):
        cam_poses = self.cam_poses_from_states(states)
        traj_len = len(cam_poses)
        seg_masks_w, obs_masks_w = self.project_c_to_w(images, cam_poses, self.model_state.tensor_store, show="")
        if self.params.get("cover_init_pos", False):
            StartMasks_R = self.add_init_pos_to_coverage.get_init_pos_masks(obs_masks_w.shape[0], obs_masks_w.device)
            StartMasks_W, _ = self.map_transform_r_to_w(StartMasks_R, cam_poses, None)
            obs_masks_w = self.add_init_pos_to_coverage(obs_masks_w, StartMasks_W)

        blank_model_state = GenericModelState()
        _, accum_obs_masks_w = self.map_accumulator_w(blank_model_state, seg_masks_w, obs_masks_w,
                                                                      add_indicator=[True] * traj_len,
                                                                      reset_indicator=[True] + [False] * ((traj_len) - 1))
        return accum_obs_masks_w

    def post_forward(self, images, states, anon_instruction, anon_instr_ref_tok_indices, object_database,
                     text_similarity_matrix, reset_mask, noisy_start_poses=None, start_poses=None, rl=False):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param anon_instruction: 1xT tensor of encoded anonymized instructions
        :param anon_instr_ref_tok_indices:
        :param object_database: {"object_images": MxQx3xHxW tensor, "object_references": list (M) of lists (Q) of strings}
        :param noisy_start_poses: list of noisy start poses (for data-augmentation). These define the path-prediction frame at training time
        :param start_poses: list of drone start poses (these should be equal in practice)
        :param rl: boolean indicating if we're doing reinforcement learning. If yes, output more than the visitation distribution
        :return:
        """
        #print("......................................................................................................")
        #print("   START FORWARD")
        # We don't use instructions here - instead use the anonymized instructions
        self.load_pref_pretrained_modules()
        self.load_pretrained_modules()

        cam_poses = self.cam_poses_from_states(states)
        g_poses = None # None pose is a placeholder for the canonical global pose.
        self.prof.tick("out")

        # If blind, zero-out the input images
        if self.blind:
            images = torch.zeros_like(images)

        batch_size = 1
        traj_len = images.shape[0]
        num_obj_ref = len(anon_instr_ref_tok_indices)

        self.model_state.tensor_store.keep_inputs("fpv", images)

        # --------------------------------------------------------------------------------------------
        # Instruction processing
        # --------------------------------------------------------------------------------------------

        # Calculate reference context embeddings
        anon_instruction_token_emb = self.token_embedding(anon_instruction)
        anon_sequence_output, (_, _) = self.anon_instruction_lstm(anon_instruction_token_emb)
        anon_embeddings = anon_sequence_output
        obj_ref_context_embeddings = anon_embeddings[:, anon_instr_ref_tok_indices, :]
        self.model_state.put("obj_ref_context_embeddings", obj_ref_context_embeddings)

        # Take the average of all token embeddings as the overall embedding
        full_embedding = torch.mean(anon_embeddings, dim=1)

        assert_shape(obj_ref_context_embeddings, [1, num_obj_ref, anon_sequence_output.shape[-1]],
                                                  "obj_ref_context_embeddings")

        self.prof.tick("text embed")
        # --------------------------------------------------------------------------------------------
        # VISION: For every object reference, obtain a first-person segmentation mask
        # --------------------------------------------------------------------------------------------
        seg_masks_fpv = self.language_conditioned_segmentation.post_forward(self.model_state, images, object_database, text_similarity_matrix)
        self.prof.tick("language_conditioned_segmentation")

        seg_masks_w, obs_masks_w = self.project_c_to_w(seg_masks_fpv, cam_poses, self.model_state.tensor_store, show="")

        self.model_state.tensor_store.keep_inputs("seg_masks_w", seg_masks_w)
        self.model_state.tensor_store.keep_inputs("M_w", obs_masks_w)
        self.prof.tick("   vision - projection")

        # --------------------------------------------------------------------------------------------
        # Accumulate segmentation masks over time
        # --------------------------------------------------------------------------------------------

        #print("        ACCUMULATION")
        # Consider the space very near the drone as observable - it is too hard to explore to consider it unobservable.
        if self.params.get("cover_init_pos", False):
            StartMasks_R = self.add_init_pos_to_coverage.get_init_pos_masks(obs_masks_w.shape[0], obs_masks_w.device)
            StartMasks_W, _ = self.map_transform_r_to_w(StartMasks_R, cam_poses, None)
            obs_masks_w = self.add_init_pos_to_coverage(obs_masks_w, StartMasks_W)

        accum_seg_masks_w, accum_obs_masks_w = self.map_accumulator_w(self.model_state, seg_masks_w, obs_masks_w, add_indicator=[True] * traj_len, reset_indicator=reset_mask)

        w_poses = g_poses
        self.model_state.tensor_store.keep_inputs("accum_obs_masks_w", accum_obs_masks_w)
        self.model_state.tensor_store.keep_inputs("accum_seg_masks_w", accum_seg_masks_w)
        self.prof.tick("map_accumulate")

        # Create a figure where the drone is drawn on the map
        if self.params["write_figures"]:
            self.model_state.tensor_store.keep_inputs("drone_poses", draw_drone_poses(cam_poses))

        # --------------------------------------------------------------------------------------------
        # GROUNDING: Build the grounding map by multiplying each object ref mask by its reference context embedding
        # --------------------------------------------------------------------------------------------
        context_grounding_map_w = self.context_grounding_map(accum_seg_masks_w, obj_ref_context_embeddings, self.model_state)
        self.prof.tick("grounding map")
        self.model_state.tensor_store.keep_inputs("context_grounding_map_w", context_grounding_map_w)

        # --------------------------------------------------------------------------------------------
        # MAP ASSEMBLY - EXTRA LAYERS
        # --------------------------------------------------------------------------------------------
        # Add layers for:
        #  * Coverage Mask
        #  * All object mask
        #  * Boundary mask
        boundary_mask_w = self._get_boundary_mask(context_grounding_map_w)
        full_map_w = torch.cat([context_grounding_map_w, accum_obs_masks_w, boundary_mask_w], dim=1)
        self.model_state.tensor_store.keep_inputs("full_map_w", full_map_w)

        s_poses = start_poses if self.params["predict_in_start_frame"] else cam_poses
        full_map_s, _ = self.map_transform_w_to_s(full_map_w, w_poses, s_poses)
        accum_M_s, _ = self.map_transform_w_to_s(accum_obs_masks_w, w_poses, s_poses)

        self.model_state.tensor_store.keep_inputs("full_map_s", full_map_s)
        self.model_state.tensor_store.keep_inputs("accum_obs_masks_s", accum_M_s)
        self.prof.tick("transform_w_to_s")

        # --------------------------------------------------------------------------------------------
        # Perturb maps for data augmentation
        # --------------------------------------------------------------------------------------------
        # Data augmentation for trajectory prediction
        map_poses_clean_select = None
        # TODO: Figure out if we can just swap out start poses for noisy poses and get rid of separate noisy poses
        if self.do_perturb_maps:
            assert noisy_start_poses is not None, "Noisy poses must be provided if we're perturbing maps"
            full_map_perturbed, perturbed_poses = self.map_transform_s_to_p(full_map_s, s_poses, noisy_start_poses)
            accum_M_p, _ = self.map_transform_s_to_p(accum_M_s, s_poses, noisy_start_poses)
        else:
            full_map_perturbed, perturbed_poses = full_map_s, s_poses
            accum_M_p = accum_M_s

        self.model_state.tensor_store.keep_inputs("full_map_perturbed", full_map_perturbed)
        self.model_state.tensor_store.keep_inputs("accum_obs_masks_p", accum_M_p)
        self.prof.tick("map_perturb")

        # --------------------------------------------------------------------------------------------
        # Visitation prediction
        # --------------------------------------------------------------------------------------------
        # ---------
        log_v_dist_p, v_dist_p_poses = self.path_predictor_lingunet(full_map_perturbed, full_embedding, perturbed_poses, tensor_store=self.model_state.tensor_store)
        # ---------
        self.print_tensor_stats("log_v_dist_p_inner", log_v_dist_p.inner_distribution)

        self.prof.tick("pathpred")

        # --------------------------------------------------------------------------------------------
        # Transform outputs
        # --------------------------------------------------------------------------------------------

        # Transform distributions back to world reference frame and keep them (these are the model outputs)
        both_inner_w, v_dist_w_poses_select = self.map_transform_p_to_w(log_v_dist_p.inner_distribution, v_dist_p_poses, None)
        log_v_dist_w = Partial2DDistribution(both_inner_w, log_v_dist_p.outer_prob_mass)
        self.model_state.tensor_store.keep_inputs("log_v_dist_w", log_v_dist_w)
        self.print_tensor_stats("both_inner_w", both_inner_w)

        # Transform distributions back to start reference frame and keep them (for auxiliary objective)
        both_inner_s, v_dist_s_poses = self.map_transform_p_to_r(log_v_dist_p.inner_distribution, v_dist_p_poses, start_poses)
        log_v_dist_s = Partial2DDistribution(both_inner_s, log_v_dist_p.outer_prob_mass)
        self.model_state.tensor_store.keep_inputs("log_v_dist_s", log_v_dist_s)

        lsfm = SpatialSoftmax2d()
        # prime number will mean that it will alternate between sim and real
        if self.get_iter() % 23 == 0 and not rl:
            for i in range(context_grounding_map_w.shape[0]):
                Presenter().show_image(full_map_w.detach().cpu()[i,0:3], f"{self.domain}_full_map_w", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_s.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_s", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_p.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_p", scale=4, waitkey=1)
                Presenter().show_image(full_map_perturbed.detach().cpu()[i,0:3], f"{self.domain}_full_map_perturbed", scale=4, waitkey=1)
                break  # Uncomment this to loop over entire trajectory

        self.prof.tick("transform_back")

        return_list = [log_v_dist_w, w_poses]
        if rl:
            internals_for_rl = {"map_coverage_w": accum_obs_masks_w, "map_uncoverage_w": 1 - accum_obs_masks_w}
            return_list.append(internals_for_rl)

        #print("     END FORWARD")
        return tuple(return_list)

    def subsample_inputs(self, batch):
        #keep_prob = self.params["step_keep_prob"]
        keep_steps = batch["plan_mask"][0]
        batch["images"] = batch["images"][:, keep_steps]
        batch["states"] = batch["states"][:, keep_steps]
        batch["actions"] = batch["actions"][:, keep_steps]
        batch["stops"] = batch["stops"][:, keep_steps]
        batch["masks"] = batch["masks"][:, keep_steps]
        # batch["md"] - skip, I only ever use the first md anyway
        batch["start_poses"] = [batch["start_poses"][0][keep_steps]]
        batch["noisy_poses"] = [batch["noisy_poses"][0][keep_steps]]
        # The following is already subsampled in dataset
        # batch["traj_ground_truth"] = batch["traj_ground_truth"][:, keep_steps]
        batch["plan_mask"] = [True] * len(keep_steps) #batch["plan_mask"][keep_steps]]
        batch["firstseg_mask"] = None # This can't be supported together with subsampling
        return batch

    def unbatch(self, batch):
        # Inputs
        images = self.to_model_device(batch["images"][0])
        seq_len = len(images)
        instructions = self.to_model_device(batch["instr"][0][:seq_len])
        instr_lengths = batch["instr_len"][0][:seq_len]
        states = self.to_model_device(batch["states"][0])
        anon_instruction = batch["anon_instr"][0]
        anon_instr_ref_tok_indices = batch["anon_instr_ref_tok_indices"][0]
        object_database = batch["object_database"][0]
        object_database["object_images"] = self.to_model_device(object_database["object_images"])
        object_database["object_vectors"] = self.to_model_device(object_database["object_vectors"])
        text_similarity_matrix = self.to_model_device(batch["text_similarity_matrix"][0])
        anon_instruction_t = torch.tensor(anon_instruction)[np.newaxis, :].to(images.device)
        # This line works for producing a set of vectors
        # self.anon_instruction_bert(torch.tensor(bert_tokenize_instruction("head towards things"))[np.newaxis, :].to(images.device))
        goal_pos_map_m = batch["goal_loc"][0]              # Goal location in the world in meters_and_metrics
        # TODO: We're taking the FIRST label here. SINGLE SEGMENT ASSUMPTION
        start_poses = batch["start_poses"][0]
        noisy_start_poses = get_noisy_poses_torch(start_poses.numpy(),
                                                  self.params["pos_variance"], self.params["rot_variance"],
                                                  cuda=False, cuda_device=None)

        # Ground truth visitation distributions (in start and global frames)
        v_dist_w_ground_truth = self.to_model_device(batch["traj_ground_truth"][0])
        v_dist_s_ground_truth, poses_s = self.map_transform_w_to_s(v_dist_w_ground_truth, None, start_poses)
        goal_pos_map_px = torch.from_numpy(transformations.pos_m_to_px(goal_pos_map_m.numpy(),
                                           self.params["global_map_size"],
                                           self.params["world_size_m"],
                                           self.params["world_size_px"]))

        goal_pos_map_px = self.to_model_device(goal_pos_map_px)
        metadata = batch["md"][0][0]
        env_id = metadata["env_id"]

        self.model_state.tensor_store.keep_inputs("v_dist_s_ground_truth", v_dist_s_ground_truth)
        self.model_state.tensor_store.keep_inputs("goal_pos_map", goal_pos_map_px)
        self.model_state.tensor_store.set_flag("env_id", env_id)

        return images, states, instructions, instr_lengths, start_poses, noisy_start_poses, metadata, \
               anon_instruction_t, anon_instr_ref_tok_indices, object_database, text_similarity_matrix

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")
        self.model_state = GenericModelState()

        # TODO: Revise this nonsense
        if isinstance(batch, list):
           batch = batch[0]

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.model_state.tensor_store

        # During training, subsample / dropout observations so that we have to process less data
        # This MIGHT build robustness against laggy observations at test time too
        if self.params["plan_every_n_steps"] > 1:
            batch = self.subsample_inputs(batch)

        images, states, instructions, instr_len, start_poses, noisy_start_poses, metadata, \
            anon_instruction_t, anon_instr_ref_tok_indices, object_database, text_similarity_matrix = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        traj_len = images.shape[0]
        reset_mask = [True] + [False] * (traj_len - 1)

        # ----------------------------------------------------------------------------
        _ = self.post_forward(images, states, anon_instruction_t, anon_instr_ref_tok_indices,
                              object_database, text_similarity_matrix, reset_mask,
                              noisy_start_poses=start_poses if eval else noisy_start_poses,
                              start_poses=start_poses)
        # ----------------------------------------------------------------------------

        # TODO: Rever this maybe
        if not self.nowriter:#self.get_iter() % 10 == 0 and not self.nowriter:
            if eval:
                name = f'{batch["md"][0][0]["env_id"]}-{batch["md"][0][0]["seg_idx"]}'
            else:
                name = str(self.get_iter())
            viz.visualize_model_dashboard_during_training(batch, self.model_state.tensor_store, name, self.run_name)
            #Presenter().show_image(viz_img, "model_dashboard", waitkey=1)

        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")

        # Calculate goal-prediction accuracy:
        # TODO: Abstract this stuff better:
        goal_pos = self.model_state.tensor_store.get_inputs_batch("goal_pos_map", cat_not_stack=True)
        success_goal = self.goal_good_criterion(
            self.model_state.tensor_store.get_inputs_batch("log_v_dist_w", cat_not_stack=True),
            goal_pos
        )
        acc = 1.0 if success_goal else 0.0
        self.goal_acc_meter.put(acc)
        goal_visible = self.goal_visible(self.model_state.tensor_store.get_inputs_batch("M_w", cat_not_stack=True), goal_pos)
        if goal_visible:
            self.visible_goal_acc_meter.put(acc)
        else:
            self.invisible_goal_acc_meter.put(acc)
        self.visible_goal_frac_meter.put(1.0 if goal_visible else 0.0)

        self.correct_goals += acc
        self.total_goals += 1

        self.prof.tick("goal_acc")

        losses, metrics = self.losses.calculate_aux_loss(tensor_store=self.model_state.tensor_store, reduce_average=True, disable_losses=disable_losses)
        loss = self.losses.combine_losses(losses, self.loss_weights)

        prefix = f"{self.model_name}_{'eval' if eval else 'train'}"
        iteration = self.get_iter()

        self.prof.tick("calc_losses")

        if self.params.get("use_visitation_discriminator") and not eval:
            v_dist_s_ground_truth = self.model_state.tensor_store.get_inputs_batch("v_dist_s_ground_truth")[:, 0]
            accum_obs_masks_s = self.model_state.tensor_store.get_inputs_batch("accum_obs_masks_s")[:, 0]
            v_dist_s_ground_truth_f = Partial2DDistribution.from_distribution_and_mask(v_dist_s_ground_truth, accum_obs_masks_s[:, 0])
            log_v_dist_s = self.model_state.tensor_store.get_inputs_batch("log_v_dist_s")[0]
            v_dist_s = log_v_dist_s.softmax()
            discriminator_loss = self.distributed_visitation_discriminator.calc_domain_loss(v_dist_s, v_dist_s_ground_truth_f, eval)
            # No need to fool the discriminator if it is already fooled:
            discriminator_loss = torch.relu(discriminator_loss)
            # TODO: Parametrize this stuff:
            loss += 0.2 * discriminator_loss
            self.writer.add_scalar(prefix + "/discriminator_loss", discriminator_loss.item(), iteration)
            self.prof.tick("distribution_loss")

        self.writer.add_scalar(prefix + "/iteration", iteration, iteration)

        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_dict(prefix, losses, iteration)
        self.writer.add_dict(prefix, metrics, iteration)

        self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/visible_goal_accuracy", self.visible_goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/invisible_goal_accuracy", self.invisible_goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/visible_goal_fraction", self.visible_goal_frac_meter.get(), iteration)

        num_object_refs = text_similarity_matrix.shape[0]
        self.writer.add_scalar(prefix + "/num_object_refs", num_object_refs, iteration)
        has_object_refs = 1 if num_object_refs > 0 else 0
        self.writer.add_scalar(prefix + "/has_object_refs", has_object_refs, iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.model_state.tensor_store

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

        real_drone = get_current_parameters()["Setup"]["real_drone"]
        d = get_current_parameters()["Data"]
        object_database_name = d["object_database_real"] if real_drone else d["object_database_sim"]

        if not eval:
            index_by_env = get_current_parameters()["Data"].get("index_supervised_data_by_env", False)
        else:
            index_by_env = False

        other_self = RPN_FewShot_Stage1_Bidomain(run_name=self.run_name, domain=self.domain, nowriter=True)
        return NavAroundNovelObjectsDataset(other_self,
                                            object_database_name,
                                            query_img_side_length=32,
                                            data=data,
                                            env_list=envs,
                                            domain=domain,
                                            dataset_names=dataset_names,
                                            dataset_prefix=dataset_prefix,
                                            aux_provider_names=data_sources,
                                            index_by_env=index_by_env,
                                            segment_level=True)

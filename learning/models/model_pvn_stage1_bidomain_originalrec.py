import torch
import torch.nn as nn
from torch.autograd import Variable

from data_io.model_io import load_pytorch_model
from data_io import env
from data_io.paths import get_logging_dir
from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.common import cuda_var
from learning.inputs.pose import Pose, get_noisy_poses_torch
from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from learning.modules.auxiliaries.path_auxiliary import PathAuxiliary2D
from learning.modules.auxiliaries.class_auxiliary import ClassAuxiliary
from learning.modules.auxiliaries.feature_reg_auxiliary import FeatureRegularizationAuxiliary2D
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.modules.img_to_map.fpv_to_global_map import FPVToGlobalMap
from learning.modules.map_to_map.ratio_path_predictor import RatioPathPredictor
from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegratorGlobalMap
from learning.modules.map_to_map.lang_filter_map_to_map import LangFilterMapProcessor
from learning.modules.map_to_map.map_batch_fill_missing import MapBatchFillMissing
from learning.modules.map_to_map.map_batch_select import MapBatchSelect
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.map_transformer import MapTransformer
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.visitation_softmax import VisitationSoftmax
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.modules.add_drone_pos_to_coverage_mask import AddDroneInitPosToCoverage

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import get_current_meters

from utils.dict_tools import dict_merge
from utils.dummy_summary_writer import DummySummaryWriter

from parameters.parameter_server import get_current_parameters
import transformations
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class PVN_Stage1_Bidomain_Original(nn.Module):

    def __init__(self, run_name="", domain="sim"):

        super(PVN_Stage1_Bidomain_Original, self).__init__()
        self.model_name = "pvn_stage1"
        self.run_name = run_name
        self.domain = domain
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")
        #self.writer = DummySummaryWriter()

        self.root_params = get_current_parameters()["ModelPVN"]
        self.params = self.root_params["Stage1"]
        self.use_aux = self.root_params["UseAux"]
        self.aux_weights = self.root_params["AuxWeights"]

        if self.params.get("weight_override"):
            aux_weights_override_name = "AuxWeightsRealOverride" if self.domain == "real" else "AuxWeightsSimOverride"
            aux_weights_override = self.root_params.get(aux_weights_override_name)
            if aux_weights_override:
                print(f"Overriding auxiliary weights for domain: {self.domain}")
                self.aux_weights = dict_merge(self.aux_weights, aux_weights_override)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.tensor_store = KeyTensorStore()
        self.losses = AuxiliaryLosses()

        # Auxiliary Objectives
        self.do_perturb_maps = self.params["perturb_maps"]
        print("Perturbing maps: ", self.do_perturb_maps)

        # Path-pred FPV model definition
        # --------------------------------------------------------------------------------------------------------------

        self.num_feature_channels = self.params["feature_channels"]# + params["relevance_channels"]
        self.num_map_channels = self.params["pathpred_in_channels"]

        self.img_to_features_w = FPVToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"], world_size_m=self.params["world_size_m"],
            res_channels=self.params["resnet_channels"], map_channels=self.params["feature_channels"],
            img_w=self.params["img_w"], img_h=self.params["img_h"], cam_h_fov=self.params["cam_h_fov"],
            domain=domain,
            img_dbg=IMG_DBG)

        self.map_accumulator_w = LeakyIntegratorGlobalMap(
            source_map_size=self.params["global_map_size"],
            world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"])

        self.add_init_pos_to_coverage = AddDroneInitPosToCoverage(
            world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"],
            map_size_px=self.params["local_map_size"])

        # Pre-process the accumulated map to do language grounding if necessary - in the world reference frame
        self.map_processor_grounding = LangFilterMapProcessor(
                embed_size=self.params["emb_size"],
                in_channels=self.params["feature_channels"],
                out_channels=self.params["relevance_channels"],
                spatial=False, cat_out=False)

        ratio_prior_channels = self.params["feature_channels"]

        # Process the global accumulated map
        self.path_predictor_lingunet = RatioPathPredictor(
            self.params["lingunet"],
            prior_channels_in=self.params["feature_channels"],
            posterior_channels_in=self.params["pathpred_in_channels"],
            dual_head=self.params["predict_confidence"],
            compute_prior=self.params["compute_prior"],
            use_prior=self.params["use_prior_only"],
            oob=self.params["clip_observability"])

        print("UNet Channels: " + str(self.num_map_channels))
        print("Feature Channels: " + str(self.num_feature_channels))

        # TODO:O Verify that config has the same randomization parameters (yaw, pos, etc)
        self.second_transform = self.do_perturb_maps or self.params["predict_in_start_frame"]

        # Sentence Embedding
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"], self.params["emb_dropout"])

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

        # Batch select is used to drop and forget semantic maps at those timestaps that we're not planning in
        self.batch_select = MapBatchSelect()
        # Since we only have path predictions for some timesteps (the ones not dropped above), we use this to fill
        # in the missing pieces by reorienting the past trajectory prediction into the frame of the current timestep
        self.map_batch_fill_missing = MapBatchFillMissing(self.params["local_map_size"], self.params["world_size_px"], self.params["world_size_m"])

        self.spatialsoftmax = SpatialSoftmax2d()
        self.visitation_softmax = VisitationSoftmax()

        #TODO:O Use CroppedMapToActionTriplet in Wrapper as Stage2
        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------

        # We add all auxiliaries that are necessary. The first argument is the auxiliary name, followed by parameters,
        # followed by variable number of names of inputs. ModuleWithAuxiliaries will automatically collect these inputs
        # that have been saved with keep_auxiliary_input() during execution
        if self.use_aux["class_features"]:
            self.losses.add_auxiliary(ClassAuxiliary2D("class_features", self.params["feature_channels"], self.params["num_landmarks"], 0,
                                                "fpv_features", "lm_pos_fpv", "lm_indices"))
        if self.use_aux["grounding_features"]:
            self.losses.add_auxiliary(ClassAuxiliary2D("grounding_features", self.params["relevance_channels"], 2, 0,
                                                "fpv_features_g", "lm_pos_fpv", "lm_mentioned"))
        if self.use_aux["class_map"]:
            self.losses.add_auxiliary(ClassAuxiliary2D("class_map", self.params["feature_channels"], self.params["num_landmarks"], 0,
                                                "S_W_select", "lm_pos_map_select", "lm_indices_select"))
        if self.use_aux["grounding_map"]:
            self.losses.add_auxiliary(ClassAuxiliary2D("grounding_map", self.params["relevance_channels"], 2, 0,
                                                "R_W_select", "lm_pos_map_select", "lm_mentioned_select"))
        # CoRL model uses alignment-model groundings
        if self.use_aux["lang"]:
            # one output for each landmark, 2 classes per output. This is for finetuning, so use the embedding that's gonna be fine tuned
            self.losses.add_auxiliary(ClassAuxiliary("lang", self.params["emb_size"], 2, self.params["num_landmarks"],
                                                "sentence_embed", "lang_lm_mentioned"))

        if self.use_aux["regularize_map"]:
            self.losses.add_auxiliary(FeatureRegularizationAuxiliary2D("regularize_map", "l1", "S_W_select"))

        lossfunc = self.params["path_loss_function"]
        if self.params["clip_observability"]:
            self.losses.add_auxiliary(PathAuxiliary2D("visitation_dist", lossfunc, self.params["clip_observability"],
                                                  "log_v_dist_s_select", "v_dist_s_ground_truth_select", "SM_S_select"))
        else:
            self.losses.add_auxiliary(PathAuxiliary2D("visitation_dist", lossfunc, self.params["clip_observability"],
                                                  "log_v_dist_s_select", "v_dist_s_ground_truth_select", "SM_S_select"))

        self.goal_good_criterion = GoalPredictionGoodCriterion(ok_distance=self.params["world_size_px"]*0.1)
        self.goal_acc_meter = MovingAverageMeter(10)
        self.visible_goal_acc_meter = MovingAverageMeter(10)
        self.invisible_goal_acc_meter = MovingAverageMeter(10)
        self.visible_goal_frac_meter = MovingAverageMeter(10)

        self.losses.print_auxiliary_info()

        self.total_goals = 0
        self.correct_goals = 0

        self.env_id = None
        self.env_img = None
        self.seg_idx = None
        self.prev_instruction = None
        self.seq_step = 0

        self.should_save_path_overlays = False

    def make_picklable(self):
        self.writer = DummySummaryWriter()

    def steal_cross_domain_modules(self, other_self):
        self.iter = other_self.iter
        self.losses = other_self.losses
        self.sentence_embedding = other_self.sentence_embedding
        self.map_accumulator_w = other_self.map_accumulator_w
        self.map_processor_grounding = other_self.map_processor_grounding
        self.path_predictor_lingunet = other_self.path_predictor_lingunet
        #self.img_to_features_w = other_self.img_to_features_w

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

    def load_state_dict(self, state_dict, strict=True):
        super(PVN_Stage1_Bidomain_Original, self).load_state_dict(state_dict, strict)

    def init_weights(self):
        self.img_to_features_w.init_weights()
        self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()
        self.map_processor_grounding.init_weights()
        self.path_predictor_lingunet.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        self.tensor_store.reset()
        self.sentence_embedding.reset()
        self.img_to_features_w.reset()
        self.map_accumulator_w.reset()
        self.map_batch_fill_missing.reset()
        self.prev_instruction = None

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]
        self.env_img = env.load_env_img(self.env_id, 256, 256)
        self.env_img = self.env_img[:, :, [2,1,0]]

    def set_save_path_overlays(self, save_path_overlays):
        self.should_save_path_overlays = save_path_overlays

    #TODO:O Figure out what to do with save_ground_truth_overlays

    def print_metrics(self):
        print(f"Model {self.model_name}:{self.domain} metrics:")
        print(f"   Goal accuracy: {float(self.correct_goals) / self.total_goals}")

    def goal_visible(self, masks, goal_pos):
        goal_mask = masks.detach()[0, 0, :, :]
        goal_pos = goal_pos[0].long().detach()
        visible = bool((goal_mask[goal_pos[0], goal_pos[1]] > 0.5).detach().cpu().item())
        return visible

    # This is called before beginning an execution sequence
    def start_sequence(self):
        self.seq_step = 0
        self.reset()
        return

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def forward(self, images, states, instructions, instr_lengths,
                plan=None, noisy_start_poses=None, start_poses=None, firstseg=None, select_only=True, halfway=False, grad_noise=False, rl=False):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param instructions: BxM LongTensor where M is the maximum length of any instruction
        :param instr_lengths: list of len B of integers, indicating length of each instruction
        :param plan: list of B booleans indicating True for timesteps where we do planning and False otherwise
        :param noisy_start_poses: list of noisy start poses (for data-augmentation). These define the path-prediction frame at training time
        :param start_poses: list of drone start poses (these should be equal in practice)
        :param firstseg: list of booleans indicating True if a new segment starts at that timestep
        :param select_only: boolean indicating whether to only compute visitation distributions for planning timesteps (default True)
        :param rl: boolean indicating if we're doing reinforcement learning. If yes, output more than the visitation distribution
        :return:
        """
        cam_poses = self.cam_poses_from_states(states)
        g_poses = None # None pose is a placeholder for the canonical global pose.
        self.prof.tick("out")

        self.tensor_store.keep_inputs("fpv", images)

        # Calculate the instruction embedding
        if instructions is not None:
            # TODO: Take batch of instructions and their lengths, return batch of embeddings. Store the last one as internal state
            # TODO: There's an assumption here that there's only a single instruction in the batch and it doesn't change
            # UNCOMMENT THE BELOW LINE TO REVERT BACK TO GENERAL CASE OF SEPARATE INSTRUCTION PER STEP
            if self.params["ignore_instruction"]:
                # If we're ignoring instructions, just feed in an instruction that consists of a single zero-token
                sent_embeddings = self.sentence_embedding(torch.zeros_like(instructions[0:1,0:1]), torch.ones_like(instr_lengths[0:1]))
            else:
                sent_embeddings = self.sentence_embedding(instructions[0:1], instr_lengths[0:1])
            self.tensor_store.keep_inputs("sentence_embed", sent_embeddings)
        else:
            sent_embeddings = self.sentence_embedding.get()

        self.prof.tick("embed")

        # Extract and project features onto the egocentric frame for each image
        F_W, M_W = self.img_to_features_w(images, cam_poses, sent_embeddings, self.tensor_store, show="", halfway=halfway)

        # For training the critic, this is as far as we need to poceed with the computation.
        # self.img_to_features_w has stored computed feature maps inside the tensor store, which will then be retrieved by the critic
        if halfway == True: # Warning: halfway must be True not truthy
            return None, None

        self.tensor_store.keep_inputs("F_w", F_W)
        self.tensor_store.keep_inputs("M_w", M_W)
        self.prof.tick("img_to_map_frame")

        # Accumulate the egocentric features in a global map
        reset_mask = firstseg if self.params["clear_history"] else None

        # Consider the space very near the drone and right under it as observed - draw ones on the observability mask
        # If we treat that space as unobserved, then there's going to be a gap in the visitation distribution, which
        # makes training with RL more difficult, as there is no reward feedback if the drone doesn't cross that gap.
        if self.params.get("cover_init_pos", False):
            StartMasks_R = self.add_init_pos_to_coverage.get_init_pos_masks(M_W.shape[0], M_W.device)
            StartMasks_W, _ = self.map_transform_r_to_w(StartMasks_R, cam_poses, None)
            M_W = self.add_init_pos_to_coverage(M_W, StartMasks_W)

        S_W, SM_W = self.map_accumulator_w(F_W, M_W, reset_mask=reset_mask, show="acc" if IMG_DBG else "")
        S_W_poses = g_poses
        self.prof.tick("map_accumulate")

        # If we're training Stage 2 with imitation learning from ground truth visitation distributions, we want to
        # compute observability masks with the same code that's used in Stage 1 to avoid mistakes.
        if halfway == "observability":
            map_uncoverage_w = 1 - SM_W
            return map_uncoverage_w

        # Throw away those timesteps that don't correspond to planning timesteps
        S_W_select, SM_W_select, S_W_poses_select, cam_poses_select, noisy_start_poses_select, start_poses_select, sent_embeddings_select = \
            self.batch_select(S_W, SM_W, S_W_poses, cam_poses, noisy_start_poses, start_poses, sent_embeddings, plan)

        #maps_m_prior_select, maps_m_posterior_select = None, None

        # Only process the maps on plannieng timesteps
        if len(S_W_select) == 0:
            return None

        self.tensor_store.keep_inputs("S_W_select", S_W_select)
        self.prof.tick("batch_select")

        # Process the map via the two map_procesors
        # Do grounding of objects in the map chosen to do so
        if self.use_aux["grounding_map"]:
            R_W_select, RS_W_poses_select = self.map_processor_grounding(S_W_select, sent_embeddings_select, S_W_poses_select, show="")
            self.tensor_store.keep_inputs("R_W_select", R_W_select)
            self.prof.tick("map_proc_gnd")
            # Concatenate grounding map and semantic map along channel dimension
            RS_W_select = torch.cat([S_W_select, R_W_select], 1)

        else:
            RS_W_select = S_W_select
            RS_W_poses_select = S_W_poses_select

        s_poses_select = start_poses_select if self.params["predict_in_start_frame"] else cam_poses_select
        RS_S_select, RS_S_poses_select = self.map_transform_w_to_s(RS_W_select, RS_W_poses_select, s_poses_select)
        SM_S_select, SM_S_poses_select = self.map_transform_w_to_s(SM_W_select, S_W_poses_select, s_poses_select)

        assert SM_S_poses_select == RS_S_poses_select, "Masks and maps should have the same pose in start frame"

        self.tensor_store.keep_inputs("RS_S_select", RS_S_select)
        self.tensor_store.keep_inputs("SM_S_select", SM_S_select)
        self.prof.tick("transform_w_to_s")

        # Data augmentation for trajectory prediction
        map_poses_clean_select = None
        # TODO: Figure out if we can just swap out start poses for noisy poses and get rid of separate noisy poses
        if self.do_perturb_maps:
            assert noisy_start_poses_select is not None, "Noisy poses must be provided if we're perturbing maps"
            RS_P_select, RS_P_poses_select = self.map_transform_s_to_p(RS_S_select, RS_S_poses_select, noisy_start_poses_select)
        else:
            RS_P_select, RS_P_poses_select = RS_S_select, RS_S_poses_select

        self.tensor_store.keep_inputs("RS_perturbed_select", RS_P_select)
        self.prof.tick("map_perturb")

        sent_embeddings_pp = sent_embeddings_select

        # Run lingunet on the map to predict visitation distribution scores (pre-softmax)
        # ---------
        log_v_dist_p_select, v_dist_p_poses_select = self.path_predictor_lingunet(RS_P_select, sent_embeddings_pp, RS_P_poses_select, tensor_store=self.tensor_store)
        # ---------

        self.prof.tick("pathpred")

        # TODO: Shouldn't we be transforming probability distributions instead of scores? Otherwise OOB space will have weird values
        # Transform distributions back to world reference frame and keep them (these are the model outputs)
        both_inner_w, v_dist_w_poses_select = self.map_transform_p_to_w(log_v_dist_p_select.inner_distribution, v_dist_p_poses_select, None)
        log_v_dist_w_select = Partial2DDistribution(both_inner_w, log_v_dist_p_select.outer_prob_mass)
        self.tensor_store.keep_inputs("log_v_dist_w_select", log_v_dist_w_select)

        # Transform distributions back to start reference frame and keep them (for auxiliary objective)
        both_inner_s, v_dist_s_poses_select = self.map_transform_p_to_r(log_v_dist_p_select.inner_distribution, v_dist_p_poses_select, start_poses_select)
        log_v_dist_s_select = Partial2DDistribution(both_inner_s, log_v_dist_p_select.outer_prob_mass)
        self.tensor_store.keep_inputs("log_v_dist_s_select", log_v_dist_s_select)

        # prime number will mean that it will alternate between sim and real
        if self.get_iter() % 23 == 0:
            lsfm = SpatialSoftmax2d()
            for i in range(S_W_select.shape[0]):
                Presenter().show_image(S_W_select.detach().cpu()[i,0:3], f"{self.domain}_s_w_select", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_s_select.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_s_select", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_p_select.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_p_select", scale=4, waitkey=1)
                Presenter().show_image(RS_P_select.detach().cpu()[i,0:3], f"{self.domain}_rs_p_select", scale=4, waitkey=1)
                break

        self.prof.tick("transform_back")

        # If we're predicting the trajectory only on some timesteps, then for each timestep k, use the map from
        # timestep k if predicting on timestep k. otherwise use the map from timestep j - the last timestep
        # that had a trajectory prediction, rotated in the frame of timestep k.
        if select_only:
            # If we're just pre-training the trajectory prediction, don't waste time on generating the missing maps
            log_v_dist_w = log_v_dist_w_select
            v_dist_w_poses = v_dist_w_poses_select
        else:
            raise NotImplementedError("select_only must be True")

        return_list = [log_v_dist_w, v_dist_w_poses]
        if rl:
            internals_for_rl = {"map_coverage_w": SM_W, "map_uncoverage_w": 1 - SM_W}
            return_list.append(internals_for_rl)

        return tuple(return_list)

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch, halfway=False):
        # Inputs
        images = self.maybe_cuda(batch["images"][0])
        seq_len = len(images)
        instructions = self.maybe_cuda(batch["instr"][0][:seq_len])
        instr_lengths = batch["instr_len"][0][:seq_len]
        states = self.maybe_cuda(batch["states"][0])

        if not halfway:

            plan_mask = batch["plan_mask"][0]  # True for every timestep that we do visitation prediction
            firstseg_mask = batch["firstseg_mask"][0]  # True for every timestep that is a new instruction segment

            # Labels (including for auxiliary losses)
            lm_pos_fpv = batch["lm_pos_fpv"][0]                # All object 2D coordinates in the first-person image
            lm_pos_map_m = batch["lm_pos_map"][0]              # All object 2D coordinates in the semantic map
            lm_indices = batch["lm_indices"][0]                # All object class indices
            goal_pos_map_m = batch["goal_loc"][0]              # Goal location in the world in meters_and_metrics
            lm_mentioned = batch["lm_mentioned"][0]            # 1/0 labels whether object was mentioned/not mentioned in template instruction
            # TODO: We're taking the FIRST label here. SINGLE SEGMENT ASSUMPTION
            lang_lm_mentioned = batch["lang_lm_mentioned"][0][0]  # integer labes as to which object was mentioned
            start_poses = batch["start_poses"][0]
            noisy_start_poses = get_noisy_poses_torch(start_poses.numpy(),
                                                      self.params["pos_variance"], self.params["rot_variance"],
                                                      cuda=False, cuda_device=None)

            # Ground truth visitation distributions (in start and global frames)
            v_dist_w_ground_truth_select = self.maybe_cuda(batch["traj_ground_truth"][0])
            start_poses_select = self.batch_select.one(start_poses, plan_mask, v_dist_w_ground_truth_select.device)
            v_dist_s_ground_truth_select, poses_s = self.map_transform_w_to_s(v_dist_w_ground_truth_select, None, start_poses_select)
            #self.tensor_store.keep_inputs("v_dist_w_ground_truth_select", v_dist_w_ground_truth_select)
            self.tensor_store.keep_inputs("v_dist_s_ground_truth_select", v_dist_s_ground_truth_select)
            #Presenter().show_image(v_dist_s_ground_truth_select.detach().cpu()[0,0], "v_dist_s_ground_truth_select", waitkey=1, scale=4)
            #Presenter().show_image(v_dist_w_ground_truth_select.detach().cpu()[0,0], "v_dist_w_ground_truth_select", waitkey=1, scale=4)

            lm_pos_map_px = [torch.from_numpy(transformations.pos_m_to_px(p.numpy(),
                                                       self.params["global_map_size"],
                                                       self.params["world_size_m"],
                                                       self.params["world_size_px"]))
                            if p is not None else None for p in lm_pos_map_m]
            goal_pos_map_px = torch.from_numpy(transformations.pos_m_to_px(goal_pos_map_m.numpy(),
                                                               self.params["global_map_size"],
                                                               self.params["world_size_m"],
                                                               self.params["world_size_px"]))

            resnet_factor = self.img_to_features_w.img_to_features.get_downscale_factor()
            lm_pos_fpv = [self.cuda_var((s / resnet_factor).long()) if s is not None else None for s in lm_pos_fpv]
            lm_indices = [self.cuda_var(s) if s is not None else None for s in lm_indices]
            lm_mentioned = [self.cuda_var(s) if s is not None else None for s in lm_mentioned]
            lang_lm_mentioned = self.cuda_var(lang_lm_mentioned)
            lm_pos_map_px = [self.cuda_var(s.long()) if s is not None else None for s in lm_pos_map_px]
            goal_pos_map_px = self.cuda_var(goal_pos_map_px)

            self.tensor_store.keep_inputs("lm_pos_fpv", lm_pos_fpv)
            self.tensor_store.keep_inputs("lm_pos_map", lm_pos_map_px)
            self.tensor_store.keep_inputs("lm_indices", lm_indices)
            self.tensor_store.keep_inputs("lm_mentioned", lm_mentioned)
            self.tensor_store.keep_inputs("lang_lm_mentioned", lang_lm_mentioned)
            self.tensor_store.keep_inputs("goal_pos_map", goal_pos_map_px)

            lm_pos_map_select = [lm_pos for i,lm_pos in enumerate(lm_pos_map_px) if plan_mask[i]]
            lm_indices_select = [lm_idx for i,lm_idx in enumerate(lm_indices) if plan_mask[i]]
            lm_mentioned_select = [lm_m for i,lm_m in enumerate(lm_mentioned) if plan_mask[i]]
            goal_pos_map_select = [pos for i,pos in enumerate(goal_pos_map_px) if plan_mask[i]]

            self.tensor_store.keep_inputs("lm_pos_map_select", lm_pos_map_select)
            self.tensor_store.keep_inputs("lm_indices_select", lm_indices_select)
            self.tensor_store.keep_inputs("lm_mentioned_select", lm_mentioned_select)
            self.tensor_store.keep_inputs("goal_pos_map_select", goal_pos_map_select)

        # We won't need this extra information
        else:
            noisy_poses, start_poses, noisy_start_poses = None, None, None
            plan_mask, firstseg_mask = None, None

        metadata = batch["md"][0][0]
        env_id = metadata["env_id"]
        self.tensor_store.set_flag("env_id", env_id)

        return images, states, instructions, instr_lengths, plan_mask, firstseg_mask, start_poses, noisy_start_poses, metadata

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")
        self.reset()

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        images, states, instructions, instr_len, plan_mask, firstseg_mask, \
         start_poses, noisy_start_poses, metadata = self.unbatch(batch, halfway=halfway)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        _ = self(images, states, instructions, instr_len,
                                        plan=plan_mask, firstseg=firstseg_mask,
                                        noisy_start_poses=start_poses if eval else noisy_start_poses,
                                        start_poses=start_poses,
                                        select_only=True,
                                        halfway=halfway,
                                        grad_noise=grad_noise)
        # ----------------------------------------------------------------------------

        if self.should_save_path_overlays:
            self.save_path_overlays(metadata)

        # If we run the model halfway, we only need to calculate features needed for the wasserstein loss
        # If we want to include more features in wasserstein critic, have to run the forward pass a bit further
        if halfway and not halfway == "v2":
            return None, self.tensor_store

        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")

        if not halfway:
            # Calculate goal-prediction accuracy:
            goal_pos = self.tensor_store.get_inputs_batch("goal_pos_map", cat_not_stack=True)
            success_goal = self.goal_good_criterion(
                self.tensor_store.get_inputs_batch("log_v_dist_w_select", cat_not_stack=True),
                goal_pos
            )
            acc = 1.0 if success_goal else 0.0
            self.goal_acc_meter.put(acc)
            goal_visible = self.goal_visible(self.tensor_store.get_inputs_batch("M_w", cat_not_stack=True), goal_pos)
            if goal_visible:
                self.visible_goal_acc_meter.put(acc)
            else:
                self.invisible_goal_acc_meter.put(acc)
            self.visible_goal_frac_meter.put(1.0 if goal_visible else 0.0)

            self.correct_goals += acc
            self.total_goals += 1

            self.prof.tick("goal_acc")

        if halfway == "v2":
            disable_losses = ["visitation_dist", "lang"]

        losses, metrics = self.losses.calculate_aux_loss(tensor_store=self.tensor_store, reduce_average=True, disable_losses=disable_losses)
        loss = self.losses.combine_losses(losses, self.aux_weights)

        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_dict(prefix, losses, iteration)
        self.writer.add_dict(prefix, metrics, iteration)

        if not halfway:
            self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), iteration)
            self.writer.add_scalar(prefix + "/visible_goal_accuracy", self.visible_goal_acc_meter.get(), iteration)
            self.writer.add_scalar(prefix + "/invisible_goal_accuracy", self.invisible_goal_acc_meter.get(), iteration)
            self.writer.add_scalar(prefix + "/visible_goal_fraction", self.visible_goal_frac_meter.get(), iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

    def get_dataset(self, data=None, envs=None, domain=None, dataset_names=None, dataset_prefix=None, eval=False, halfway_only=False):
        # TODO: Maybe use eval here
        data_sources = []
        # If we're running auxiliary objectives, we need to include the data sources for the auxiliary labels
        #if self.use_aux_class_features or self.use_aux_class_on_map or self.use_aux_grounding_features or self.use_aux_grounding_on_map:
        #if self.use_aux_goal_on_map:
        if not halfway_only:
            data_sources.append(aup.PROVIDER_LM_POS_DATA)
            data_sources.append(aup.PROVIDER_GOAL_POS)

            # Adding these in this order will compute poses with added noise and compute trajectory ground truth
            # in the reference frame of these noisy poses
            data_sources.append(aup.PROVIDER_START_POSES)

            if self.do_perturb_maps:
                print("PERTURBING MAPS!")
                # TODO: The noisy poses from the provider are not actually used!! Those should replace states instead!
                data_sources.append(aup.PROVIDER_NOISY_POSES)
                # TODO: Think this through. Perhaps we actually want dynamic ground truth given a noisy start position
                if self.params["predict_in_start_frame"]:
                    data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
                else:
                    data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY)
            else:
                print("NOT Perturbing Maps!")
                data_sources.append(aup.PROVIDER_NOISY_POSES)
                if self.params["predict_in_start_frame"]:
                    data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC)
                else:
                    data_sources.append(aup.PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC)

            data_sources.append(aup.PROVIDER_LANDMARKS_MENTIONED)

            templates = get_current_parameters()["Environment"]["templates"]
            if templates:
                data_sources.append(aup.PROVIDER_LANG_TEMPLATE)

        return SegmentDataset(data=data, env_list=envs, domain=domain, dataset_names=dataset_names, dataset_prefix=dataset_prefix, aux_provider_names=data_sources, segment_level=True)

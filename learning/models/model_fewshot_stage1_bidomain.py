import torch
from data_io import spacy_singleton
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.autograd import Variable

from bert import bert_tools
from data_io.model_io import load_pytorch_model
from data_io import env
from data_io import parsing
from data_io.paths import get_logging_dir
from bert.bert_tools import OBJ_REF_TOK
from data_io.instructions import get_all_instructions, get_word_to_token_map
from data_io.tokenization import bert_tokenize_instruction, bert_untokenize_instruction
from data_io.instructions import tokenize_instruction
from learning.datasets.nav_around_novel_objects_dataset import NavAroundNovelObjectsDataset
from learning.datasets.object_reference_dataset import ObjectReferenceDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.pose import Pose, get_noisy_poses_torch
from learning.modules.auxiliaries.path_auxiliary import PathAuxiliary2D
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.modules.img_to_map.fpv_to_global_map import FPVToGlobalMap
from learning.modules.img_to_map.project_to_global_map import ProjectToGlobalMap
from learning.modules.map_to_map.ratio_path_predictor import RatioPathPredictor
from learning.modules.map_to_map.leaky_integrator_w import LeakyIntegrator
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from learning.modules.map_transformer import MapTransformer
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.visitation_softmax import VisitationSoftmax
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.modules.add_drone_pos_to_coverage_mask import AddDroneInitPosToCoverage

from learning.models.model_object_reference_recognizer import ModelObjectReferenceRecognizer
from learning.models.model_few_shot_instance_recognizer import ModelFewShotInstanceRecognizer

import learning.models.visualization.viz_html_fewshot_stage1_bidomain as viz

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.utils import draw_drone_poses

from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import get_current_meters

from utils.dict_tools import dict_merge
from utils.dummy_summary_writer import DummySummaryWriter

from transformers import BertConfig, BertModel, BertTokenizer

from parameters.parameter_server import get_current_parameters
import transformations
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

IMG_POOL_FACTOR = 4


class FewShot_Stage1_Bidomain(nn.Module):

    def __init__(self, run_name="", domain="sim", nowriter=False):
        super(FewShot_Stage1_Bidomain, self).__init__()
        self.model_name = "fspvn_stage1"
        self.run_name = run_name
        self.domain = domain
        if nowriter:
            self.writer = DummySummaryWriter()
        else:
            self.writer = LoggingSummaryWriter.make_singleton(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")

        self.root_params = get_current_parameters()["ModelFSPVN"]
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
        self.nlp = spacy_singleton.load("en_core_web_lg")
        self.ord = ObjectReferenceDataset()

        """
        bert_params = self.params["anon_bert"]
        anon_bert_config = BertConfig(
            vocab_size_or_config_json_file=bert_params["vocab_size"],
            hidden_size=bert_params["hidden_size"],
            num_hidden_layers=bert_params["num_hidden_layers"],
            num_attention_heads=bert_params["num_attention_heads"],
            intermediate_size=bert_params["intermediate_size"],
            hidden_act="gelu",
            hidden_dropout_prob=bert_params["hidden_dropout_prob"],
            attention_probs_dropout_prob=bert_params["attention_probs_dropout_prob"],
            max_position_embeddings=bert_params["max_position_embeddings"],
            type_vocab_size=bert_params["type_vocab_size"],
            initializer_range=bert_params["initializer_range"],
            layer_norm_eps=bert_params["layer_norm_eps"]
        )
        self.anon_instruction_bert = BertModel(anon_bert_config)
        """
        self.anon_instruction_emb = nn.Embedding(num_embeddings=2200, embedding_dim=32)
        self.anon_instruction_lstm = nn.LSTM(input_size=32,
                                             hidden_size=64,
                                             num_layers=2,
                                             bias=True,
                                             batch_first=True,
                                             bidirectional=True)

        # TODO: Finish implementation
        self.loaded_pretrained_modules = False

        # This is wrapped in a list so that it never gets moved to GPU.
        self.wrapped_object_reference_recognizer = [ModelObjectReferenceRecognizer(run_name + "--obj_ref_rec", domain)]
        self.wrapped_instance_recognizer = [ModelFewShotInstanceRecognizer(run_name + "--instance_rec", domain)]
        # Switch the models to eval mode, to disable dropout and such
        self.wrapped_object_reference_recognizer[0].eval()
        self.wrapped_instance_recognizer[0].eval()

        self.img_to_features_w = FPVToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"], world_size_m=self.params["world_size_m"],
            res_channels=self.params["resnet_channels"], map_channels=self.params["feature_channels"],
            img_w=self.params["img_w"], img_h=self.params["img_h"], cam_h_fov=self.params["cam_h_fov"],
            domain=domain,
            img_dbg=IMG_DBG)

        self.project_c_to_w = ProjectToGlobalMap(
            source_map_size=self.params["global_map_size"], world_size_px=self.params["world_size_px"], world_size_m=self.params["world_size_m"],
            img_w=self.params["img_w"] / IMG_POOL_FACTOR, img_h=self.params["img_h"] / IMG_POOL_FACTOR, cam_h_fov=self.params["cam_h_fov"],
            domain=domain, img_dbg=IMG_DBG
        )

        self.map_accumulator_w = LeakyIntegrator(
            source_map_size=self.params["global_map_size"],
            world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"])

        self.add_init_pos_to_coverage = AddDroneInitPosToCoverage(
            world_size_px=self.params["world_size_px"],
            world_size_m=self.params["world_size_m"],
            map_size_px=self.params["local_map_size"])

        # Process the global accumulated map
        self.path_predictor_lingunet = RatioPathPredictor(
            self.params["lingunet"],
            prior_channels_in=self.params["feature_channels"],
            posterior_channels_in=self.params["pathpred_in_channels"],
            dual_head=self.params["predict_confidence"],
            compute_prior=self.params["compute_prior"],
            use_prior=self.params["use_prior_only"],
            oob=self.params["clip_observability"])

        self.second_transform = self.do_perturb_maps or self.params["predict_in_start_frame"]

        # Sentence Embedding
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"], self.params["emb_dropout"])

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
                                                  "log_v_dist_s", "v_dist_s_ground_truth", "accum_M_s"))

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
        iter = int(self.iter.data[0])
        #print("GetIter: ", iter)
        return iter

    def inc_iter(self):
        self.iter += 1

    def load_state_dict(self, state_dict, strict=True):
        super(FewShot_Stage1_Bidomain, self).load_state_dict(state_dict, strict)
        # Load pre-trained models
        self.load_pretrained_modules()

    def load_pretrained_modules(self):
        if not self.loaded_pretrained_modules:
            obj_recognizer_model_name = self.root_params.get("object_recognizer_model_name")
            print(f"Loading object reference recognizer: {obj_recognizer_model_name}")
            load_pytorch_model(self.wrapped_object_reference_recognizer[0], obj_recognizer_model_name)

            few_shot_recognizer_model_name = self.root_params.get("few_shot_recognizer_model_name")
            print(f"Loading few shot object recognizer: {few_shot_recognizer_model_name}")
            load_pytorch_model(self.wrapped_instance_recognizer[0], few_shot_recognizer_model_name)

            self.loaded_pretrained_modules = True

    def init_weights(self):
        self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()
        self.path_predictor_lingunet.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        self.tensor_store.reset()
        self.sentence_embedding.reset()
        self.map_accumulator_w.reset()
        self.prev_instruction = None

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]
        self.env_img = env.load_env_img(self.env_id, 256, 256)
        self.env_img = self.env_img[:, :, [2, 1, 0]]

    def set_save_path_overlays(self, save_path_overlays):
        self.should_save_path_overlays = save_path_overlays

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

    def extract_object_references_and_contexts(self, instruction_strings_batch):
        batch_anonymized_instrucions = []
        batch_object_references = []
        batch_reference_embeddings = []
        batch_chunks = []
        for instruction_string in instruction_strings_batch:
            instruction_doc = self.nlp(instruction_string)
            chunk_strings, chunk_contexts, chunk_indices = parsing.get_noun_chunks_and_contexts_from_spacy_doc(
                instruction_string, instruction_doc)

            if len(chunk_strings) > 0:
                chunk_embeddings = [self.ord._vectorize(s) for s in chunk_strings]
                #chunk_embeddings = [self.nlp(s).vector for s in chunk_strings]
                chunk_embeddings_t = torch.stack([torch.tensor(c) for c in chunk_embeddings], dim=0)
                chunk_scores = self.wrapped_object_reference_recognizer[0](chunk_embeddings_t)
                chunk_classes = self.wrapped_object_reference_recognizer[0].threshold(chunk_scores)
                # Filter out object references - noun chunks with predicted class 0 (class 1 is for spurious chunks)
                # It's important to convert to .item first! Otherwise it doesn't work in one of the conda envs.
                object_ref_strings_indices_classes = list(filter(lambda m: m[2][0].item() < 0.5, zip(chunk_strings, chunk_indices, chunk_classes, chunk_embeddings_t)))
                object_references = [o[0] for o in object_ref_strings_indices_classes]
                object_reference_indices = [o[1] for o in object_ref_strings_indices_classes]
                object_reference_embeddings = [o[3] for o in object_ref_strings_indices_classes]
                anonymized_instruction, anon_ref_indices = parsing.anonymize_objects(instruction_string, object_reference_indices)
            else:
                anonymized_instruction = instruction_string
                object_references = []
                object_reference_embeddings = []

            batch_anonymized_instrucions.append(anonymized_instruction)
            batch_object_references.append(object_references)
            batch_reference_embeddings.append(object_reference_embeddings)
            batch_chunks.append(chunk_strings)
        return batch_anonymized_instrucions, batch_object_references, batch_reference_embeddings, batch_chunks

    def compute_object_reference_similarity_matrices(self, b_obj_refs, b_obj_ref_embeddings, b_db_strings, b_db_string_embeddings):
        """
        :param b_obj_refs: List (over batch) of lists (over chunks) of strings
        :param b_obj_ref_embeddings: List (over batch) of NxD tensors, where N is number of object refs
        :param b_db_strings: List (over batch) of lists (over objects) of lists (for each object) of strings
        :return:
        """
        b_similarity_matrices = []
        # TODO: Get rid of raw strings here, only leave embeddings
        for object_references, obj_ref_embeddings, db_strings, db_string_embeddings in zip(
                b_obj_refs, b_obj_ref_embeddings, b_db_strings, b_db_string_embeddings):

            num_refs = len(object_references)
            num_objs = len(db_strings)
            similarity_mat = torch.zeros((num_refs, num_objs))

            # TODO: We currently assume that every object has the same number of descriptions.
            #  in future, we might want to change that here.
            if len(obj_ref_embeddings) > 0:
                obj_ref_embedding_mat = torch.stack(obj_ref_embeddings)
                db_string_embedding_mat = torch.stack([torch.stack(query_embeddings, dim=0)
                                                       for query_embeddings in db_string_embeddings])

                # Use kernel density estimation in the embedding space to assign probabilities to different clusters
                # Compute probability density of an object reference given each object in the database.
                cluster_variance = 0.2
                # For the following, axes are ordered as: objrefs in instruction, objects in db, queries per obj, dims
                a = obj_ref_embedding_mat[:, np.newaxis, np.newaxis, :]
                covariance_mat = torch.eye(300) * cluster_variance
                mu = db_string_embedding_mat[np.newaxis, :, :, :]

                # Multivatiage Gaussian kernel density
                # x1 = (x - mu) * \Sigma^-1
                x1 = torch.tensordot(a - mu, covariance_mat.inverse(), dims=((3,), (0,)))
                # x2 = (x - mu) * \Sigma^-1 * (x - mu)^T
                x2 = (x1 * (a - mu)).sum(3)
                densities_per_ref_per_image = torch.exp(-0.5 * x2)

                densities_per_ref_per_object = densities_per_ref_per_image.sum(2)
                prob_of_object_given_ref = densities_per_ref_per_object / (
                        densities_per_ref_per_object.sum(1, keepdim=True) + 1e-10)
                similarity_mat = prob_of_object_given_ref.detach()

            b_similarity_matrices.append(similarity_mat)

        return b_similarity_matrices

    def embed_db_strings(self, b_db_strings):
        b_db_string_embeddings = []
        for db_strings in b_db_strings:
            db_string_embeddings = []
            for obj_strings in db_strings:
                obj_vectors = [torch.from_numpy(self.nlp(s).vector) for s in obj_strings]
                db_string_embeddings.append(obj_vectors)
            b_db_string_embeddings.append(db_string_embeddings)
        return b_db_string_embeddings

    def print_tensor_stats(self, name, tensor):
        if tensor.contiguous().view([-1]).shape[0] > 0:
            prefix = f"{self.model_name}_stats/{name}/"
            iteration = self.get_iter()
            self.writer.add_scalar(prefix + "min", tensor.min().item(), iteration)
            self.writer.add_scalar(prefix + "max", tensor.max().item(), iteration)
            self.writer.add_scalar(prefix + "mean", tensor.mean().item(), iteration)
            self.writer.add_scalar(prefix + "std", tensor.std().item(), iteration)
            #print(name, "min:", tensor.min().item(), "max:", tensor.max().item(), "mean:", tensor.mean().item(), "std:", tensor.std().item())
        #else:
        #    print(name, "empty")

    def preprocess_batch(self, batch):
        """
        :param batch:
        :return:
        """
        # Unpack
        string_instructions_nl = [b[0] for b in batch["instr_nl"]]
        db_strings_batch = [nod["object_references"] for nod in batch["object_database"]]
        add_to_batch = self.pre_forward(string_instructions_nl, db_strings_batch)
        batch = dict_merge(batch, add_to_batch)
        return batch

    def pre_forward(self, instruction_strings_batch, db_strings_batch):
        """
        The part of forward function that is not differentiable and should run inside the dataset to be parallelized.
        :param instruction_strings_batch: List if instruction strings
        :param db_strings_batch: List (over batch) of lists (over objects) of lists (for each object) of strings
        :return:
        """
        # Make sure all pre-trained modules are loaded
        self.load_pretrained_modules()

        anonymized_instructions_batch, object_references_batch, object_reference_embeddings_batch, chunks_batch = (
            self.extract_object_references_and_contexts(instruction_strings_batch))
        # TODO: Finish this implementation by computing the string similarity matrix
        db_string_embeddings_batch = self.embed_db_strings(db_strings_batch)
        similarity_matrix_batch = self.compute_object_reference_similarity_matrices(
            object_references_batch, object_reference_embeddings_batch, db_strings_batch, db_string_embeddings_batch)

        if (similarity_matrix_batch[0] != similarity_matrix_batch[0]).any():
            raise ValueError("NaN in similarity matrix")

        anon_instr_batch = [tokenize_instruction(i, word2token=self.word2token) for i in anonymized_instructions_batch]
        obj_ref_indices_batch = [[i for i, t in enumerate(anon_instr) if t==self.obj_ref_tok_idx] for anon_instr in anon_instr_batch]
        # TODO: Uncomment for BERT
        # anon_instr_and_tok_indices_batch = [bert_tokenize_instruction(i, return_special_token_indices=True)
        #                                    for i in anonymized_instructions_batch]
        #anon_instr_batch = [x[0] for x in anon_instr_and_tok_indices_batch]
        #obj_ref_indices_batch = [x[1] for x in anon_instr_and_tok_indices_batch]

        outputs = {
            "anon_instr_nl": anonymized_instructions_batch,
            "anon_instr": anon_instr_batch,
            "anon_instr_ref_tok_indices": obj_ref_indices_batch,
            "obj_ref": object_references_batch,
            "noun_chunks": chunks_batch,
            "similarity_matrix": similarity_matrix_batch
        }
        return outputs

    def forward(self, images, states, instructions, instr_lengths,
                anon_instruction, anon_instr_ref_tok_indices, object_database, similarity_matrix,
                noisy_start_poses=None, start_poses=None, rl=False, noshow=False):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param instructions: BxM LongTensor where M is the maximum length of any instruction
        :param instr_lengths: list of len B of integers, indicating length of each instruction
        :param anon_instruction: 1xT tensor of encoded anonymized instruction
        :param anon_instr_ref_tok_indices:
        :param object_database: {"object_images": MxQx3xHxW tensor, "object_references": list (M) of lists (Q) of strings}
        :param similarity_matrix: NxM matrix, where each row is a probability distribution over object_database
        :param noisy_start_poses: list of noisy start poses (for data-augmentation). These define the path-prediction frame at training time
        :param start_poses: list of drone start poses (these should be equal in practice)
        :param rl: boolean indicating if we're doing reinforcement learning. If yes, output more than the visitation distribution
        :return:
        """
        #print("......................................................................................................")
        cam_poses = self.cam_poses_from_states(states)
        g_poses = None # None pose is a placeholder for the canonical global pose.
        self.prof.tick("out")

        batch_size = 1
        traj_len = images.shape[0]

        self.tensor_store.keep_inputs("fpv", images)

        # --------------------------------------------------------------------------------------------
        # Instruction processing
        # --------------------------------------------------------------------------------------------
        # Calculate the full instruction embedding
        # TODO: Move towards a more advanced embedding model
        if self.params["ignore_instruction"]:
            # If we're ignoring instructions, just feed in an instruction that consists of a single zero-token
            sent_embeddings = self.sentence_embedding(torch.zeros_like(instructions[0:1, 0:1]), torch.ones_like(torch.tensor(instr_lengths[0:1])))
        else:
            sent_embeddings = self.sentence_embedding(instructions[0:1], instr_lengths[0:1])
        #sent_embeddings = self.sentence_embedding(instructions, instr_lengths)
        self.tensor_store.keep_inputs("sentence_embed", sent_embeddings)
        self.prof.tick("embed")
        self.print_tensor_stats("sent_embeddings", sent_embeddings)

        # Calculate reference context embeddings
        # TODO: Bug is here - deviceassert if obj_ref token is present
        anon_instruction_emb = self.anon_instruction_emb(anon_instruction)
        self.print_tensor_stats("anon_instruction_emb", anon_instruction_emb)
        sequence_output, (_, _) = self.anon_instruction_lstm(anon_instruction_emb)
        #sequence_output, pooled_output = self.anon_instruction_bert(anon_instruction)
        anon_embeddings = sequence_output
        obj_ref_context_embeddings = anon_embeddings[:, anon_instr_ref_tok_indices, :]
        self.print_tensor_stats("anon_embeddings", anon_embeddings)

        # --------------------------------------------------------------------------------------------
        # For every object in the novel object dataset, predict a segmentation mask in the environment
        # and project this mask to the world reference frame
        # --------------------------------------------------------------------------------------------
        # TODO: Prune novel object dataset to exclude objects that are unlikely to have been mentioned
        # images is Tx3xHxW, query_images: MxQx3xHxW
        query_images = object_database["object_images"]
        num_objects = query_images.shape[0]
        num_queries = query_images.shape[1]
        query_h = query_images.shape[3]
        query_w = query_images.shape[4]
        # turn images into MxTx3xHxW and query_images into MxTxQx3xHxW
        scene_images = images[:, np.newaxis, :, :, :]
        query_images = query_images[np.newaxis, :, :, :, :, :]
        scene_images = scene_images.repeat(1, num_objects, 1, 1, 1)
        query_images = query_images.repeat(traj_len, 1, 1, 1, 1, 1)
        # finally turn images into B*Tx3xHxW and query_images into B*TxQx3xHxW
        scene_images = scene_images.view(traj_len * num_objects, 3, images.shape[2], images.shape[3])
        query_images = query_images.view(traj_len * num_objects, num_queries, 3, query_h, query_w)
        self.print_tensor_stats("query_images", query_images)
        self.print_tensor_stats("scene_images", scene_images)

        self.wrapped_instance_recognizer[0].to(images.device)
        logit_seg_masks = self.wrapped_instance_recognizer[0](query_images, scene_images).detach()
        self.print_tensor_stats("logit_seg_masks", logit_seg_masks.inner_distribution)
        seg_masks = logit_seg_masks.softmax()
        self.print_tensor_stats("seg_masks", seg_masks.inner_distribution)
        h, w = seg_masks.inner_distribution.shape[2], seg_masks.inner_distribution.shape[3]
        inner_seg_masks = seg_masks.inner_distribution.view([traj_len, num_objects, h, w])
        outer_seg_masks = seg_masks.outer_prob_mass.view([traj_len, num_objects])
        inner_seg_masks_small = F.avg_pool2d(inner_seg_masks, IMG_POOL_FACTOR)
        self.print_tensor_stats("inner_seg_masks_small", inner_seg_masks_small)

        self.tensor_store.keep_inputs("obj_masks_fpv_inner", inner_seg_masks)
        self.tensor_store.keep_inputs("obj_masks_fpv_outer", outer_seg_masks)

        # Project segmentation masks to the global reference frame
        obj_masks_w, M_w = self.project_c_to_w(inner_seg_masks_small, cam_poses, self.tensor_store, show="")

        # Extract and project features onto the egocentric frame for each image
        #F_W, FM_W = self.img_to_features_w(images, cam_poses, sent_embeddings, self.tensor_store, show="")

        self.tensor_store.keep_inputs("obj_masks_w", obj_masks_w)
        self.tensor_store.keep_inputs("M_w", M_w)
        self.prof.tick("img_to_map_frame")

        # --------------------------------------------------------------------------------------------
        # Accumulate segmentation masks over time
        # --------------------------------------------------------------------------------------------

        # Consider the space very near the drone as observable - it is too hard to explore to consider it unobservable.
        if self.params.get("cover_init_pos", False):
            StartMasks_R = self.add_init_pos_to_coverage.get_init_pos_masks(M_w.shape[0], M_w.device)
            StartMasks_W, _ = self.map_transform_r_to_w(StartMasks_R, cam_poses, None)
            M_w = self.add_init_pos_to_coverage(M_w, StartMasks_W)

        accum_obj_masks_w, accum_M_w = self.map_accumulator_w(obj_masks_w, M_w, add_mask=[True] * traj_len, reset_mask=[True] + [False] * (traj_len - 1),
                                           show="acc" if IMG_DBG else "")
        #semantic_map_w, accum_SM_M_w = self.map_accumulator_w(F_W, FM_W, add_mask=[True] * traj_len, reset_mask=[True] + [False] * (traj_len - 1),
        #                                   show="acc" if IMG_DBG else "")
        self.print_tensor_stats("accum_obj_masks_w", accum_obj_masks_w)

        w_poses = g_poses
        self.tensor_store.keep_inputs("accum_M_w", accum_M_w)
        self.tensor_store.keep_inputs("accum_obj_masks_w", accum_obj_masks_w)
        self.prof.tick("map_accumulate")

        # Create a figure where the drone is drawn on the map
        if self.params["write_figures"]:
            self.tensor_store.keep_inputs("drone_poses", draw_drone_poses(cam_poses))

        # --------------------------------------------------------------------------------------------
        # Build the object grounding map by averaging object masks according to their probabilities
        # --------------------------------------------------------------------------------------------
        accum_obj_masks_w_plus = accum_obj_masks_w[:, np.newaxis, :, :, :]  # Insert object reference axis
        similarity_matrix_plus = similarity_matrix[np.newaxis, :, :, np.newaxis, np.newaxis] # Insert spatiotemporal axes
        weighted_obj_masks_w = accum_obj_masks_w_plus * similarity_matrix_plus # Weigh each novel object mask by it's weight
        object_reference_masks_w = weighted_obj_masks_w.sum(2)  # Sum across novel objects
        self.tensor_store.keep_inputs("object_reference_masks_w", object_reference_masks_w)
        self.print_tensor_stats("object_reference_masks_w", object_reference_masks_w)
        self.print_tensor_stats("similarity_matrix_plus", similarity_matrix_plus)

        # Axes ordering: time x objects x embedding_dim x H x W
        # Insert spatiotemporal axes (time, height, width)
        obj_ref_context_embeddings_plus = obj_ref_context_embeddings[0, np.newaxis, :, :, np.newaxis, np.newaxis]
        # Insert embedding axis (will be the new channel axis)
        object_reference_masks_w_plus = object_reference_masks_w[:, :, np.newaxis, :, :]
        grounding_map_w = object_reference_masks_w_plus * obj_ref_context_embeddings_plus
        # Each location is the average (over objects) of the embedding vectors
        # We can't use mean - the epsilon is needed, because sometimes the 1st axis is empty (if there are no obj refs)
        grounding_map_w = grounding_map_w.sum(1) / (grounding_map_w.shape[1] + 1e-10)
        self.print_tensor_stats("grounding_map_w", grounding_map_w)

        # The full map is a concatenation of grounding map and semantic map, but normalize each first
        grounding_map_w = self.grounding_map_norm(grounding_map_w)
        #semantic_map_w = self.semantic_map_norm(semantic_map_w)
        #full_map_w = torch.cat([grounding_map_w, semantic_map_w], dim=1)
        full_map_w = grounding_map_w
        self.print_tensor_stats("full_map_w", full_map_w)


        s_poses = start_poses if self.params["predict_in_start_frame"] else cam_poses
        full_map_s, _ = self.map_transform_w_to_s(full_map_w, w_poses, s_poses)
        accum_M_s, _ = self.map_transform_w_to_s(accum_M_w, w_poses, s_poses)

        self.tensor_store.keep_inputs("full_map_s", full_map_s)
        self.tensor_store.keep_inputs("accum_M_s", accum_M_s)
        self.prof.tick("transform_w_to_s")
        self.print_tensor_stats("full_map_s", full_map_s)

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

        # TODO: Figure out why this is unused!
        self.tensor_store.keep_inputs("full_map_perturbed", full_map_perturbed)
        self.tensor_store.keep_inputs("accum_M_p", accum_M_p)
        self.prof.tick("map_perturb")

        # --------------------------------------------------------------------------------------------
        # Visitation prediction
        # --------------------------------------------------------------------------------------------
        # ---------
        log_v_dist_p, v_dist_p_poses = self.path_predictor_lingunet(full_map_perturbed, sent_embeddings, perturbed_poses, tensor_store=self.tensor_store)
        # ---------
        self.print_tensor_stats("log_v_dist_p_inner", log_v_dist_p.inner_distribution)

        self.prof.tick("pathpred")

        # --------------------------------------------------------------------------------------------
        # Transform outputs
        # --------------------------------------------------------------------------------------------

        # Transform distributions back to world reference frame and keep them (these are the model outputs)
        both_inner_w, v_dist_w_poses_select = self.map_transform_p_to_w(log_v_dist_p.inner_distribution, v_dist_p_poses, None)
        log_v_dist_w = Partial2DDistribution(both_inner_w, log_v_dist_p.outer_prob_mass)
        self.tensor_store.keep_inputs("log_v_dist_w", log_v_dist_w)
        self.print_tensor_stats("both_inner_w", both_inner_w)

        # Transform distributions back to start reference frame and keep them (for auxiliary objective)
        both_inner_s, v_dist_s_poses = self.map_transform_p_to_r(log_v_dist_p.inner_distribution, v_dist_p_poses, start_poses)
        log_v_dist_s = Partial2DDistribution(both_inner_s, log_v_dist_p.outer_prob_mass)
        self.tensor_store.keep_inputs("log_v_dist_s", log_v_dist_s)

        lsfm = SpatialSoftmax2d()
        # prime number will mean that it will alternate between sim and real
        if self.get_iter() % 23 == 0 and not noshow:
            for i in range(grounding_map_w.shape[0]):
                Presenter().show_image(full_map_w.detach().cpu()[i,0:3], f"{self.domain}_full_map_w", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_s.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_s", scale=4, waitkey=1)
                Presenter().show_image(lsfm(log_v_dist_p.inner_distribution).detach().cpu()[i], f"{self.domain}_v_dist_p", scale=4, waitkey=1)
                Presenter().show_image(full_map_perturbed.detach().cpu()[i,0:3], f"{self.domain}_full_map_perturbed", scale=4, waitkey=1)
                break  # Uncomment this to loop over entire trajectory

        self.prof.tick("transform_back")

        return_list = [log_v_dist_w, w_poses]
        if rl:
            internals_for_rl = {"map_coverage_w": accum_M_w, "map_uncoverage_w": 1 - accum_M_w}
            return_list.append(internals_for_rl)

        return tuple(return_list)

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch):
        # Inputs
        images = self.maybe_cuda(batch["images"][0])
        seq_len = len(images)
        instructions = self.maybe_cuda(batch["instr"][0][:seq_len])
        instr_lengths = batch["instr_len"][0][:seq_len]
        states = self.maybe_cuda(batch["states"][0])

        anon_instruction = batch["anon_instr"][0]
        anon_instr_ref_tok_indices = batch["anon_instr_ref_tok_indices"][0]
        object_database = batch["object_database"][0]
        object_database["object_images"] = self.maybe_cuda(object_database["object_images"])
        similarity_matrix = self.maybe_cuda(batch["similarity_matrix"][0])
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
        v_dist_w_ground_truth = self.maybe_cuda(batch["traj_ground_truth"][0])
        v_dist_s_ground_truth, poses_s = self.map_transform_w_to_s(v_dist_w_ground_truth, None, start_poses)
        self.tensor_store.keep_inputs("v_dist_s_ground_truth", v_dist_s_ground_truth)
        #Presenter().show_image(v_dist_s_ground_truth_select.detach().cpu()[0,0], "v_dist_s_ground_truth_select", waitkey=1, scale=4)
        #Presenter().show_image(v_dist_w_ground_truth_select.detach().cpu()[0,0], "v_dist_w_ground_truth_select", waitkey=1, scale=4)

        goal_pos_map_px = torch.from_numpy(transformations.pos_m_to_px(goal_pos_map_m.numpy(),
                                           self.params["global_map_size"],
                                           self.params["world_size_m"],
                                           self.params["world_size_px"]))

        goal_pos_map_px = self.cuda_var(goal_pos_map_px)
        self.tensor_store.keep_inputs("goal_pos_map", goal_pos_map_px)
        metadata = batch["md"][0][0]
        env_id = metadata["env_id"]
        self.tensor_store.set_flag("env_id", env_id)

        return images, states, instructions, instr_lengths, start_poses, noisy_start_poses, metadata, \
               anon_instruction_t, anon_instr_ref_tok_indices, object_database, similarity_matrix

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")
        self.reset()

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        images, states, instructions, instr_len,  start_poses, noisy_start_poses, metadata, \
            anon_instruction_t, anon_instr_ref_tok_indices, object_database, similarity_matrix = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        _ = self(images, states, instructions, instr_len,
                 anon_instruction_t, anon_instr_ref_tok_indices, object_database, similarity_matrix,
                 noisy_start_poses=start_poses if eval else noisy_start_poses,
                 start_poses=start_poses)
        # ----------------------------------------------------------------------------

        if self.get_iter() % 100 == 0:
            viz_img = viz.visualize_model_dashboard(batch, self.tensor_store, self.get_iter(), self.run_name)
            #Presenter().show_image(viz_img, "model_dashboard", waitkey=1)

        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")

        # Calculate goal-prediction accuracy:
        goal_pos = self.tensor_store.get_inputs_batch("goal_pos_map", cat_not_stack=True)
        success_goal = self.goal_good_criterion(
            self.tensor_store.get_inputs_batch("log_v_dist_w", cat_not_stack=True),
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

        losses, metrics = self.losses.calculate_aux_loss(tensor_store=self.tensor_store, reduce_average=True, disable_losses=disable_losses)
        loss = self.losses.combine_losses(losses, self.aux_weights)

        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_scalar(prefix, iteration, iteration)

        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_dict(prefix, losses, iteration)
        self.writer.add_dict(prefix, metrics, iteration)

        self.writer.add_scalar(prefix + "/goal_accuracy", self.goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/visible_goal_accuracy", self.visible_goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/invisible_goal_accuracy", self.invisible_goal_acc_meter.get(), iteration)
        self.writer.add_scalar(prefix + "/visible_goal_fraction", self.visible_goal_frac_meter.get(), iteration)

        num_object_refs = similarity_matrix.shape[0]
        self.writer.add_scalar(prefix + "/num_object_refs", num_object_refs, iteration)
        has_object_refs = 1 if num_object_refs > 0 else 0
        self.writer.add_scalar(prefix + "/has_object_refs", has_object_refs, iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

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
            object_database_name = get_current_parameters()["Data"]["object_database_train"]
        else:
            object_database_name = get_current_parameters()["Data"]["object_database_dev"]

        other_self = FewShot_Stage1_Bidomain(run_name=self.run_name, domain=self.domain, nowriter=True)
        return NavAroundNovelObjectsDataset(other_self,
                                            object_database_name,
                                            query_img_side_length=32,
                                            data=data,
                                            env_list=envs,
                                            domain=domain,
                                            dataset_names=dataset_names,
                                            dataset_prefix=dataset_prefix,
                                            aux_provider_names=data_sources,
                                            segment_level=True)

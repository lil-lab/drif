import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from data_io.paths import get_logging_dir
from data_io.tokenization import bert_untokenize_instruction
from data_io.instructions import debug_untokenize_instruction
from learning.datasets.object_recognizer_dataset import ObjectRecognizerDataset
import learning.datasets.aux_data_providers as aup

from learning.modules.auxiliaries.cross_entropy_2d import CrossEntropy2DAuxiliary
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses

from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple

from learning.modules.resnet.resnet_13_s import ResNet13S
from learning.modules.unet.lingunet_5_oob import Lingunet5OOB

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter

from parameters.parameter_server import get_current_parameters
import transformations
from visualization import Presenter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class ObjectRecognizer(nn.Module):
    # TODO: Support bi-domain operation (left placeholders for now)

    def __init__(self, run_name="", domain="sim"):

        super(ObjectRecognizer, self).__init__()
        self.model_name = "object_recognizer.json"
        self.run_name = run_name
        self.domain = None
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")

        self.params = get_current_parameters()["ObjectRecognizer"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Path-pred FPV model definition
        # --------------------------------------------------------------------------------------------------------------
        self.sentence_embedding = SentenceEmbeddingSimple(
            self.params["word_emb_size"], self.params["emb_size"], self.params["emb_layers"], self.params["emb_dropout"])

        self.feature_net = ResNet13S(self.params["feature_channels"], down_pad=True)
        self.lingunet = Lingunet5OOB(self.params["lingunet"])
        self.softmax2d = SpatialSoftmax2d()

        # In-case we want to predict on maps instead of images
        # self.img_to_features_w = FPVToGlobalMap(
        #    source_map_size=self.params["global_map_size"],
        #    world_size_px=self.params["world_size_px"],
        #    world_size_m=self.params["world_size_m"],
        #    res_channels=self.params["resnet_channels"], map_channels=self.params["feature_channels"],
        #    img_w=self.params["img_w"], img_h=self.params["img_h"], cam_h_fov=self.params["cam_h_fov"],
        #    domain=domain,
        #    img_dbg=IMG_DBG)

        # self.map_accumulator_w = LeakyIntegratorGlobalMap(
        #    source_map_size=self.params["global_map_size"],
        #    world_size_px=self.params["world_size_px"],
        #    world_size_m=self.params["world_size_m"])

        # self.add_init_pos_to_coverage = AddDroneInitPosToCoverage(
        #    world_size_px=self.params["world_size_px"],
        #    world_size_m=self.params["world_size_m"],
        #    map_size_px=self.params["local_map_size"])

        # Objective function
        # --------------------------------------------------------------------------------------------------------------
        self.tensor_store = KeyTensorStore()
        self.losses = AuxiliaryLosses()
        self.losses.add_auxiliary(CrossEntropy2DAuxiliary("cross_entropy_2d", "pred_obj_masks", "obj_mask_labels"))
        self.aux_weights = {"cross_entropy_2d": 1.0}
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

    def both_domain_parameters(self, other_self):
        # This function iterates and yields parameters from this module and the other module, but does not yield
        # shared parameters twice.
        # First yield all of the other module's parameters
        for p in other_self.parameters():
            yield p
        # Then yield all the parameters from the this module that are not shared with the other one
        # for p in self.img_to_features_w.parameters():
        #    yield p
        return

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def load_state_dict(self, state_dict, strict=True):
        super(ObjectRecognizer, self).load_state_dict(state_dict, strict)

    def init_weights(self):
        #self.img_to_features_w.init_weights()
        #self.map_accumulator_w.init_weights()
        self.sentence_embedding.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        self.tensor_store.reset()
        self.sentence_embedding.reset()
        # self.img_to_features_w.reset()
        # self.map_accumulator_w.reset()
        self.prev_instruction = None

    def print_metrics(self):
        print(f"Model {self.model_name}:{self.domain} metrics:")
        print(f"   Goal accuracy: {float(self.correct_goals) / self.total_goals}")

    def forward(self, images, states, chunks, chunk_lengths):
        """
        :param images: BxCxHxW batch of images (observations)
        :param states: BxK batch of drone states
        :param instructions: BxM LongTensor where M is the maximum length of any instruction
        :param instr_lengths: list of len B of integers, indicating length of each instruction
        :return:
        """
        g_poses = None # None pose is a placeholder for the canonical global pose.
        self.prof.tick("out")

        self.tensor_store.keep_inputs("fpv", images)

        # Calculate the instruction embedding
        # TODO: Take batch of instructions and their lengths, return batch of embeddings. Store the last one as internal state
        # TODO: handle this
        # ASSUMING IT'S THE SAME INSTRUCTION SEGMENT (PREDICT ONE SEGMENT AT A TIME).
        # UNCOMMENT THE BELOW LINE TO REVERT BACK TO GENERAL CASE OF SEPARATE INSTRUCTION PER STEP
        if self.params["ignore_instruction"]:
            # If we're ignoring instructions, just feed in an instruction that consists of a single zero-token
            raise NotImplementedError()
        else:
            chunk_embeddings = self.sentence_embedding(chunks, chunk_lengths)
        self.tensor_store.keep_inputs("chunk_embeddings", chunk_embeddings)

        self.prof.tick("embed")

        fpv_features = self.feature_net(images)
        predicted_masks = self.lingunet(fpv_features, chunk_embeddings)
        self.tensor_store.keep_inputs("pred_obj_masks", predicted_masks)

        # Extract and project features onto the egocentric frame for each image
        # F_W, M_W = self.img_to_features_w(images, cam_poses, sent_embeddings, self.tensor_store, show="", halfway=halfway)
        # if halfway == True and not halfway == "v2":
        #    return None, None
        # self.tensor_store.keep_inputs("F_w", F_W)
        # self.tensor_store.keep_inputs("M_w", M_W)
        # self.prof.tick("img_to_map_frame")
        # S_W, SM_W = self.map_accumulator_w(F_W, M_W, reset_mask=reset_mask, show="acc" if IMG_DBG else "")
        # S_W_poses = g_poses
        # self.tensor_store.keep_inputs("SM_w", SM_W)
        # self.prof.tick("map_accumulate")

        return predicted_masks

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch):
        # Inputs
        images = self.maybe_cuda(batch["images"])
        seq_len = len(images)
        chunks = self.maybe_cuda(batch["tok_chunks"][:seq_len])
        chunk_lengths = batch["chunk_len"][:seq_len]
        states = self.maybe_cuda(batch["states"])

        # Labels (including for auxiliary losses)
        lm_pos_fpv = batch["lm_pos_fpv"]                # All object 2D coordinates in the first-person image
        lm_pos_map_m = batch["lm_pos_map"]              # All object 2D coordinates in the semantic map
        lm_indices = batch["lm_indices"]                # All object class indices
        lm_pos_map_px = [torch.from_numpy(transformations.pos_m_to_px(p.numpy(),
                                                   self.params["global_map_size"],
                                                   self.params["world_size_m"],
                                                   self.params["world_size_px"]))
                        if p is not None else None for p in lm_pos_map_m]

        #resnet_factor = self.img_to_features_w.img_to_features.get_downscale_factor()
        lm_pos_fpv = [self.cuda_var(s.long()) if s is not None else None for s in lm_pos_fpv]
        lm_indices = [self.cuda_var(s) if s is not None else None for s in lm_indices]
        lm_pos_map_px = [self.cuda_var(s.long()) if s is not None else None for s in lm_pos_map_px]
        self.tensor_store.keep_inputs("lm_pos_fpv", lm_pos_fpv)
        self.tensor_store.keep_inputs("lm_pos_map", lm_pos_map_px)
        self.tensor_store.keep_inputs("lm_indices", lm_indices)

        obj_mask_labels = self.maybe_cuda(batch["obj_masks"])
        obj_mask_labels.inner_distribution = F.max_pool2d(obj_mask_labels.inner_distribution, self.feature_net.get_downscale_factor())
        self.tensor_store.keep_inputs("obj_mask_labels", obj_mask_labels)

        metadata = batch["md"][0]
        env_id = metadata["env_id"]
        self.tensor_store.set_flag("env_id", env_id)

        return images, states, chunks, chunk_lengths, metadata

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, viz_only=False):
        self.prof.tick("out")
        self.reset()

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        images, states, chunks, chunk_lengths, metadata = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        masks = self(images, states, chunks, chunk_lengths)
        obj_prob = masks.softmax()
        self.prof.tick("forward")
        # ----------------------------------------------------------------------------

        labels = self.tensor_store.get_inputs_batch("obj_mask_labels")[0]
        if self.get_iter() % 27 == 0 or viz_only:
            i = min(5, images.shape[0] - 1)
            prob_overlay = obj_prob.visualize(idx=i)
            base_image = Presenter().prep_image(images[i])
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
            overlaid_image = Presenter().overlaid_image(base_image, prob_overlay, channel=2)
            labels_overlay = labels.visualize(idx=i)
            overlaid_image = Presenter().overlaid_image(overlaid_image, labels_overlay, channel=0)
            h = overlaid_image.shape[0]
            w = overlaid_image.shape[1]
            overlaid_image = cv2.resize(overlaid_image, (2*w, 2*h))
            chunk_tokens = [c.item() for c in chunks[i, :chunk_lengths[i].item()]]
            chunk_raw = debug_untokenize_instruction(chunk_tokens)
            cv2.putText(overlaid_image, chunk_raw, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 1)
            Presenter().show_image(overlaid_image)
            if viz_only:
                return overlaid_image

        # Calculate in/out accuracy, recall and precision
        pred_out = (obj_prob.outer_prob_mass > 0.5).type(torch.int32)
        labels_out = (labels.outer_prob_mass > 0.5).type(torch.int32)
        correct = (pred_out == labels_out).type(torch.int32)
        correct_pred_out = (pred_out * correct)
        correct_pred_in = ((1 - pred_out) * correct)
        correct_labels_out = (labels_out * correct)
        correct_labels_in = ((1 - labels_out) * correct)

        num_total = len(correct)
        num_pred_out = pred_out.sum()
        num_labels_out = labels_out.sum()
        num_pred_in = num_total - num_pred_out
        num_labels_in = num_total - num_labels_out

        accuracy = correct.sum().float() / num_total
        precision_out = correct_pred_out.sum().float() / (num_pred_out.float() + 1e-9)
        precision_in = correct_pred_in.sum().float() / (num_pred_in.float() + 1e-9)
        recall_out = correct_labels_out.sum().float() / (num_labels_out.float() + 1e-9)
        recall_in = correct_labels_in.sum().float() / (num_labels_in.float() + 1e-9)

        perf_metrics = {
            "in_out_accuracy": accuracy.item(),
            "out_precision": precision_out.item(),
            "in_precision": precision_in.item(),
            "out_recall": recall_out.item(),
            "in_recall": recall_in.item(),
        }

        losses, metrics = self.losses.calculate_aux_loss(tensor_store=self.tensor_store, reduce_average=True)
        loss = self.losses.combine_losses(losses, self.aux_weights)

        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_dict(prefix, losses, iteration)
        self.writer.add_dict(prefix, metrics, iteration)
        self.writer.add_dict(prefix, perf_metrics, iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

    def get_dataset(self, data=None, envs=None, domain=None, dataset_names=None, dataset_prefix=None, eval=False, halfway_only=False):
        # TODO: Maybe use eval here
        data_sources = []
        data_sources.append(aup.PROVIDER_LM_POS_DATA)
        return ObjectRecognizerDataset(env_list=envs,
                                       domain=domain,
                                       dataset_names=dataset_names,
                                       dataset_prefix=dataset_prefix,
                                       aux_provider_names=data_sources,
                                       segment_level=True,
                                       eval=eval)

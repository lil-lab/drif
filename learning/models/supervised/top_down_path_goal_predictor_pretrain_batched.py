import numpy as np
import torch
from learning.modules.sentence_embeddings.sentence_embedding_simple import SentenceEmbeddingSimple
from tensorboardX import SummaryWriter
from torch import nn as nn
from torch.autograd import Variable

from data_io.instructions import debug_untokenize_instruction
from data_io.weights import enable_weight_saving
from learning.datasets.top_down_dataset import TopDownDataset
from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.sequence import sequence_list_to_tensor
from learning.inputs.vision import viz_img
from learning.modules.gather_2d import Gather2D
from learning.modules.resnet.resnet_13_light import ResNet13Light
from learning.modules.resnet.resnet_conditional import ResNetConditional
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter
from learning.modules.sentence_embeddings.sentence_embedding_self_attention import SentenceEmbeddingSelfAttention
from learning.modules.unet.unet_5_contextual import Unet5Contextual
from learning.modules.unet.unet_5_contextual_bneck3 import Unet5ContextualBneck
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.crossentropy2d import CrossEntropy2d
from learning.utils import get_n_params
from visualization import Presenter
import torch.nn.functional as F


#YAW_RANGE = 1.00
#YAW_RANGE = 1.59
#YAW_RANGE = 2.30
#YAW_RANGE = 3.14
#YAW_RANGE = 0
YAW_RANGE = 0.5

attention = True
splitemb = True

if attention:
    lstm_size = 30
    attention_heads = 5
    sentence_embedding_size = lstm_size * 2 * attention_heads
else:
    sentence_embedding_size = 30

sentence_embedding_layers = 1
word_embedding_size = 20

cut_gradients = False
NLL = False
BCE = False
CE = True

RESNET = False


class ModelTopDownPathGoalPredictor(nn.Module):

    def __init__(self, run_name, ignore_lang=False, class_loss=True, ground_loss=True):
        super(ModelTopDownPathGoalPredictor, self).__init__()
        self.run_name = run_name
        self.model_name = "top_down_path_pred_pretrain"
        self.writer = SummaryWriter(log_dir="runs/" + run_name)

        self.ignore_lang = ignore_lang
        self.class_loss = class_loss
        self.ground_loss = ground_loss

        # The feature net extracts the 2D feature map from the input image.
        # The label_pool down-sizes the ground-truth labels, which are input at the same size as the input image
        # The output predicted labels are the size of the feature map
        self.feature_net = ResNet13Light(32, down_pad=True)
        self.label_pool = nn.MaxPool2d(8)

        if self.ground_loss:
            self.lang_filter = MapLangSemanticFilter(sentence_embedding_size, 32, 3)
            self.aux_ground_linear = nn.Linear(3, 2)
            enable_weight_saving(self.lang_filter, "ground_filter")
            enable_weight_saving(self.aux_ground_linear, "ground_aux_linear")

        if RESNET:
            self.unet = ResNetConditional(sentence_embedding_size, 35, 2)
        else:
            unet_c_in= 35 if self.ground_loss else 32
            unet_hc1 = 48 if self.ground_loss else 48
            unet_hb1 = 24 if self.ground_loss else 24
            self.unet = Unet5ContextualBneck(unet_c_in, 2, sentence_embedding_size, hc1=unet_hc1, hb1=unet_hb1, hc2=128, split_embedding=splitemb)

        if attention:
            self.sentence_embedding = SentenceEmbeddingSelfAttention(
                word_embedding_size, lstm_size, sentence_embedding_layers, attention_heads=attention_heads)
        else:
            self.sentence_embedding = SentenceEmbeddingSimple(word_embedding_size, sentence_embedding_size, sentence_embedding_layers)

        self.gather2d = Gather2D()

        if self.class_loss:
            self.aux_class_linear = nn.Linear(32, 64)
            enable_weight_saving(self.aux_class_linear, "class_aux_linear")

        print("Sentence Embedding #Params: ", get_n_params(self.sentence_embedding))
        print("U-Net #Params: ", get_n_params(self.unet))
        print("Class auxiliary: ", self.class_loss)
        print("Ground auxiliary: ", self.ground_loss)

        # Enable saving of pre-trained weights
        enable_weight_saving(self.feature_net, "feature_resnet_light")
        enable_weight_saving(self.unet, "unet")
        enable_weight_saving(self.sentence_embedding, "sentence_embedding")

        if NLL:
            #self.mask_loss = nn.BCELoss()
            self.mask_loss = nn.NLLLoss2d()
        elif BCE:
            self.mask_loss = nn.BCEWithLogitsLoss()
        elif CE:
            self.spatialsoftmax = SpatialSoftmax2d()
            self.mask_loss = CrossEntropy2d()
        else:
            self.mask_loss = nn.MSELoss()

        self.aux_loss = nn.CrossEntropyLoss(reduce=True, size_average=True)
        self.epoch_numbers = {"train": 0, "eval": 0}
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.5)
        self.dropout3d = nn.Dropout3d(0.5)

        self.viz_images = []
        self.instructions = []


    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.sentence_embedding.init_weights()
        self.unet.init_weights()
        if self.ground_loss:
            self.aux_ground_linear.weight.data.normal_(0.001)
            self.aux_ground_linear.bias.data.fill_(0)
        if self.class_loss:
            self.aux_class_linear.weight.data.normal_(0.001)
            self.aux_class_linear.bias.data.fill_(0)

    def write_eoe_summaries(self, inference_type, epoch_num):
        pass

    def write_summaires(self, prefix, idx, total_loss, main_loss, emb_loss, class_loss, gnd_loss):
        full_prefix = self.model_name + "/" + prefix + "/"
        if self.writer is None:
            return

        self.writer.add_scalar(full_prefix + "total_loss", total_loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "main_loss", main_loss.data[0], idx)
        self.writer.add_scalar(full_prefix + "class_loss", class_loss.data[0], idx)
        if class_loss is not None:
            self.writer.add_scalar(full_prefix + "emb_loss", emb_loss.data[0], idx)
        if gnd_loss is not None:
            self.writer.add_scalar(full_prefix + "gnd_loss", gnd_loss.data[0], idx)

    def get_dataset(self, data=None, envs=None, eval=False, dataset_name=None, seg_level=True):
        return TopDownDataset(env_list=envs,
                              instr_negatives=False,
                              instr_negatives_similar_only=False,
                              seg_level=seg_level,
                              yaw_rand_range=0.0 if eval else YAW_RANGE,
                              img_w=512,
                              img_h=512,
                              map_w=256,
                              map_h=256,
                              incl_path=True,
                              incl_endpoint=True)

    def get_viz(self):
        presenter = Presenter()
        out = {
            "viz_img": []
        }
        for i, img in enumerate(self.viz_images):
            instruction = self.instructions[i]
            if len(instruction.view([-1])) < 2:
                instruction = [0]
            else:
                instruction = list(instruction.data.cpu().numpy().squeeze())
            instruction_str = debug_untokenize_instruction(instruction)
            viz_img = presenter.overlay_text(img, instruction_str)
            out["viz_img"].append(viz_img)
        return out

    def forward(self, images, instructions, instruction_masks):
        emb = self.sentence_embedding(instructions, torch.sum(instruction_masks, 1))

        # If the embedding returns an internal auxiliary, loss, pass it along
        emb_loss = cuda_var(torch.zeros([1]), self.is_cuda, self.cuda_device)
        if type(emb) is tuple:
            emb, emb_loss = emb

        feature_map = self.feature_net(images)
        feature_map = self.dropout2d(feature_map)

        if self.ground_loss:
            self.lang_filter.precompute_conv_weights(emb)
            ground_map = self.lang_filter(feature_map)
            feature_map = torch.cat([feature_map, ground_map], dim=1)

        # TODO: Testing breaking of gradients between ResNet and UNet
        if cut_gradients:
            feature_map_fwd = Variable(feature_map.data)
        else:
            feature_map_fwd = feature_map

        #if self.ground_loss:
        #    feature_map_fwd = feature_map_fwd[:, 0:3, :, :]

        pred_mask = self.unet(feature_map_fwd, emb)

        return pred_mask, feature_map, emb_loss

    def sup_loss_on_batch(self, batch, eval=False, viz=False):

        if eval:
            self.eval()
        else:
            self.train()

        images = cuda_var(batch["images"], self.is_cuda, self.cuda_device)
        instructions = cuda_var(batch["instr"], self.is_cuda, self.cuda_device)
        instruction_masks = cuda_var(batch["instr_mask"], self.is_cuda, self.cuda_device)
        label_masks = cuda_var(batch["traj_labels"], self.is_cuda, self.cuda_device)

        # Each of the above is a list of lists of tensors, where the outer list is over the batch and the inner list
        # is over the segments. Loop through and accumulate loss for each batch sequentially, and for each segment.
        # Reset model state (embedding etc) between batches, but not between segments.
        # We don't process each batch in batch-mode, because it's complicated, with the varying number of segments and all.

        batch_size = len(images)
        total_class_loss = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device), requires_grad=True)
        total_ground_loss = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device), requires_grad=True)
        count = 0

        label_masks = self.label_pool(label_masks)
        mask_pred, features, emb_loss = self(images, instructions, instruction_masks)


        if BCE:
            mask_pred_flat = mask_pred.view(-1, 1)
            label_masks_flat = label_masks - torch.min(label_masks)
            label_masks_flat = label_masks_flat / (torch.max(label_masks_flat) + 1e-9)
            label_masks_flat = label_masks_flat.view(-1, 1).clamp(0, 1)
            main_loss = self.mask_loss(mask_pred_flat, label_masks_flat)

        elif NLL:
            mask_pred_1 = F.softmax(mask_pred, 1, _stacklevel=5)
            mask_pred_2 = 1 - mask_pred_1
            mask_pred_1 = mask_pred_1.unsqueeze(1)
            mask_pred_2 = mask_pred_2.unsqueeze(1)
            mask_pred = torch.cat((mask_pred_1, mask_pred_2), dim=1)
            label_masks = label_masks.clamp(0,1)
            if self.is_cuda:
                label_masks = label_masks.type(torch.cuda.LongTensor)
            else:
                label_masks = label_masks.type(torch.LongTensor)
            main_loss = self.mask_loss(mask_pred, label_masks)

        elif CE:
            # Crossentropy2D internally applies logsoftmax to mask_pred,
            # but labels are already assumed to be a valid probability distribution, so no softmax is applied
            main_loss = self.mask_loss(mask_pred, label_masks)
            # So for nice plotting, we must manually do it
            mask_pred = self.spatialsoftmax(mask_pred)
        else:
            main_loss = self.mask_loss(mask_pred, label_masks)



        # sum emb loss if batch size > 1
        if type(emb_loss) == tuple:
            emb_loss = sum(emb_loss)

        # Extract the feature vectors corresponding to every landmark's location in the map
        # Apply a linear layer to classify which of the 64 landmarks it is
        # The landmark positions have to be divided by the same factor as the ResNet scaling factor
        lcount = 0
        for i in range(batch_size):
            if self.class_loss and len(batch["lm_pos"][i]) > 0:
                lcount += 1
                landmark_pos = cuda_var(batch["lm_pos"][i], self.is_cuda, self.cuda_device)
                landmark_indices = cuda_var(batch["lm_indices"][i], self.is_cuda, self.cuda_device)
                landmark_coords = (landmark_pos / 8).long()
                lm_features = self.gather2d(features[i:i+1, 0:32], landmark_coords)
                lm_pred = self.aux_class_linear(lm_features)
                class_loss = self.aux_loss(lm_pred, landmark_indices)
                total_class_loss = total_class_loss + class_loss

            if self.ground_loss and len(batch["lm_pos"][i]) > 0:
                 landmark_pos = cuda_var(batch["lm_pos"][i], self.is_cuda, self.cuda_device)
                 landmark_mentioned = cuda_var(batch["lm_mentioned"][i], self.is_cuda, self.cuda_device)
                 landmark_coords = (landmark_pos / 8).long()
                 g_features = self.gather2d(features[i:i+1, 32:35], landmark_coords)
                 lm_pred = self.aux_ground_linear(g_features)
                 ground_loss = self.aux_loss(lm_pred, landmark_mentioned)
                 total_ground_loss = total_ground_loss + ground_loss

        total_class_loss = total_class_loss / (lcount + 1e-9)
        total_ground_loss = total_ground_loss / (lcount + 1e-9)
        count += 1

        # Just visualization and debugging code
        if self.get_iter() % 50 == 0:
            presenter = Presenter()
            pred_viz_np = presenter.overlaid_image(images[0].data, mask_pred[0].data)
            labl_viz_np = presenter.overlaid_image(images[0].data, label_masks[0].data)
            comp = np.concatenate((pred_viz_np, labl_viz_np), axis=1)
            presenter.show_image(comp, "path_pred")

            if hasattr(self.sentence_embedding, "save_att_map"):
                self.sentence_embedding.save_att_map(self.get_iter(), i)

        total_loss = main_loss + 0.1 * total_class_loss + 0.001 * emb_loss + 0.1 * total_ground_loss
        total_loss = total_loss / (count + 1e-9)

        self.write_summaires("eval" if eval else "train", self.get_iter(),
                             total_loss, main_loss, emb_loss,
                             total_class_loss,
                             total_ground_loss)
        self.inc_iter()

        return total_loss
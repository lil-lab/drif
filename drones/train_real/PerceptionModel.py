import torch
from tensorboardX import SummaryWriter
from torch import nn as nn
from torch.autograd import Variable
import numpy as np

from learning.inputs.common import empty_float_tensor, cuda_var
from utils.simple_profiler import SimpleProfiler

from visualization import Presenter
from learning.datasets.fpv_image_dataset import FpvImageDataset#, IMG_HEIGHT, IMG_WIDTH
from learning.datasets.fpv_data_augmentation import get_display

#from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from learning.modules.img_to_img.img_to_features import ImgToFeatures
from env_config.definitions.landmarks import get_landmark_index_to_name

from drones.train_real.DomainAuxiliary import L2Auxiliary
import parameters.parameter_server as P


PROFILE = False


class PerceptionModel(nn.Module):
    def __init__(self, run_name, domain_loss_type=None, real=True):
        super(PerceptionModel, self).__init__()
        self.run_name = run_name
        self.real = real
        self.model_name = "PerceptionModel"
        self.train_writer = SummaryWriter(log_dir="runs/" + run_name + "/train")
        self.test_writer = SummaryWriter(log_dir="runs/" + run_name + "/test")
        self.presenter = Presenter()
        self.prefix = self.model_name + "/"
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        self.params = P.get_current_parameters()["Model"]
        self.aux_weights = P.get_current_parameters()["AuxWeightsPerception"]

        self.img_to_features = ImgToFeatures(channels=self.params["resnet_channels"],
                                             out_channels=self.params["feature_channels"],
                                             img_w=self.params["img_w"],
                                             img_h=self.params["img_h"])
        self.domain_loss_type = domain_loss_type

        self.loss = nn.CrossEntropyLoss(size_average=True)
        self.epoch_numbers = {}

        dropout = 0.5
        self.class_auxiliary = ClassAuxiliary2D("aux_class",
                                                None,
                                                self.params["feature_channels"],
                                                self.params["num_landmarks"],
                                                dropout,
                                                "fpv_features", "lm_pos_fpv", "lm_indices")

        if self.domain_loss_type == "l2":
            self.use_domain = True
            self.domain_auxiliary = L2Auxiliary(name="aux_domain_l2")


    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.img_to_features.cuda(device)
        self.class_auxiliary.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.img_to_features.init_weights()

    def forward(self, images):
        self.prof.tick("out")
        features_fpv = self.img_to_features.forward(images)
        return features_fpv

    def cuda_var(self, tensor):
        return cuda_var(tensor, self.is_cuda, self.cuda_device)

    def get_dataset(self, dataset_name, envs=None, eval=False, real=True):
        dataset = FpvImageDataset(envs, dataset_name, eval, real)
        return dataset

    def sup_loss_on_batch(self, batch, verbose=False, display=True, features=None):

        images = batch["images"]  # batch['images']#[sample['images'] for sample in batch]
        lm_indices = batch["lm_indices"]  # batch['lm_idx']#[sample['lm_idx'] for sample in batch]
        lm_pos_fpv = batch["lm_pos_fpv"]
        env_ids = batch["env_ids"]
        poses = batch["poses"]
        labels = batch["labels"]
        batch_size = len(images)

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        prof.tick("out")
        if verbose:
            for i_b in range(batch_size):

                print("\n \n \n Environment id:", env_ids[i_b])
                if not (lm_indices is None):

                    idx2name = get_landmark_index_to_name()
                    lm_indices_np = np.array(lm_indices[i_b])
                    lm_names = []
                    for i in lm_indices_np:
                        lm_names.append(idx2name[i])
                    print("The landmarks are:", lm_names)

                else:
                    print("No landmark")

                img_to_display = np.array(images[i_b], dtype=int).reshape(self.params['img_h'], self.params['img_w'], 3)
                get_display(img_to_display, np.array(lm_pos_fpv[i_b]))

            prof.tick("verbose")

        if batch_size > 1:
            images.squeeze_()

        total_class_accuracy = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))
        total_class_loss = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))
        prof.tick("cuda vars")

        # if features have not been precomputed
        if features is None:
            feature_maps_fpv_all = self(images)
        else:
            feature_maps_fpv_all = features
        prof.tick("features forward")


        for b in range(batch_size):

            prof.tick("loading")
            b_lm_indices = lm_indices[b]
            b_lm_pos_fpv = lm_pos_fpv[b]
            b_label = labels[b]

            if b_lm_indices is not None:
                b_lm_indices = self.cuda_var(b_lm_indices)
            b_lm_pos_fpv = self.cuda_var((b_lm_pos_fpv / 8).long())

            if display:
                r = np.random.rand()
                if r > 0.99:  # probability 0.01 to display images/features
                    real_or_sim = "real" if b_label > 0 else "sim"
                    c = feature_maps_fpv_all.shape[1]
                    h = feature_maps_fpv_all.shape[2]
                    w = feature_maps_fpv_all.shape[3]
                    preds = self.class_auxiliary.cls_linear(feature_maps_fpv_all[b].view([c, -1]).transpose(0, 1))
                    num_lms = preds.shape[1]
                    pred = preds.transpose(0, 1).view(num_lms, h, w)[[0, 3, 24], :, :]

                    Presenter().show_image(feature_maps_fpv_all[b].data[0:3], "fpv_features "+real_or_sim, torch=True, scale=8, waitkey=1)
                    Presenter().show_image(images[b].data, "fpv image " + real_or_sim, torch=True,
                                           scale=1, waitkey=1)
                    out = Presenter().overlaid_image(images[b].detach(), pred.detach(), gray_bg=True)
                    Presenter().show_image(out, "preds over image " + real_or_sim, torch=True, scale=1, waitkey=1)


            if b_lm_pos_fpv is None:
                continue
            prof.tick("display")

            # try:
            features_to_class = feature_maps_fpv_all[b].unsqueeze(0)
            #if detach_class:
            #    features_to_class.detach()

            class_loss, class_accuracy, _ = self.class_auxiliary(features_to_class,
                                            [b_lm_pos_fpv], [b_lm_indices])
            prof.tick("class loss")


            # except Exception as e:
            #    print(e)
            prof.tick("auxiliary loss")

            total_class_loss += class_loss
            total_class_accuracy += class_accuracy
            prof.loop()
            prof.print_stats(10)

        total_class_loss /= batch_size + 1e-9
        total_class_accuracy /= batch_size + 1e-9

        total_loss = total_class_loss * self.aux_weights["aux_class"]

        batch_metrics = {"/loss/total": total_loss, "/loss/class": total_class_loss,
                         "/accuracy/class": total_class_accuracy}

        return batch_metrics

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from learning.datasets.object_multi_matching_dataset_multi import ObjectMultiMatchingDatasetMulti
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.grounding.image_metric_embedding import ImageMetricEmbedding
from learning.models.navigation_model_component_base import NavigationModelComponentBase

from learning.models.visualization.viz_html_simple_matching_network import visualize_model, visualize_model_from_state
from learning.modules.generic_model_state import GenericModelState

from utils.simple_profiler import SimpleProfiler
from learning.meters_and_metrics.meter_server import get_current_meters

import parameters.parameter_server as P

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False

COSDIST = False
L2DIST = True
MAXSIM = True
AVGMAXSIM = False
AVGSIM = False

ABSOLOSS = True


class ModelMultiMatchingNetwork(NavigationModelComponentBase):

    def __init__(self, run_name="", domain="sim"):
        super(ModelMultiMatchingNetwork, self).__init__(run_name, domain)
        self.model_name = "multi_matching_multi"
        self.train_dataset = None
        self.eval_dataset = None

        self.root_params = P.get_current_parameters()["ModelSimpleMatchingNetwork"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.losses = AuxiliaryLosses()

        self.image_embedding = ImageMetricEmbedding()

        # The margin between matching and mismatching object distances
        self.tripled_alpha = 1.0

        # Matching objects must be closer than this. Mismatching objects must be further than this
        self.margin_beta = 2.0

        self.model_state = GenericModelState()

        self.runcount = 0

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------
        # TODO: Add distribution cross-entropy auxiliary objective
        self.losses.print_auxiliary_info()

    def init_weights(self):
        super().init_weights()
        self.image_embedding.init_weights()

    def unbatch(self, batch, halfway=False):
        obj_a, objs_b, objs_c, id_ab, id_c = batch
        obj_a = self.to_model_device(obj_a)
        objs_b = self.to_model_device(objs_b)
        objs_c = self.to_model_device(objs_c)
        return obj_a, objs_b, objs_c, id_ab, id_c

    def score(self, vec_a, vecs_b):
        #dist = ((vec_a - vec_b) ** 2).mean(1)
        vecs_a = vec_a[:, np.newaxis, :]
        similarity = -((vecs_a - vecs_b) ** 2).mean(2)
        best_similarity = similarity.max(dim=1)
        scores = -best_similarity.values
        best = best_similarity.indices

        return scores, best

    def score_matrix(self, vecs_a_ex, vecs_b_ex):
        """
        :param vecs_a_ex: NxD stack of vector encodings
        :param vecs_b_ex: MxQxD stack of vector encodings
        :return: NxM similarity matrix
        """
        # Create dimensions Q and M
        vecs_a_ex = vecs_a_ex[:, np.newaxis, np.newaxis, :]
        # Create dimension N:
        vecs_b_ex = vecs_b_ex[np.newaxis, :, :, :]
        # Both of the above are N x M x Q x C
        # N x M x Q matrix:
        similarity = -((vecs_a_ex - vecs_b_ex) ** 2).mean(3)
        best_similarity = similarity.max(dim=2)   # Max along Q axis
        scores = -best_similarity.values
        best = best_similarity.indices
        return scores, best

    def score_1_to_n_matrix(self, vec_a, vecs_c):
        """
        :param vec_a: BxD batch of D-dimensional vectors
        :param vecs_c: BxNxQxD batch vectors for N objects, Q images each, with D-dimensions
        :return: BxN matrix of similarity scores
        """
        vecs_a_ex = vec_a[:, np.newaxis, np.newaxis, :]
        vecs_c_ex = vecs_c
        similarity = -((vecs_a_ex - vecs_c_ex) ** 2).mean(3)
        best_similarity = similarity.max(dim=2)
        scores = -best_similarity.values
        best = best_similarity.indices
        return scores, best

    def forward(self, b_img_a, imgs_b):
        """
        :param b_img_a: batch of input images of shape Nx3xHxW
        :param imgs_b: batch of reference images of shape MxQx3xHxW
        N is the number of image regions, M is the novel object dataset (nod) dimension, Q is the number of images per nod item.
        It is assumed that the same novel object dataset is used for all timesteps (over B).
        :return: Stack of similarity matrices of shape BxQ
        """
        b_vecs_a = self.image_embedding.encode(b_img_a)  # NxD
        vecs_b = self.image_embedding.batch_encode(imgs_b)     # MxQxD
        scores, best = self.score_matrix(b_vecs_a, vecs_b)
        return scores

    def calculate_stats(self, scores_ab, scores_ac, ids_ab, ids_c):
        for score_ab, score_ac, id_ab, id_c in zip(scores_ab, scores_ac, ids_ab, ids_c):
            correct = score_ab.item() < score_ac.item()
            total_count = self.model_state.get("total_count", 0)
            total_mistakes = self.model_state.get("total_mistakes", 0)
            object_totals = self.model_state.get("object_totals", {})
            object_mistake_counts = self.model_state.get("object_mistake_counts", {})
            object_mistake_object_count = self.model_state.get("object_mistake_object_count", {})
            object_total_object_count = self.model_state.get("object_total_object_count", {})

            total_count += 1

            if id_ab not in object_totals:
                object_totals[id_ab] = 0
            object_totals[id_ab] += 1

            if not correct:
                total_mistakes += 1
                if id_ab not in object_mistake_counts:
                    object_mistake_counts[id_ab] = 0
                object_mistake_counts[id_ab] += 1

                if id_ab not in object_mistake_object_count:
                    object_mistake_object_count[id_ab] = {}
                if id_c not in object_mistake_object_count[id_ab]:
                    object_mistake_object_count[id_ab][id_c] = 0
                object_mistake_object_count[id_ab][id_c] += 1

            if id_ab not in object_total_object_count:
                object_total_object_count[id_ab] = {}
            if id_c not in object_total_object_count[id_ab]:
                object_total_object_count[id_ab][id_c] = 0
            object_total_object_count[id_ab][id_c] += 1

            self.model_state.put("total_count", total_count)
            self.model_state.put("total_mistakes", total_mistakes)
            self.model_state.put("object_totals", object_totals)
            self.model_state.put("object_mistake_counts", object_mistake_counts)
            self.model_state.put("object_mistake_object_count", object_mistake_object_count)
            self.model_state.put("object_total_object_count", object_total_object_count)

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[], model_state=None, viz=True):
        self.prof.tick("out")

        if model_state:
            self.model_state = model_state

        obj_a, objs_b, objs_c_all, id_ab, ids_c = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        vec_a = self.image_embedding.encode(obj_a)
        self.prof.tick("embed_a")

        vecs_b = self.image_embedding.batch_encode(objs_b)
        self.prof.tick("embed_b")

        batch_size = objs_c_all.shape[0]
        num_objs_c = objs_c_all.shape[1]
        num_imgs_c = objs_c_all.shape[2]
        c = objs_c_all.shape[3]
        h = objs_c_all.shape[4]
        w = objs_c_all.shape[5]
        objs_c_all_flat = objs_c_all.view([batch_size, num_objs_c * num_imgs_c, c, h, w])
        vecs_c_all_flat = self.image_embedding.batch_encode(objs_c_all_flat)
        vecs_c_all = vecs_c_all_flat.view([batch_size, num_objs_c, num_imgs_c, vecs_c_all_flat.shape[2]])

        self.prof.tick("embed_c")

        score_ab, b_best = self.score(vec_a, vecs_b)

        score_ac_matrix, c_best = self.score_1_to_n_matrix(vec_a, vecs_c_all)
        c_min = score_ac_matrix.min(1)
        score_ac_min = c_min.values
        best_matches = c_min.indices
        # TODO: Check correctness:
        #closest_c_best = c_best.index_select(index=best_matches, dim=1)
        closest_c_best = torch.stack([c_best[i][best_matches[i]] for i in range(batch_size)])
        closest_obj_c = torch.stack([objs_c_all[i][best_matches[i]] for i in range(batch_size)])
        #closest_obj_c = objs_c_all.index_select(index=best_matches, dim=1)
        closest_obj_id_c = [ids_c[i][best_matches[i].item()] for i in range(batch_size)]
        # ^- BxN matrix of similarity scores from vec_a to each of the N other objects

        loss = torch.mean(F.relu(score_ab - self.margin_beta)) + \
               torch.mean(F.relu(self.margin_beta - score_ac_min)) + \
               torch.mean(F.relu(score_ab[:, np.newaxis] - score_ac_min + self.tripled_alpha))

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        self.calculate_stats(score_ab, score_ac_min, id_ab, closest_obj_id_c)
        self.model_state.put("obj_a", obj_a)
        self.model_state.put("objs_b", objs_b)
        self.model_state.put("objs_c_all", objs_c_all)
        self.model_state.put("objs_c", closest_obj_c)
        self.model_state.put("scores_ab", score_ab)
        self.model_state.put("scores_ac", score_ac_min)
        self.model_state.put("scores_ac_all", score_ac_matrix)
        self.model_state.put("b_best", b_best)
        self.model_state.put("c_best", closest_c_best)
        if eval and viz:
            if viz:
                visualize_model_from_state(self.model_state, iteration=self.get_iter(), run_name=self.run_name)
            self.runcount += 1
        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")
        self.prof.tick("calc_losses")

        # Calculate metrics
        total = score_ab.shape[0]
        correct = (score_ab < score_ac_min).float().sum(0)
        accuracy = correct / (total + 1e-30)

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss.item(), iteration)

        self.writer.add_scalar(prefix, accuracy.item(), iteration)

        # Embedding distances so that we know roughly.
        mean_same_class_score = score_ab.mean().item()
        mean_closest_diff_class_score = score_ac_min.mean().item()
        mean_diff_class_score = score_ac_matrix.mean().item()
        self.writer.add_scalar(prefix + "/same_class_dist", mean_same_class_score, iteration)
        self.writer.add_scalar(prefix + "/closest_diff_class_dist", mean_closest_diff_class_score, iteration)
        self.writer.add_scalar(prefix + "/diff_class_dist", mean_diff_class_score, iteration)

        # Per-ID same-class scores
        for obj_id, score in zip(id_ab, score_ab):
            pass

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.model_state

    def _init_dataset(self, eval=False):
        if eval:
            dataset_name = P.get_current_parameters()["Data"].get("object_matching_test_dataset")
        else:
            dataset_name = P.get_current_parameters()["Data"].get("object_matching_training_dataset")
        grayscale = P.get_current_parameters()["Data"].get("grayscale", False)
        if grayscale:
            print("Matching network running GRAYSCALE!")
        return ObjectMultiMatchingDatasetMulti(
            run_name=self.run_name, dataset_name=dataset_name, eval=eval,grayscale=grayscale)

    def get_dataset(self, eval=False):
        if eval and not self.eval_dataset:
            self.eval_dataset = self._init_dataset(eval)
        if not eval and not self.train_dataset:
            self.train_dataset = self._init_dataset(eval)
        return self.eval_dataset if eval else self.train_dataset

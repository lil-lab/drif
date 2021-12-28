import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_io.paths import get_logging_dir
from learning.datasets.object_reference_dataset_with_database import ObjectReferenceDatasetWithDatabase
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.generic_model_state import GenericModelState
from learning.meters_and_metrics.binary_classification_metrics import BinaryClassificationMetrics
from learning.models.navigation_model_component_base import NavigationModelComponentBase

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter

import parameters.parameter_server as P

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class ModelObjectReferenceRecognizerWithDb(NavigationModelComponentBase):

    def __init__(self, run_name="", domain="sim", nowriter=False):
        super(ModelObjectReferenceRecognizerWithDb, self).__init__(run_name, domain, "obj_ref_rec", nowriter)

        self.root_params = P.get_current_parameters()["ModelObjectReferenceRecognizer"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        chunk_embedding_dim = self.root_params["embedding_dim"]
        hidden_size = 20
        self.linear1 = nn.Linear(chunk_embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size + 10, 1)
        self.act = nn.LeakyReLU()

        self.metric = BinaryClassificationMetrics()

        self.model_state = GenericModelState()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()#weight=torch.tensor([0.05, 1.00]).cuda())

        self.dropout = nn.Dropout(p=0.5)
        self.pos_threshold = 0.03
        #self.pos_threshold = 0.02

    def init_weights(self):
        pass

    def threshold(self, scores):
        probs = torch.sigmoid(scores)
        classes = (probs > self.pos_threshold).long()
        return classes

    def forward(self, chunk_vectors, chunk_affinities):
        """
        :param chunk_vectors: BxD batch of embedding vectors (e.g. Glove)
        :return:
        """
        x = self.dropout(chunk_vectors)

        affinity_scores = chunk_affinities.clamp(0, 1.0)
        bin_centers = (torch.arange(0, 1, 0.1) ** 3 + 1e-3)[None, :]
        cumulative_encoding = (affinity_scores[:, None] > bin_centers).long().float() * 4.0 - 2

        h = self.act(self.linear1(x))
        h = self.dropout(h)

        vecdim = len(x.shape) - 1
        h_and_enc = torch.cat([h, cumulative_encoding], dim=vecdim)
        if vecdim > 1: # Remove spurious axis
            assert h_and_enc.shape[1] == 1
            h_and_enc = h_and_enc[:, 0, :]

        chunk_scores = self.linear2(h_and_enc)
        return chunk_scores

    def unbatch(self, batch):
        # Inputs
        chunk_embeddings = self.to_model_device(batch[0])
        chunk_labels = self.to_model_device(batch[1][:, np.newaxis]).float()
        chunk_affinities = self.to_model_device(batch[2][:, np.newaxis]).float()
        return chunk_embeddings, chunk_affinities, chunk_labels

    def reset_metrics(self):
        self.metric.reset()

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.model_state

        chunk_embeddings, chunk_affinities, chunk_labels = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        print("NUM_POSITIVE LABELS: ", chunk_labels.sum().detach().item())

        # ----------------------------------------------------------------------------
        scores = self(chunk_embeddings, chunk_affinities)
        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")

        loss = self.bce_with_logits_loss(scores, chunk_labels)

        # Log values, and upload to meters
        self.metric.log_predictions(scores, chunk_labels)
        self.metric.consolidate()

        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss.item(), iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.model_state

    def get_dataset(self, eval=False):
        d = P.get_current_parameters()["Data"]
        object_database_name = d["object_database_sim"]
        return ObjectReferenceDatasetWithDatabase(eval=eval, object_database_name=object_database_name)
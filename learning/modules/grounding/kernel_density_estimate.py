import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

DEFAULT_SPATIAL_SIZE = (32, 32)
TEXT_KDE_VARIANCE = 0.5


class KernelDensityEstimate(torch.nn.Module):
    def __init__(self, gaussian_variance=None):
        super(KernelDensityEstimate, self).__init__()
        if gaussian_variance is None:
            self.gaussian_variance = nn.Parameter(torch.Tensor([0.5]).float(), requires_grad=True)
        else:
            self.gaussian_variance = gaussian_variance

    def forward(self, a_embeddings, b_embedding_sets, return_densities=False):
        """
        :param distance_tensor: NxMxQ tensor of distances, where N are points, M are clusters, and Q are points in each of the M clusters
        :param gaussian_variance: Gaussian kernel variance to use in the density estimate
        :return: NxM probability matrix, where for each of the N points, the row is a categorial distribution over the M clusters.
        """
        # Use kernel density estimation in the embedding space to assign probabilities to different clusters
        # Compute probability density of an object reference given each object in the database.
        # For the following, axes are ordered as: objrefs in instruction, objects in db, queries per obj, dims
        a = a_embeddings[:, np.newaxis, np.newaxis, :]
        covariance_mat = torch.eye(a.shape[3], device=a_embeddings.device) * self.gaussian_variance
        mu = b_embedding_sets[np.newaxis, :, :, :]

        # NumQueries x NumDBObjects x NumPrototypesPerObject x Dimension

        # Multivatiage Gaussian kernel density
        # x1 = (x - mu) * \Sigma^-1
        x1 = torch.tensordot(a - mu, covariance_mat.inverse(), dims=((3,), (0,)))
        # x2 = (x - mu) * \Sigma^-1 * (x - mu)^T
        x2 = (x1 * (a - mu)).sum(3)
        densities_per_ref_per_point = torch.exp(-0.5 * x2)
        densities_per_ref_per_cluster = densities_per_ref_per_point.sum(2)
        if return_densities:
            return densities_per_ref_per_cluster
        prob_of_cluster_given_ref = densities_per_ref_per_cluster / (
                densities_per_ref_per_cluster.sum(1, keepdim=True) + 1e-10
        )
        if (prob_of_cluster_given_ref != prob_of_cluster_given_ref).any():
            raise ValueError("NaN in KDE result")
        return prob_of_cluster_given_ref

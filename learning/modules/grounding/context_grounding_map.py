import torch
import numpy as np


class ContextGroundingMap(torch.nn.Module):
    def __init__(self):
        super(ContextGroundingMap, self).__init__()

    def forward(self, instance_seg_masks_w, instance_embddings, model_state):
        """

        :param instance_seg_masks_w: T x <num_objrefs + 1> x H x W
        :param instance_embddings: 1 x <num_objrefs> x dim
        :return:
        """
        assert len(instance_seg_masks_w.shape) == 4, (
            f"instance_seg_masks_w should be T x <num_objrefs + 1> x H x W. Actual shape: {instance_seg_masks_w.shape}")
        assert len(instance_embddings.shape) == 3, (
            f"instance_embddings should be 1 x <num_objrefs> x D. Actual shape: {instance_embddings.shape}")
        assert instance_seg_masks_w.shape[1] == instance_embddings.shape[1] + 1, (
            f"There should be one more mask than there are instance embeddings (the all-object mask)"
            f"instance_seg_masks_w.shape: {instance_seg_masks_w.shape}, instance_embeddings.shape: {instance_embddings.shape}")

        accum_object_reference_masks_w = instance_seg_masks_w[:, 1:]
        accum_all_obj_mask_w = instance_seg_masks_w[:, 0:1]
        # Axes ordering: time x objects x embedding_dim x H x W
        # Insert spatiotemporal axes (time, height, width)
        obj_ref_context_embeddings_plus = instance_embddings[0, np.newaxis, :, :, np.newaxis, np.newaxis]
        # Insert embedding axis (will be the new channel axis)
        object_reference_masks_w_plus = accum_object_reference_masks_w[:, :, np.newaxis, :, :]
        grounding_map_w_per_obj = object_reference_masks_w_plus * obj_ref_context_embeddings_plus
        # Each location is the average (over objects) of the embedding vectors
        # We can't use mean - the epsilon is needed, because sometimes the 1st axis is empty (if there are no obj refs)
        grounding_map_w = grounding_map_w_per_obj.sum(1) / (grounding_map_w_per_obj.shape[1] + 1e-10)
        full_context_grounding_map = torch.cat([accum_all_obj_mask_w, grounding_map_w], dim=1)

        model_state.tensor_store.keep_inputs("accum_all_obj_mask_w", accum_all_obj_mask_w)
        model_state.tensor_store.keep_inputs("per_object_context_maps", grounding_map_w_per_obj)

        return full_context_grounding_map
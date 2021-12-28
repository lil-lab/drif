import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import itertools

from data_io.model_io import load_pytorch_model

from learning.models.model_region_refinement import ModelRegionRefinementNetwork
from learning.models.model_facebook_rpn_wrapper import ModelFacebookRPNWrapper

from learning.modules.grounding.kernel_density_estimate import KernelDensityEstimate, TEXT_KDE_VARIANCE
from learning.modules.grounding.text_metric_embedding import TextMetricEmbedding
from learning.modules.grounding.image_metric_embedding import ImageMetricEmbedding
from learning.modules.image_resize import ImageResizer


class LanguageConditionedSegmentation(nn.Module):

    def __init__(self, run_name, domain):
        super().__init__()
        self.rpn = ModelFacebookRPNWrapper(run_name=f"{run_name}--rpn", domain=domain, nowriter=True)
        self.refiner = ModelRegionRefinementNetwork(run_name=f"{run_name}-refiner", domain=domain, nowriter=True)
        #self.rpn.make_picklable()
        #self.refiner.make_picklable()
        self.query_scale = 32

        self.image_resizer = ImageResizer()
        self.text_embedding = TextMetricEmbedding()
        self.image_embedding = ImageMetricEmbedding()
        self.text_kernel_density_estimate = KernelDensityEstimate(gaussian_variance=TEXT_KDE_VARIANCE)
        self.image_kernel_density_estimate = KernelDensityEstimate(gaussian_variance=2.0)

        self.loaded_pretrained_modules = False

    def enable_logging(self):
        self.rpn.enable_logging()
        self.refiner.enable_logging()

    def make_picklable(self):
        self.rpn.make_picklable()
        self.refiner.make_picklable()

    def load_pref_pretrained_modules(self, path_dict, real_drone):
        matching_model_name = path_dict.get("matching_model_name", False) or (
            path_dict.get("matching_model_name_real") if real_drone else path_dict.get("matching_model_name_sim"))
        print(f"Loading region matching network: {matching_model_name}")
        # TODO: Uncomment the namespace after training a new matching model
        load_pytorch_model(self.image_embedding, matching_model_name, namespace="image_embedding")

    def load_pretrained_modules(self, path_dict, real_drone):
        rpn_model_name = path_dict.get("rpn_model_name", False) or (
            path_dict.get("rpn_model_name_real") if real_drone else path_dict.get("rpn_model_name_sim"))
        print(f"Loading region proposal network: {rpn_model_name}")
        load_pytorch_model(self.rpn, rpn_model_name)
        # ...
        refinement_model_name = path_dict.get("refinement_model_name", False) or (
            path_dict.get("refinement_model_name_real") if real_drone else path_dict.get("refinement_model_name_sim"))
        print(f"Loading region refinement network: {refinement_model_name}")
        load_pytorch_model(self.refiner, refinement_model_name)

    def compute_image_similarity_matrices(self, b_images, b_bboxes, query_vectors):
        """
        :param b_images: Tx3xHxW sequence of T images
        :param b_bboxes: List of Bx4 tensors, each a stack of B bounding boxes in xmin, ymin, xmax, ymax ordering
        :param query_images: MxQx3xHxW tensor of Q images for each of the M novel objects
        :return:
        """
        traj_len = len(b_images)
        M = query_vectors.shape[0]

        # First resize and normalize all bounding box crops one by one, and stack them into a single tensor
        flat_indices = []
        flat_crops_norm = []
        for i, (image, bboxes) in enumerate(zip(b_images, b_bboxes)):
            # Crop each bbox out of the image and get a batch Nx3xHxW
            crops_oddsized = []
            for xmin, ymin, xmax, ymax in bboxes:
                crop = image[:, int(ymin):int(ymax), int(xmin):int(xmax)].contiguous().detach()
                crops_oddsized.append(crop)
            if len(bboxes) > 0:
                # Resize each crop to a square as expected by the matching and refinement models
                crops_square_norm = self.image_resizer.resize_to_target_size_and_normalize(crops_oddsized)
                flat_indices += [i] * crops_square_norm.shape[0]
                flat_crops_norm.append(crops_square_norm)
        flat_indices = torch.tensor(flat_indices, device=b_images.device, dtype=torch.long)
        if len(flat_crops_norm) == 0:
            flat_crops_norm = torch.zeros([0, 3, self.query_scale, self.query_scale],
                                          dtype=torch.float32, device=image.device)
        else:
            flat_crops_norm = torch.cat(flat_crops_norm, dim=0)

        # Then encode all the crops in all the images at once
        stack_crop_vectors = self.image_embedding.encode(flat_crops_norm).detach()

        # Finally, get probability of each novel object for each of the image crops
        stack_prob_of_object_given_crop = self.image_kernel_density_estimate(stack_crop_vectors, query_vectors).detach().float()

        # Before returning, chunk the stacked results back accoding to timesteps
        # Create dummy elements for those timesteps where there are no bounding boxes detected in images
        zero_sim_matrix = torch.zeros([0, M], dtype=torch.float32, device=image.device)
        zero_crops = torch.zeros([0, 3, self.query_scale, self.query_scale],
                                             dtype=torch.float64,device=b_images.device)
        sim_matrices = [[] for _ in range(traj_len)]
        all_crops = [[] for _ in range(traj_len)]
        for idx, crop, sim_matrix in zip(flat_indices, flat_crops_norm, stack_prob_of_object_given_crop):
            sim_matrices[idx].append(sim_matrix)
            all_crops[idx].append(crop)

        all_sim_matrices = [torch.stack(m, dim=0) if len(m) > 0 else zero_sim_matrix for m in sim_matrices]
        all_crops = [torch.stack(m, dim=0) if len(m) > 0 else zero_crops for m in all_crops]
        return all_sim_matrices, all_crops, flat_crops_norm, flat_indices

    def refine_crops_to_masks(self, images, flat_crops_norm, flat_indices, all_bboxes):
        """
        :param images: Tx3xHxW tensor of T images in the trajectory
        :param flat_crops_norm: #TODO
        :param flat_indices: #TODO
        :param all_bboxes: List of Bx4 tensors, each a stack of bounding boxes in format xmin, ymin, xmax, ymax
        :return: T-length list of Bx1xHxW dimensional masks with same spatial size as the input image.
        """
        T = images.shape[0]
        H = images.shape[2]
        W = images.shape[3]
        # TODO: Flatten the boxes at once, and do it outside of image similarity and refinement functions.
        flat_boxes = list(itertools.chain(*all_bboxes))

        num_crops = flat_crops_norm.shape[0]
        all_object_masks = torch.zeros([num_crops, 1, H, W], dtype=flat_crops_norm.dtype, device=images.device)

        # Only run the refiner if we have at least one detected object
        if num_crops > 0:
            # Call the refiner to get the masks
            # MAKE SURE TO DETACH SO THAT WE DON'T BACKPROP INTO THE REFINER
            refined_mask_crops = self.refiner(flat_crops_norm).detach()
            # Then place the masks at the bounding box in the FPV image to get the first-person segmentation masks.
            for i in range(refined_mask_crops.shape[0]):
                mask_crop = refined_mask_crops[i]
                xmin, ymin, xmax, ymax = flat_boxes[i]
                s_mask_crop = F.interpolate(mask_crop.unsqueeze(0), size=(int(ymax)-int(ymin), int(xmax)-int(xmin)), mode="nearest")[0]
                all_object_masks[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = s_mask_crop

        # Finally, chunk the masks according to their timesteps into a List of Lists of masks
        object_masks_out = [[] for _ in range(T)]
        for t, object_mask in zip(flat_indices, all_object_masks):
            object_masks_out[t].append(object_mask)

        object_masks_out = [torch.stack(m) if len(m) > 0 else
                            torch.zeros([0, 1, H, W], device=images.device, dtype=all_object_masks.dtype)
                            for m in object_masks_out]

        return object_masks_out

    def chunk_affinity_scores(self, noun_chunks, object_database):
        # Compute the text similarity matrix
        device = next(self.image_embedding.parameters()).device
        reference_embeddings = self.text_embedding.encode(noun_chunks, device)
        database_text_embeddings = self.text_embedding.batch_encode(object_database["object_references"], device)
        text_similarity_matrix = self.text_kernel_density_estimate(reference_embeddings, database_text_embeddings, return_densities=True)
        chunk_database_affinities = text_similarity_matrix.max(1).values if text_similarity_matrix.shape[0] > 0 else torch.zeros([], device=device)
        return chunk_database_affinities

    def pre_forward(self, object_references, object_database):
        """
        # TODO: Don't run pre-forward on every step when executing an instruction!
        The part of forward function that is not differentiable and should run inside the dataset to be parallelized.
        :param instruction_string: String instruction
        :param db_strings_batch: List (over batch) of lists (over objects) of lists (for each object) of strings
        :return:
        """

        # Encode novel objects into vectors and add to the database - these won't change during execution
        device = next(self.image_embedding.parameters()).device
        cdev = object_database["object_images"].device
        object_images = object_database["object_images"].to(device)
        object_vectors = self.image_embedding.batch_encode(object_images).detach()
        object_vectors = object_vectors.to(cdev)
        add_to_database = {"object_vectors": object_vectors}

        # Compute the text similarity matrix
        reference_embeddings = self.text_embedding.encode(object_references, device)
        database_text_embeddings = self.text_embedding.batch_encode(object_database["object_references"], device)
        text_similarity_matrix = self.text_kernel_density_estimate(reference_embeddings, database_text_embeddings)

        return text_similarity_matrix, add_to_database

    def post_forward(self, model_state, images, object_database, text_similarity_matrix):
        self.rpn.eval()
        self.refiner.eval()

        traj_len = len(images)

        # 1. Extract bounding box proposals in the scene images:
        bboxes, objectness_probs = self.rpn(images)
        # Here we have a B-length list of Ki x 4 tensors storing the predicted bounding boxes
        # For each image in the batch (trajectory)
        # Compute a list of Ki x M similarity matrices
        # images is Tx3xHxW, query_images: MxQx3xHxW
        #object_vectors = model_state.get("object_vectors")
        object_vectors = object_database["object_vectors"]
        num_objects = object_vectors.shape[0]

        # 2. Compute grid of P(o | b)   ; P(object | bounding box)
        visual_similarity_matrix, all_crops, flat_crops, flat_indices = (
            self.compute_image_similarity_matrices(images, bboxes, object_vectors))

        model_state.tensor_store.keep_inputs("visual_similarity_matrix", visual_similarity_matrix)
        model_state.tensor_store.keep_inputs("region_crops", all_crops)

        # 3. Retrieve P(o | r) ; P(object | object reference text)
        text_similarity_matrix = text_similarity_matrix

        # 3. Compute P(b | r) for each bounding box and object reference, for each timestep
        p_b_given_r_probs = []
        # TODO: see if we can batch some things across trajectory length
        for i in range(traj_len):
            # N - number of references, M - number of novel objects, B - number of bounding boxes recognized
            # The output will be a BxN matrix
            p_o_given_b = visual_similarity_matrix[i]  # BxM matrix
            p_o_given_r = text_similarity_matrix  # NxM matrix
            num_bboxes = p_o_given_b.shape[0]

            # Probability of each bounding box is the objectness score
            if len(objectness_probs[i]) > 0:
                p_b = torch.stack(objectness_probs[i], dim=0)  # B-length vector
                p_b = p_b / (torch.sum(p_b) + 1e-10)  # Normalize into a valid prob distribution
            else:
                p_b = torch.tensor([], device=p_o_given_r.device)
            # p_b = torch.tensor([1 / (num_bboxes + 1e-9)] * num_bboxes, device=p_e_given_b.device)

            p_o_given_b = p_o_given_b[:, :, np.newaxis]  # BxMx1 matrix
            p_o_given_r = p_o_given_r.transpose(0, 1)[np.newaxis, :, :]  # 1xMxN matrix
            p_b = p_b[:, np.newaxis, np.newaxis]  # Bx1x1 matrix

            # Calculate P(e) by marginalizing P(e,b) over b.
            # This assumes that every object corresponds to a weighted combination of bounding boxes.
            # p_e = (p_e_given_b * p_b).sum(0, keepdim=True)
            p_e = torch.tensor([1 / num_objects] * num_objects, device=p_b.device)
            p_e = p_e[np.newaxis, :, np.newaxis]

            p_b_given_e = p_o_given_b * p_b / (p_e + 1e-9)  # BxMx1 matrix
            p_b_and_e_given_r = p_b_given_e * p_o_given_r  # BxMxN matrix of P(bi,ej|rk)
            # Marginalize over novel objects
            p_b_given_r = p_b_and_e_given_r.sum(1)  # BxN matrix of P(bi|rk)
            # This shouldn't actually sum to 1 over bounding boxes! If it did, it meant that all objects are always observed
            # It should sum to less than 1, but sometimes sums to more?
            # YES! It's because p(b) is probability density of a bounding box, not the precise probability.
            p_b_given_r_probs.append(p_b_given_r)

        model_state.tensor_store.keep_inputs("grounding_matrix", p_b_given_r_probs)

        # 4. For each crop, compute refinement
        all_region_masks = self.refine_crops_to_masks(images, flat_crops, flat_indices, bboxes)
        # Mask is a list (over timesteps) of lists (over bboxes) for masks for each object that has a bounding box

        # 5. Compute a mask for each object reference by doing a probability-weighted average of recognized object masks
        all_grounded_object_reference_masks = []
        all_all_object_masks = []
        for t in range(traj_len):
            # BxN matrix where B is number of bounding boxes, N is number of object references
            p_b_given_r = p_b_given_r_probs[t]
            # Bx1xHxW tensor where B is number of bounding boxes
            masks = all_region_masks[t]
            # Reshape both to: BxNx1xHxW
            masks_exp = masks[:, np.newaxis, :, :, :]
            p_b_given_r_exp = p_b_given_r[:, :, np.newaxis, np.newaxis, np.newaxis]
            # For each object reference, aggregate masks
            weighted_masks = masks_exp * p_b_given_r_exp
            if weighted_masks.shape[0] > 0 and weighted_masks.shape[1] > 0:
                object_reference_masks_t = weighted_masks.max(0).values.float()
            else:
                object_reference_masks_t = weighted_masks.sum(0).float()
            # The resulting mask is Nx1xHxW
            all_grounded_object_reference_masks.append(object_reference_masks_t)

            # Also compute general "objectness" mask that includes all objects, by taking a max over all b-box masks
            if masks.shape[0] > 0:
                all_object_mask = masks.max(0).values
            else:  # If there are no bounding boxes, initialize a black masks of zeroes.
                all_object_mask = torch.zeros([masks.shape[1], masks.shape[2], masks.shape[3]],
                                              device=images.device, dtype=masks.dtype)
            all_all_object_masks.append(all_object_mask)

        all_grounded_object_reference_masks_fpv = torch.stack(all_grounded_object_reference_masks, dim=0)  # TxNx1xHxW
        all_grounded_object_reference_masks_fpv = all_grounded_object_reference_masks_fpv[:, :, 0, :, :]  # TxNxHxW

        # Combined mask of all objecty-looking things
        all_all_object_masks_fpv = torch.stack(all_all_object_masks, dim=0)  # Tx1xHxW

        model_state.tensor_store.keep_inputs("region_masks", all_region_masks)
        model_state.tensor_store.keep_inputs("object_reference_masks_fpv", all_grounded_object_reference_masks_fpv)

        # 6. Project segmentation masks to the global reference frame
        full_fpv = torch.cat([all_all_object_masks_fpv, all_grounded_object_reference_masks_fpv], dim=1)
        model_state.tensor_store.keep_inputs("full_masks_fpv", full_fpv)
        return full_fpv

    def forward(self, model_state, images, object_references, object_database):
        self.pre_forward(model_state, object_references, object_database)
        return self.post_forward(model_state, images)

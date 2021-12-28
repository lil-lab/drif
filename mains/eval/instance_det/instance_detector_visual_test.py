import os
import torch
import numpy as np
import imageio
from learning.inputs.vision import standardize_image
from learning.datasets.few_shot_instance_dataset import FewShotInstanceDataset
from data_io.models import load_model

from visualization import Presenter

import parameters.parameter_server as P

QUICK_TEST = False


def instance_detector_quick_test():
    model, model_loaded = load_model()
    quick_test_dir = P.get_current_parameters()["Data"]["quick_test_dataset"]
    quick_test_out_dir = P.get_current_parameters()["Data"]["quick_test_outdir"]
    examples = os.listdir(quick_test_dir)
    for example in examples:
        example_dir = os.path.join(quick_test_dir, example)
        scene_path = os.path.join(example_dir, "scene.png")
        query_path = os.path.join(example_dir, "query.png")
        scene_img = imageio.imread(scene_path)
        query_img = imageio.imread(query_path)
        scene_image_t = torch.from_numpy(standardize_image(scene_img))[np.newaxis, :, :, :].cuda()
        query_image_t = torch.from_numpy(standardize_image(query_img))[np.newaxis, :, :, :].cuda()
        query_images_t = query_image_t[:, np.newaxis, :, :, :]

        predicted_scores = model(query_images_t, scene_image_t)
        Presenter().show_image(scene_img, "scene", scale=4, waitkey=False)
        Presenter().show_image(query_img, "query", scale=4, waitkey=False)
        predicted_dist = predicted_scores.softmax()
        print("Prob OOB: ", predicted_dist.outer_prob_mass.item())
        predicted_dist.show("prediction", scale=4, waitkey=False)

        out_dir = os.path.join(quick_test_out_dir, example)
        os.makedirs(out_dir, exist_ok=True)
        imageio.imsave(os.path.join(out_dir, "scene.png"), scene_img)
        imageio.imsave(os.path.join(out_dir, "query.png"), query_img)
        pred_image = predicted_dist.visualize()
        imageio.imsave(os.path.join(out_dir, "prediction.png"), pred_image)


def instance_detector_eval():
    model, model_loaded = load_model()
    dataset = model.get_dataset(eval=True)
    # Enable data augmentation for testing
    #dataset.eval = False
    for example in dataset:
        scene_image_t = example["scene"][np.newaxis, :, :, :].cuda()
        #query_images_t = [q.cuda()[np.newaxis, :, :, :] for q in example["query"]]
        query_images_t = example["query"].cuda()[np.newaxis, :, :, :]
        mask_image_t = example["mask"][np.newaxis, np.newaxis, :, :].cuda()

        predicted_scores = model(query_images_t, scene_image_t)
        Presenter().show_image(scene_image_t, "scene", scale=4, waitkey=False)
        for i, scale_images in enumerate(query_images_t):
            for j, query_image in enumerate(scale_images):
                Presenter().show_image(query_image, f"query_{j}_scale_{i}", scale=4, waitkey=False)
        Presenter().show_image(mask_image_t, "mask", scale=4, waitkey=False)
        predicted_dist = predicted_scores.softmax()
        print("Prob OOB: ", predicted_dist.outer_prob_mass.item())
        predicted_dist.show("prediction", scale=4, waitkey=True)


if __name__ == "__main__":
    P.initialize_experiment()
    if QUICK_TEST:
        instance_detector_quick_test()
    else:
        instance_detector_eval()
import cv2
import torch
import os
import numpy as np
from imageio import imsave

from data_io.paths import get_results_dir
from data_io.models import load_model
from torch.utils.data.dataloader import DataLoader

from data_io.instructions import get_word_to_token_map, get_all_instructions, debug_untokenize_instruction, tokenize_instruction
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.instructions import get_restricted_env_id_lists
from data_io.env import load_env_split
from parameters.parameter_server import initialize_experiment, get_current_parameters

from visualization import Presenter


# Supervised learning parameters
def train_supervised():
    initialize_experiment()

    setup = get_current_parameters()["Setup"]
    supervised_params = get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]
    dataset_names = get_current_parameters()["Data"]["dataset_names"]

    model, model_loaded = load_model()

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
    dataset = model.get_dataset(data=None, envs=dev_envs, domain="sim", dataset_names=dataset_names, dataset_prefix="supervised", eval=True)
    train_i, dev_i, test_i, corpus = get_all_instructions()
    token2word, word2token = get_word_to_token_map(corpus)

    if hasattr(dataset, "set_word2token"):
        dataset.set_word2token(token2word, word2token)

    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        timeout=0,
        drop_last=False)

    outdir = os.path.join(get_results_dir(), "interactive_obj_rec")
    os.makedirs(outdir, exist_ok=True)

    for batch in dataloader:
        if batch is None:
            continue

        images = model.to_model_device(batch["images"][0:1])
        states = model.to_model_device(batch["states"][0:1])

        env_id = batch["md"][0]["env_id"]
        set_idx = batch["md"][0]["set_idx"]
        seg_idx = batch["md"][0]["seg_idx"]
        out_fname = f"{env_id}_{set_idx}_{seg_idx}.png"

        Presenter().show_image(images[0], "Input image", scale=4)

        query = input("Input query >>")
        query_tok = tokenize_instruction(query, word2token)
        query_tok_t = torch.tensor([query_tok]).to(images.device)
        quert_tok_lenghts = torch.tensor([len(query_tok)]).to(images.device)

        masks = model(images, states, query_tok_t, quert_tok_lenghts)
        obj_prob = masks.softmax()

        prob_overlay = obj_prob.visualize(idx=0)
        base_image = Presenter().prep_image(images[0])
        overlaid_image = Presenter().overlaid_image(base_image, prob_overlay, channel=2)
        h = overlaid_image.shape[0]
        w = overlaid_image.shape[1]
        overlaid_image = cv2.resize(overlaid_image, (4 * w, 4 * h))
        cv2.putText(overlaid_image, query, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 1)
        Presenter().show_image(overlaid_image)

        print("saving to: ", out_fname)
        squashed = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
        squashed = squashed - squashed.min()
        squashed = squashed / (squashed.max() + 1e-9)
        squshed = (squashed * 255).astype(np.uint8)
        imsave(os.path.join(outdir, out_fname), squshed)


if __name__ == "__main__":
    train_supervised()

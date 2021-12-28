import os
import numpy as np
import cv2
import torch

from data_io.models import load_model
from data_io.instructions import get_correct_eval_env_id_list
from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map

from learning.datasets.top_down_dataset import apply_affine, apply_affine_on_pts
from learning.inputs.common import cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.affine_2d import Affine2D

from data_io.instructions import debug_untokenize_instruction
from pomdp.pomdp_interface import PomdpInterface

from torch.utils.data import DataLoader
from visualization import Presenter

import parameters.parameter_server as P

import torch.nn.functional as F
import torch.nn

# Supervised learning parameters
SUPERVISED_EPOCHS = 100
EVAL_INTERVAL = 10


def write_empty_instruction():
    with open("tmp_instr.txt", "w") as fp:
        fp.write("")


def write_instruction(instruction_str):
    with open("tmp_instr.txt", "w") as fp:
        fp.write(instruction_str)


def write_real_instruction(instruction_str):
    with open("tmp_instr_real.txt", "w") as fp:
        fp.write(instruction_str)


def read_instruction_file():
    with open("tmp_instr.txt", "r") as fp:
        instruction = fp.readline()
    return instruction


def rollout_model(model, env, env_id, set_id, seg_id, tok_instruction):
    state = env.reset(seg_id)
    while True:
        action = model.get_action(state, tok_instruction)
        state, reward, done = env.step(action)
        cv2.waitKey(1)
        if action[3] > 0.5:
            print("Done!")
            break
    return True


def launch_ui():
    python_script = os.path.join(os.path.dirname(__file__), "interact_ui.py")
    os.system("python3 " + python_script + " &")


def train_top_down_pred():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    launch_ui()

    env = PomdpInterface()

    model, model_loaded = load_model(model_name_override=setup["top_down_model"],
                                     model_file_override=setup["top_down_model_file"])

    exec_model, wrapper_model_loaded = load_model(model_name_override=setup["wrapper_model"],
                                                  model_file_override=setup["wrapper_model_file"])

    affine2d = Affine2D()
    if model.is_cuda:
        affine2d.cuda()

    eval_envs = get_correct_eval_env_id_list()
    train_instructions, dev_instructions, test_instructions, corpus = get_all_instructions(max_size=setup["max_envs"])
    all_instr = {**train_instructions, **dev_instructions, **train_instructions}
    token2term, word2token = get_word_to_token_map(corpus)

    dataset_name = P.get_current_parameters().get("Data").get("dataset_name")
    dataset = model.get_dataset(envs=eval_envs, dataset_prefix=dataset_name, dataset_prefix="supervised", eval=True, seg_level=False)
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    for b, batch in enumerate(dataloader):

        images = batch["images"]
        instructions = batch["instr"]
        label_masks = batch["traj_labels"]
        affines = batch["affines_g_to_s"]
        env_ids = batch["env_id"]
        set_idxs = batch["set_idx"]
        seg_idxs = batch["seg_idx"]

        env_id = env_ids[0][0]
        set_idx = set_idxs[0][0]
        env.set_environment(env_id, instruction_set=all_instr[env_id][set_idx]["instructions"])
        env.reset(0)

        num_segments = len(instructions[0])

        write_instruction("")
        write_real_instruction("None")
        instruction_str = read_instruction_file()
        print("Initial instruction: ", instruction_str)

        # TODO: Reset model state here if we keep any temporal memory etc
        for s in range(num_segments):
            start_state = env.reset(s)
            keep_going = True
            real_instruction = cuda_var(instructions[0][s], setup["cuda"], 0)
            tmp = list(real_instruction.data.cpu()[0].numpy())
            real_instruction_str = debug_untokenize_instruction(tmp)
            write_real_instruction(real_instruction_str)
            #write_instruction(real_instruction_str)
            #instruction_str = real_instruction_str
            image = cuda_var(images[0][s], setup["cuda"], 0)
            label_mask = cuda_var(label_masks[0][s], setup["cuda"], 0)
            affine_g_to_s = affines[0][s]

            while keep_going:
                write_real_instruction(real_instruction_str)

                while True:
                    cv2.waitKey(200)
                    instruction = read_instruction_file()
                    if instruction == "CMD: Next":
                        print("Advancing")
                        keep_going = False
                        write_empty_instruction()
                        break
                    elif instruction == "CMD: Reset":
                        print("Resetting")
                        env.reset(s)
                        write_empty_instruction()
                    elif len(instruction.split(" ")) > 1:
                        instruction_str = instruction
                        print("Executing: ", instruction_str)
                        break

                if not keep_going:
                    continue

                #instruction_str = read_instruction_file()
                # TODO: Load instruction from file
                tok_instruction = tokenize_instruction(instruction_str, word2token)
                instruction_t = torch.LongTensor(tok_instruction).unsqueeze(0)
                instruction_v = cuda_var(instruction_t, setup["cuda"], 0)
                instruction_mask = torch.ones_like(instruction_v)
                tmp = list(instruction_t[0].numpy())
                instruction_dbg_str = debug_untokenize_instruction(tmp, token2term)

                res = model(image, instruction_v, instruction_mask)
                mask_pred = res[0]
                shp = mask_pred.shape
                mask_pred = F.softmax(mask_pred.view([2, -1]), 1).view(shp)
                #mask_pred = softmax2d(mask_pred)

                # TODO: Rotate the mask_pred to the global frame
                affine_s_to_g = np.linalg.inv(affine_g_to_s)
                S = 8.0
                affine_scale_up = np.asarray([[S, 0, 0],
                                             [0, S, 0],
                                              [0, 0, 1]])
                affine_scale_down = np.linalg.inv(affine_scale_up)

                affine_pred_to_g = np.dot(affine_scale_down, np.dot(affine_s_to_g, affine_scale_up))
                #affine_pred_to_g_t = torch.from_numpy(affine_pred_to_g).float()

                mask_pred_np = mask_pred.data.cpu().numpy()[0].transpose(1, 2, 0)
                mask_pred_g_np = apply_affine(mask_pred_np, affine_pred_to_g, 32, 32)
                print("Sum of global mask: ", mask_pred_g_np.sum())
                mask_pred_g = torch.from_numpy(mask_pred_g_np.transpose(2, 0, 1)).float()[np.newaxis, :, :, :]
                exec_model.set_ground_truth_visitation_d(mask_pred_g)

                # Create a batch axis for pytorch
                #mask_pred_g = affine2d(mask_pred, affine_pred_to_g_t[np.newaxis, :, :])

                mask_pred_np[:, :, 0] -= mask_pred_np[:, :, 0].min()
                mask_pred_np[:, :, 0] /= (mask_pred_np[:, :, 0].max() + 1e-9)
                mask_pred_np[:, :, 0] *= 2.0
                mask_pred_np[:, :, 1] -= mask_pred_np[:, :, 1].min()
                mask_pred_np[:, :, 1] /= (mask_pred_np[:, :, 1].max() + 1e-9)

                presenter = Presenter()
                #presenter.show_image(mask_pred_g_np, "mask_pred_g", torch=False, waitkey=1, scale=4)
                pred_viz_np = presenter.overlaid_image(image.data, mask_pred_np, channel=0)
                # TODO: Don't show labels
                # TODO: OpenCV colours
                #label_mask_np = p.data.cpu().numpy()[0].transpose(1,2,0)

                labl_viz_np = presenter.overlaid_image(image.data, label_mask.data, channel=0)
                viz_img_np = np.concatenate((pred_viz_np, labl_viz_np), axis=1)
                viz_img_np = pred_viz_np

                viz_img = presenter.overlay_text(viz_img_np, instruction_dbg_str)
                cv2.imshow("interactive viz", viz_img)
                cv2.waitKey(100)

                rollout_model(exec_model, env, env_ids[0][s], set_idxs[0][s], seg_idxs[0][s], tok_instruction)
                write_instruction("")


if __name__ == "__main__":
    train_top_down_pred()

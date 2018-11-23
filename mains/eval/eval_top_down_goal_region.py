import numpy as np
import torch

from data_io.instructions import get_correct_eval_env_id_list
from data_io.models import load_model

from learning.inputs.common import cuda_var
from torch.utils.data import DataLoader
import parameters.parameter_server as P

OK_DIST = 3.2


def evaluate():
    P.initialize_experiment()

    model, model_loaded = load_model()
    eval_envs = get_correct_eval_env_id_list()

    model.eval()
    dataset = model.get_dataset(data=None, envs=eval_envs, dataset_name="supervised", eval=eval, seg_level=False)
    dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            timeout=0)

    count = 0
    success = 0
    total_dist = 0

    for batch in dataloader:
        if batch is None:
            print("None batch!")
            continue

        images = batch["images"]
        instructions = batch["instr"]
        label_masks = batch["traj_labels"]

        # Each of the above is a list of lists of tensors, where the outer list is over the batch and the inner list
        # is over the segments. Loop through and accumulate loss for each batch sequentially, and for each segment.
        # Reset model state (embedding etc) between batches, but not between segments.
        # We don't process each batch in batch-mode, because it's complicated, with the varying number of segments and all.
        # TODO: This code is outdated and wrongly discretizes the goal location. Grab the fixed version from the old branch.

        batch_size = len(images)
        print("batch: ", count)
        print("successes: ", success)

        for i in range(batch_size):
            num_segments = len(instructions[i])

            for s in range(num_segments):
                instruction = cuda_var(instructions[i][s], model.is_cuda, model.cuda_device)
                instruction_mask = torch.ones_like(instruction)
                image = cuda_var(images[i][s], model.is_cuda, model.cuda_device)
                label_mask = cuda_var(label_masks[i][s], model.is_cuda, model.cuda_device)

                label_mask = model.label_pool(label_mask)

                goal_mask_l = label_mask[0, 1, :, :]
                goal_mask_l_np = goal_mask_l.data.cpu().numpy()
                goal_mask_l_flat = np.reshape(goal_mask_l_np, [-1])
                max_index_l = np.argmax(goal_mask_l_flat)
                argmax_loc_l = np.asarray([int(max_index_l / goal_mask_l_np.shape[1]), int(max_index_l % goal_mask_l_np.shape[1])])

                if np.sum(goal_mask_l_np) < 0.01:
                    continue

                mask_pred, features, emb_loss = model(image, instruction, instruction_mask)
                goal_mask = mask_pred[0,1,:,:]
                goal_mask_np = goal_mask.data.cpu().numpy()
                goal_mask_flat = np.reshape(goal_mask_np, [-1])
                max_index = np.argmax(goal_mask_flat)

                argmax_loc = np.asarray([int(max_index / goal_mask_np.shape[1]), int(max_index % goal_mask_np.shape[1])])

                dist = np.linalg.norm(argmax_loc - argmax_loc_l)
                if dist < OK_DIST:
                    success += 1
                count += 1
                total_dist += dist

    print("Correct goal predictions: ", success)
    print("Total evaluations: ", count)
    print("total dist: ", total_dist)
    print("avg dist: ", total_dist / float(count))
    print("success rate: ", success / float(count))


if __name__ == "__main__":
    evaluate()
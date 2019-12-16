import os
import scipy as sp
import json

from data_io.models import load_model
from data_io.instructions import get_correct_eval_env_id_list
from data_io.paths import get_results_dir, get_results_path

from torch.utils.data import DataLoader

import parameters.parameter_server as P

# Supervised learning parameters
SUPERVISED_EPOCHS = 100
EVAL_INTERVAL = 10


def evaluate_top_down_pred():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]

    model, model_loaded = load_model()

    eval_envs = get_correct_eval_env_id_list()

    dataset_name = P.get_current_parameters().get("Data").get("dataset_name")
    dataset = model.get_dataset(envs=eval_envs, dataset_prefix=dataset_name, dataset_prefix="supervised", eval=eval)
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    total_loss = 0
    count = 0
    num_batches = len(dataloader)
    for b, batch in enumerate(dataloader):
        loss_var = model.sup_loss_on_batch(batch, eval=True, viz=True)
        total_loss += loss_var.data[0]
        count += 1
        print("batch: " + str(b) + " / " + str(num_batches) + \
              " loss: " + str(loss_var.data[0]))
    avg_loss = total_loss / count

    results_dir = get_results_dir(setup["run_name"])
    results_json_path = get_results_path(setup["run_name"])
    os.makedirs(results_dir, exist_ok=True)

    viz = model.get_viz()
    for key, lst in viz.items():
        for i, img in enumerate(lst):
            img_path = os.path.join(results_dir, key + str(i) + "_" + setup["model"]  + ".jpg")
            sp.misc.imsave(img_path, img)
            print("Saved image: " + img_path)

    with open(results_json_path, "w") as fp:
        json.dump({"loss": avg_loss}, fp)


if __name__ == "__main__":
    evaluate_top_down_pred()

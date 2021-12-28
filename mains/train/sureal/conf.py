import os
import shutil

from data_io.paths import get_model_dir, get_dataset_dir
import parameters.parameter_server as P


def copy_seed_dataset(from_dataset, to_dataset):
    from_path = get_dataset_dir(from_dataset)
    to_path = get_dataset_dir(to_dataset)
    if not os.path.exists(to_path):
        print(f"Copying dataset from {from_dataset} to {to_dataset}")
        shutil.copytree(from_path, to_path)
    else:
        # Don't prompt user in headless mode
        if P.get_current_parameters()["Environment"]["headless"]:
            return
        print("DATASET EXISTS! Continue?")
        c = input(" (y/n) >>> ")
        if c == "y":
            return
        else:
            raise ValueError("Not continuing: Dataset exists")


# Model file produced by RL trainer (containing both Stage 1 and Stage 2)
def epoch_rl_filename(run_name, epoch, model):
    return f"comb/{run_name}_{model}_RL_epoch_{epoch}"


# Model file produced by Stage 1
def epoch_sup_filename(run_name, epoch, model, domain="sim"):
    return f"comb/{run_name}_SUP_{model}_{domain}_{epoch}"


def rl_dataset_name(run_name):
    return f"_rl_sup_data_{run_name}"

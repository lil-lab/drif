from data_io.paths import get_model_dir, get_dataset_dir
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.models import load_model
from data_io.instructions import get_restricted_env_id_lists
from rollout.simple_parallel_rollout import SimpleParallelPolicyRoller
from rollout.rollout_sampler import RolloutSampler
import math
from evaluation.evaluate_nl import DataEvalNL

import random
import multiprocessing as mp

from learning.training.trainer_supervised import Trainer

import os
import shutil
import parameters.parameter_server as P

# ----------------------------------------------------------------------------------------------------------------
# Helpers:
# ----------------------------------------------------------------------------------------------------------------


def copy_seed_dataset(from_dataset, to_dataset, seed_count):
    from_dir = get_dataset_dir(from_dataset)
    to_dir = get_dataset_dir(to_dataset)
    if os.path.exists(to_dir):
        print("DATASET EXISTS! Continue where left off?")
        c = input(" (y/n) >>> ")
        if c == "y":
            return
        else:
            raise ValueError(f"Not continuing: Dataset {to_dataset} exists. Delete it if you like and try again")
    os.makedirs(to_dir)
    from_files = os.listdir(from_dir)

    train_ids, dev_ids, test_ids = get_restricted_env_id_lists()
    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    test_ids = set(test_ids)

    file_envs = [int(f.split("supervised_train_data_env_")[1]) for f in from_files]
    files_and_envs = list(zip(from_files, file_envs))
    random.shuffle(files_and_envs)

    files_to_copy = []
    train_envs_copied = 0
    for file, env in files_and_envs:
        if env in train_ids and train_envs_copied < seed_count:
            files_to_copy.append(file)
            if env in train_ids:
                train_envs_copied += 1

    print(f"Copying {train_envs_copied} train envs, and all dev/test envs from {from_dataset} to {to_dataset}")

    for file in files_to_copy:
        from_path = os.path.join(from_dir, file)
        to_path = os.path.join(to_dir, file)
        shutil.copy(from_path, to_path)


# Model file produced by Stage 1
def epoch_dag_filename(run_name, epoch):
    return f"dag/{run_name}_DAG_{epoch}"


def dagger_dataset_name(run_name):
    return f"_dagger_data_{run_name}"


def prune_dataset(run_name, count):
    dataset_dir = get_dataset_dir(dagger_dataset_name(run_name))
    files = os.listdir(dataset_dir)
    deleted = 0
    if len(files) > count:
        num_drop = len(files) - count
        files_to_drop = random.sample(files, num_drop)
        for file in files_to_drop:
            filepath = os.path.join(dataset_dir, file)
            os.remove(filepath)
            deleted += 1
    print(f"Deleted {deleted} files when pruning dataset {dataset_dir}")


def dagger_beta(params, iteration):
    oracle_factor = params["oracle_decay_factor"]
    return math.pow(oracle_factor, iteration)

# ----------------------------------------------------------------------------------------------------------------
# Main:
# ----------------------------------------------------------------------------------------------------------------


def train_dagger_simple():
    # ----------------------------------------------------------------------------------------------------------------
    # Load params and configure stuff

    P.initialize_experiment()
    params = P.get_current_parameters()["SimpleDagger"]
    setup = P.get_current_parameters()["Setup"]
    num_iterations = params["num_iterations"]
    sim_seed_dataset = params.get("sim_seed_dataset")
    run_name = setup["run_name"]
    device = params.get("device", "cuda:1")
    dataset_limit = params.get("dataset_size_limit_envs")
    seed_count = params.get("seed_count")

    # Trigger rebuild if necessary before going into all the threads and processes
    _ = get_restricted_env_id_lists(ignore_min_augment_len=True)

    # Initialize the dataset
    if sim_seed_dataset:
        copy_seed_dataset(from_dataset=sim_seed_dataset, to_dataset=dagger_dataset_name(run_name), seed_count=seed_count or dataset_limit)
        gap = 0
    else:
        # TODO: Refactor this into a prompt function
        data_path = get_dataset_dir(dagger_dataset_name(run_name))
        if os.path.exists(data_path):
            print("DATASET EXISTS! Continue where left off?")
            c = input(" (y/n) >>> ")
            if c != "y":
                raise ValueError(f"Not continuing: Dataset {data_path} exists. Delete it if you like and try again")
        else:
            os.makedirs(data_path, exist_ok=True)
        gap = dataset_limit - len(os.listdir(data_path))

    print("SUPP: Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # ----------------------------------------------------------------------------------------------------------------
    # Load / initialize model

    model = load_model(setup["model"], setup["model_file"], domain="sim")[0].to(device)
    oracle = load_model("oracle")[0]

    # ----------------------------------------------------------------------------------------------------------------
    # Continue where we left off - load the model and set the iteration/epoch number

    for start_iteration in range(10000):
        epfname = epoch_dag_filename(run_name, start_iteration)
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_iteration > 0:
        print(f"DAG: CONTINUING DAGGER TRAINING FROM ITERATION: {start_iteration}")
        load_pytorch_model(model, epoch_dag_filename(run_name, start_iteration - 1))

    # ----------------------------------------------------------------------------------------------------------------
    # Intialize trainer

    trainer = Trainer(model, epoch=start_iteration, name=setup["model"], run_name=setup["run_name"])
    trainer.set_dataset_names([dagger_dataset_name(run_name)])

    # ----------------------------------------------------------------------------------------------------------------
    # Initialize policy roller

    roller = SimpleParallelPolicyRoller(
        num_workers=params["num_workers"],
        device=params["device"],
        policy=model,
        oracle=oracle,
        dataset_save_name=dagger_dataset_name(run_name),
        no_reward=True)
    rollout_sampler = RolloutSampler(roller)

    # ----------------------------------------------------------------------------------------------------------------
    # Train DAgger - loop over iteartions, in each, prune, rollout and train an epoch

    print("SUPP: Beginning training...")
    for iteration in range(start_iteration, num_iterations):
        print(f"DAG: Starting iteration {iteration}")

        # Remove extra rollouts to keep within DAggerFM limit
        prune_dataset(run_name, dataset_limit)

        # Rollout and collect more data for training and evaluation
        policy_state = model.get_policy_state()
        rollout_sampler.sample_n_rollouts(
            n=gap if iteration == 0 else params["train_envs_per_iteration"],
            policy_state=policy_state,
            sample=False,
            envs="train",
            dagger_beta=dagger_beta(params, iteration))

        eval_rollouts = rollout_sampler.sample_n_rollouts(
            n=params["eval_envs_per_iteration"],
            policy_state=policy_state,
            sample=False,
            envs="dev",
            dagger_beta=0)

        # Kill airsim instances so that they don't take up GPU memory and in general slow things down during training
        roller.kill_airsim()

        # Evaluate success / metrics and save to tensorboard
        if setup["eval_nl"]:
            evaler = DataEvalNL(run_name, entire_trajectory=False, save_images=False)
            evaler.evaluate_dataset(eval_rollouts)
            results = evaler.get_results()
            print("Results:", results)
            evaler.write_summaries(setup["run_name"], "dagger_eval", iteration)

        # Do one epoch of supervised training
        print("SUPP: Beginning Epoch")
        train_loss = trainer.train_epoch(train_envs=train_envs, eval=False)
        #test_loss = trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True)

        # Save the model to file
        print("SUPP: Epoch", iteration, "train_loss:", train_loss)
        save_pytorch_model(model, epoch_dag_filename(run_name, iteration))


if __name__ == "__main__":
    #mp.set_start_method("spawn")
    train_dagger_simple()

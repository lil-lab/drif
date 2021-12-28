from data_io.paths import get_model_dir, get_dataset_dir
from data_io.model_io import save_pytorch_model, load_pytorch_model, load_model_state_dict
from data_io.models import load_model
from data_io.instructions import get_restricted_env_id_lists

from sharing_strategy import SHARING_STRATEGY
import torch.multiprocessing as mp

from learning.training.trainer_rl import TrainerRL
from learning.training.trainer_supervised import Trainer

from utils.dict_tools import dict_merge
import os
import shutil
from time import sleep

import parameters.parameter_server as P


def receive_stage1_state(conn, halt=False):
    stage1_model_state_dict = None
    if halt:
        while not conn.poll():
            print("   RLP: Waiting for Stage 1 model")
            sleep(1)
    # Grab the latest model if there are multiple on the pipe
    while conn.poll():
        msg = conn.recv()
        if msg[0] == "stage1_model_state_dict":
            stage1_model_state_dict = msg[1]
        elif msg[0] == "stage1_model_path":
            model_path = msg[1]
            return model_path
            stage1_model_state_dict = load_model_state_dict(model_path)
    return stage1_model_state_dict


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

def train_supervised_worker(rl_process_conn):
    mp.set_sharing_strategy(SHARING_STRATEGY)
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    rlsup = P.get_current_parameters()["SuReAL"]
    setup["trajectory_length"] = setup["sup_trajectory_length"]
    run_name = setup["run_name"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]
    sup_device = rlsup.get("sup_train_device")

    print("SUPP: Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # Load the starter model and save it at epoch 0
    # Supervised worker to use GPU 1, RL will use GPU 0. Simulators run on GPU 2
    model_sim = load_model(setup["sup_model"], setup["sup_model_file"], domain="sim")[0].to(sup_device)
    # ----------------------------------------------------------------------------------------------------------------

    print("SUPP: Initializing trainer")
    rlsup_params = P.get_current_parameters()["SuReAL"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")

    # Use the SuReAL temporary dataset for training stage 1 model
    trainer = Trainer(model_sim, epoch=0, name=model_sim.model_name, run_name=run_name)
    dataset_names = [rl_dataset_name(run_name)]
    trainer.set_dataset_names(dataset_names)

    # ----------------------------------------------------------------------------------------------------------------
    # Continue where left off based on present models
    for start_sup_epoch in range(10000):
        epfname = epoch_sup_filename(run_name, start_sup_epoch, model="stage1", domain="sim")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_sup_epoch == 0:
        save_pytorch_model(model_sim, epoch_sup_filename(run_name, -1, model="stage1", domain="sim"))
    if start_sup_epoch > 0:
        print(f"SUPP: CONTINUING SUP TRAINING FROM EPOCH: {start_sup_epoch}")
        load_pytorch_model(model_sim, epoch_sup_filename(run_name, start_sup_epoch - 1, model="stage1", domain="sim"))
        trainer.set_start_epoch(start_sup_epoch)

    # ----------------------------------------------------------------------------------------------------------------
    print("SUPP: Beginning training...")
    for epoch in range(start_sup_epoch, num_epochs):
        # Tell the RL process that a new Stage 1 model is ready for loading
        print("SUPP: Sending model to RL")
        latest_model_name = epoch_sup_filename(run_name, epoch-1, model="stage1", domain="sim")
        model_sim.reset()
        rl_process_conn.send(["stage1_model_path", latest_model_name])

        if not sim_seed_dataset:
            ddir = get_dataset_dir(rl_dataset_name(run_name))
            os.makedirs(ddir, exist_ok=True)
            while len(os.listdir(ddir)) < 20:
                print("SUPP: Waiting for rollouts to appear")
                sleep(3)

        print("SUPP: Beginning Epoch")
        train_loss = trainer.train_epoch(train_envs=train_envs, eval=False)
        test_loss = trainer.train_epoch(train_envs=dev_envs, eval=True)
        print ("SUPP: Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        save_pytorch_model(model_sim, epoch_sup_filename(run_name, epoch, model="stage1", domain="sim"))


def train_rl_worker(sup_process_conn):
    mp.set_sharing_strategy(SHARING_STRATEGY)
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    setup["trajectory_length"] = setup["rl_trajectory_length"]
    run_name = setup["run_name"]
    rlsup = P.get_current_parameters()["SuReAL"]
    params = P.get_current_parameters()["RL"]
    # These need to be distinguished between supervised and RL because supervised trains on ALL envs, RL only on 6000-7000
    setup["env_range_start"] = setup["rl_env_range_start"]
    setup["env_range_end"] = setup["rl_env_range_end"]
    rl_train_device = rlsup.get("rl_train_device")
    rollout_device = rlsup.get("rollout_device")

    trainer = TrainerRL(params=dict_merge(setup, params), save_rollouts_to_dataset=rl_dataset_name(run_name), train_device=rl_train_device, rollout_device=rollout_device)

    # -------------------------------------------------------------------------------------
    # TODO: Continue (including figure out how to initialize Supervised Stage 1 real/sim/critic and RL Stage 2 policy
    start_rl_epoch = 0
    for start_rl_epoch in range(10000):
        epfname = epoch_rl_filename(run_name, start_rl_epoch, model="full")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_rl_epoch > 0:
        print(f"RLP: CONTINUING RL TRAINING FROM EPOCH: {start_rl_epoch}")
        load_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, start_rl_epoch-1, model="full"))
        trainer.set_start_epoch(start_rl_epoch)
    # Wait for supervised process to send it's model
    sleep(2)

    # -------------------------------------------------------------------------------------

    print("RLP: Beginning training...")
    for rl_epoch in range(start_rl_epoch, 10000):
        # Get the latest Stage 1 model. Halt on the first epoch so that we can actually initialize the Stage 1
        new_stage1_path = receive_stage1_state(sup_process_conn, halt=(rl_epoch == start_rl_epoch))
        if new_stage1_path:
            print(f"RLP: Re-loading latest Stage 1 model")
            trainer.reload_stage1_from_path(new_stage1_path)

        train_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=False, envs="train")
        dev_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=True, envs="dev")

        print("RLP: RL Epoch", rl_epoch, "train reward:", train_reward, "dev reward:", dev_reward)
        save_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, rl_epoch, model="full"))
        save_pytorch_model(trainer.full_model.stage1_visitation_prediction, epoch_rl_filename(run_name, rl_epoch, model="stage1"))
        save_pytorch_model(trainer.full_model.stage2_action_generation, epoch_rl_filename(run_name, rl_epoch, model="stage2"))


DEBUG_SUP = False
DEBUG_RL = False


def train_sureal():
    P.initialize_experiment()
    ctx = mp.get_context("spawn")

    pipe_rl_end, pipe_sup_end = ctx.Pipe()

    rlsup_params = P.get_current_parameters()["SuReAL"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")
    run_name = P.get_current_parameters()["Setup"]["run_name"]

    # Trigger rebuild if necessary before going into all the threads and processes
    _ = get_restricted_env_id_lists()
    _ = get_restricted_env_id_lists(ignore_min_augment_len=True)

    if sim_seed_dataset:
        copy_seed_dataset(from_dataset=sim_seed_dataset, to_dataset=rl_dataset_name(run_name))

    if DEBUG_SUP:
        train_supervised_worker(pipe_sup_end)
    elif DEBUG_RL:
        train_rl_worker(pipe_rl_end)
    elif False:
        rl_process = ctx.Process(target=train_rl_worker, args=[pipe_rl_end])
        sup_proces = ctx.Process(target=train_supervised_worker, args=[pipe_sup_end])
        rl_process.start()
        sup_proces.start()
    else:
        sup_proces = ctx.Process(target=train_supervised_worker, args=[pipe_sup_end])
        sup_proces.start()
        train_rl_worker(pipe_rl_end)


if __name__ == "__main__":
    mp.set_sharing_strategy(SHARING_STRATEGY)
    try:
        mp.set_start_method('spawn')
    except Exception as e:
        print("Error setting start method", e)
    train_sureal()

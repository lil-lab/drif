from data_io.paths import get_model_dir, get_dataset_dir
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.models import load_model
from data_io.instructions import get_restricted_env_id_lists

import multiprocessing as mp
from multiprocessing import Process, Pipe

from learning.training.trainer_rl import TrainerRL
from learning.training.trainer_supervised_bidomain_bidata import TrainerBidomainBidata

from utils.dict_tools import dict_merge
import os
import shutil
from time import sleep

import parameters.parameter_server as P

DEBUG_SUP = False
DEBUG_RL = False


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
    return stage1_model_state_dict


def copy_seed_dataset(from_dataset, to_dataset):
    from_path = get_dataset_dir(from_dataset)
    to_path = get_dataset_dir(to_dataset)
    if not os.path.exists(to_path):
        print(f"Copying dataset from {from_dataset} to {to_dataset}")
        shutil.copytree(from_path, to_path)
    else:
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
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    rlsup = P.get_current_parameters()["RLSUP"]
    setup["trajectory_length"] = setup["sup_trajectory_length"]
    run_name = setup["run_name"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]
    sup_device = rlsup.get("sup_device", "cuda:1")

    model_oracle_critic = None

    print("SUPP: Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # Load the starter model and save it at epoch 0
    # Supervised worker to use GPU 1, RL will use GPU 0. Simulators run on GPU 2
    model_sim = load_model(setup["sup_model"], setup["sim_model_file"], domain="sim")[0].to(sup_device)
    model_real = load_model(setup["sup_model"], setup["real_model_file"], domain="real")[0].to(sup_device)
    model_critic = load_model(setup["sup_critic_model"], setup["critic_model_file"])[0].to(sup_device)

    # ----------------------------------------------------------------------------------------------------------------

    print("SUPP: Initializing trainer")
    rlsup_params = P.get_current_parameters()["RLSUP"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")

    # TODO: Figure if 6000 or 7000 here
    trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
    train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
    train_envs_sim = [e for e in train_envs if e < 7000]
    dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
    dev_envs_sim = [e for e in dev_envs if e < 7000]
    sim_datasets = [rl_dataset_name(run_name)]
    real_datasets = ["real"]
    trainer.set_dataset_names(sim_datasets=sim_datasets, real_datasets=real_datasets)

    # ----------------------------------------------------------------------------------------------------------------
    for start_sup_epoch in range(10000):
        epfname = epoch_sup_filename(run_name, start_sup_epoch, model="stage1", domain="sim")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_sup_epoch > 0:
        print(f"SUPP: CONTINUING SUP TRAINING FROM EPOCH: {start_sup_epoch}")
        load_pytorch_model(model_real, epoch_sup_filename(run_name, start_sup_epoch-1, model="stage1", domain="real"))
        load_pytorch_model(model_sim, epoch_sup_filename(run_name, start_sup_epoch-1, model="stage1", domain="sim"))
        load_pytorch_model(model_critic, epoch_sup_filename(run_name, start_sup_epoch-1, model="critic", domain="critic"))
        trainer.set_start_epoch(start_sup_epoch)

    # ----------------------------------------------------------------------------------------------------------------
    print("SUPP: Beginning training...")
    for epoch in range(start_sup_epoch, num_epochs):
        # Tell the RL process that a new Stage 1 model is ready for loading
        print("SUPP: Sending model to RL")
        model_sim.reset()
        rl_process_conn.send(["stage1_model_state_dict", model_sim.state_dict()])
        if DEBUG_RL:
            while True:
                sleep(1)

        if not sim_seed_dataset:
            ddir = get_dataset_dir(rl_dataset_name(run_name))
            os.makedirs(ddir, exist_ok=True)
            while len(os.listdir(ddir)) < 20:
                print("SUPP: Waiting for rollouts to appear")
                sleep(3)

        print("SUPP: Beginning Epoch")
        train_loss = trainer.train_epoch(env_list_common=train_envs_common, env_list_sim=train_envs_sim, eval=False)
        test_loss = trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True)
        print ("SUPP: Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        save_pytorch_model(model_real, epoch_sup_filename(run_name, epoch, model="stage1", domain="real"))
        save_pytorch_model(model_sim, epoch_sup_filename(run_name, epoch, model="stage1", domain="sim"))
        save_pytorch_model(model_critic, epoch_sup_filename(run_name, epoch, model="critic", domain="critic"))


def train_rl_worker(sup_process_conn):
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    setup["trajectory_length"] = setup["rl_trajectory_length"]
    run_name = setup["run_name"]
    rlsup = P.get_current_parameters()["RLSUP"]
    params = P.get_current_parameters()["RL"]
    num_rl_epochs = params["num_epochs"]
    # These need to be distinguished between supervised and RL because supervised trains on ALL envs, RL only on 6000-7000
    setup["env_range_start"] = setup["rl_env_range_start"]
    setup["env_range_end"] = setup["rl_env_range_end"]
    rl_device = rlsup.get("rl_device", "cuda:0")

    trainer = TrainerRL(params=dict_merge(setup, params), save_rollouts_to_dataset=rl_dataset_name(run_name), device=rl_device)

    # -------------------------------------------------------------------------------------
    # TODO: Continue (including figure out how to initialize Supervised Stage 1 real/sim/critic and RL Stage 2 policy
    start_rl_epoch = 0
    for start_rl_epoch in range(num_rl_epochs):
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
    for rl_epoch in range(start_rl_epoch, num_rl_epochs):
        # Get the latest Stage 1 model. Halt on the first epoch so that we can actually initialize the Stage 1
        new_stage1_model_state_dict = receive_stage1_state(sup_process_conn, halt=(rl_epoch == start_rl_epoch))
        if new_stage1_model_state_dict:
            print(f"RLP: Re-loading latest Stage 1 model")
            trainer.reload_stage1(new_stage1_model_state_dict)

        train_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=False, envs="train")
        dev_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=True, envs="dev")

        print("RLP: RL Epoch", rl_epoch, "train reward:", train_reward, "dev reward:", dev_reward)
        save_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, rl_epoch, model="full"))
        save_pytorch_model(trainer.full_model.stage1_visitation_prediction, epoch_rl_filename(run_name, rl_epoch, model="stage1"))
        save_pytorch_model(trainer.full_model.stage2_action_generation, epoch_rl_filename(run_name, rl_epoch, model="stage2"))


def train_sureal():
    P.initialize_experiment()
    ctx = mp.get_context("spawn")

    pipe_rl_end, pipe_sup_end = ctx.Pipe()

    rlsup_params = P.get_current_parameters()["RLSUP"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")
    run_name = P.get_current_parameters()["Setup"]["run_name"]

    # Trigger rebuild if necessary before going into all the threads and processes
    _ = get_restricted_env_id_lists()
    _ = get_restricted_env_id_lists(full=True)

    if sim_seed_dataset:
        copy_seed_dataset(from_dataset=sim_seed_dataset, to_dataset=rl_dataset_name(run_name))

    if DEBUG_SUP:
        train_supervised_worker(pipe_sup_end)
    elif DEBUG_RL:
        # Start supervised learning in another process. Keep RL in main process.
        sup_proces = ctx.Process(target=train_supervised_worker, args=[pipe_sup_end])
        sup_proces.start()
        train_rl_worker(pipe_rl_end)
    else:
        rl_process = ctx.Process(target=train_rl_worker, args=[pipe_rl_end])
        sup_proces = ctx.Process(target=train_supervised_worker, args=[pipe_sup_end])

        rl_process.start()
        sup_proces.start()


if __name__ == "__main__":
    train_sureal()

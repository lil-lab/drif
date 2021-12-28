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

import parameters.parameter_server as P

def send_receive_rl(conn, rl_epoch):
    conn.send(["rl_epoch", rl_epoch])
    new_supervised_epoch = None
    stage1_model = None
    if conn.poll():
        msg = conn.recv()
        if msg[0] == "supervised_epoch":
            new_supervised_epoch = msg[1]
        elif msg[0] == "stage1_model_state_dict":
            stage1_model_state_dict = msg[1]
    return new_supervised_epoch, stage1_model_state_dict

# Model file produced by RL trainer (containing both Stage 1 and Stage 2)
def epoch_rl_filename(run_name, epoch):
    return f"comb/{run_name}_RL_epoch_{epoch}"

# Model file produced by Stage 1
def epoch_sup_filename(run_name, epoch, domain="sim"):
    return f"comb/{run_name}_SUP_{domain}_{epoch}"

def rl_dataset_name(run_name):
    return f"_rl_sup_data_{run_name}"

def train_supervised_worker(rl_process_conn):
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    setup["trajectory_length"] = setup["sup_trajectory_length"]
    run_name = setup["run_name"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]

    model_oracle_critic = None

    print("SUPP: Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # Load the starter model and save it at epoch 0
    model_sim, _ = load_model(setup["sup_model"], setup["sim_model_file"], domain="sim")
    model_real, _ = load_model(setup["sup_model"], setup["real_model_file"], domain="real")
    model_critic, _ = load_model(setup["sup_critic_model"], setup["critic_model_file"])

    # Supervised worker to use GPU 1, RL will use GPU 0. Simulators run on GPU 2
    model_sim = model_sim.to("cuda:1")
    model_real = model_real.to("cuda:1")
    model_critic = model_critic.to("cuda:1")

    # ----------------------------------------------------------------------------------------------------------------

    print("SUPP: Initializing trainer")
    trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
    train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
    # TODO: Figure if 6000 or 7000 here
    train_envs_sim = [e for e in train_envs if e < 7000]
    dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
    dev_envs_sim = [e for e in dev_envs if e < 7000]

    print("SUPP: Env list sizes: ")
    print(f"SUPP:    sim train: {len(train_envs_sim)}")
    print(f"SUPP:    com train: {len(train_envs_common)}")
    print(f"SUPP:    sim dev: {len(dev_envs_sim)}")
    print(f"SUPP:    com dev: {len(dev_envs_common)}")

    # TODO: Switch to the actual real dataset when we start working with real data again
    # Simulator data comes from the joint dataset of oracle rollouts and RL rollouts
    # Real data comes from the oracle rollouts only - we can't have RL rollouts for that
    #sim_datasets = ["simulator", rl_dataset_name(run_name)]
    sim_datasets = [rl_dataset_name(run_name)]
    real_datasets = ["simulator"]
    trainer.set_dataset_names(sim_datasets=sim_datasets, real_datasets=real_datasets)

    # ----------------------------------------------------------------------------------------------------------------
    print("SUPP: Finding start epoch")
    for start_sup_epoch in range(10000):
        epfname = epoch_sup_filename(run_name, start_sup_epoch, domain="sim")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break

    if start_sup_epoch > 0:
        start_sup_epoch -= 1
        print(f"SUPP: CONTINUING SUPERVISED TRAINING FROM EPOCH: {start_sup_epoch}")
        load_pytorch_model(trainer.model_sim, epoch_sup_filename(run_name, start_sup_epoch, domain="sim"))
        load_pytorch_model(trainer.model_real, epoch_sup_filename(run_name, start_sup_epoch, domain="real"))
        load_pytorch_model(trainer.model_critic, epoch_sup_filename(run_name, start_sup_epoch, domain="critic"))
        trainer.set_start_epoch(start_sup_epoch)

    # ----------------------------------------------------------------------------------------------------------------
    print("SUPP: Beginning training...")
    best_test_loss = 1000
    for epoch in range(start_sup_epoch, num_epochs):
        save_pytorch_model(model_real, epoch_sup_filename(run_name, epoch, domain="real"))
        save_pytorch_model(model_sim, epoch_sup_filename(run_name, epoch, domain="sim"))
        save_pytorch_model(model_critic, epoch_sup_filename(run_name, epoch, domain="critic"))

        # Tell the RL process that a new Stage 1 model is ready for loading
        print("SUPP: Sending model to RL")
        rl_process_conn.send(["supervised_epoch", epoch])
        # Clear model state
        model_sim.reset()
        rl_process_conn.send(["stage1_model_state_dict", model_sim.state_dict()])
        print("SUPP: Beginning Epoch")

        train_loss = trainer.train_epoch(env_list_common=train_envs_common, env_list_sim=train_envs_sim, eval=False)
        test_loss = trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True)
        print ("SUPP: Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)


def train_rl_worker(sup_process_conn):
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    setup["trajectory_length"] = setup["rl_trajectory_length"]
    run_name = setup["run_name"]
    params = P.get_current_parameters()["RL"]
    # These need to be distinguished between supervised and RL because supervised trains on ALL envs, RL only on 6000-7000
    setup["env_range_start"] = setup["rl_env_range_start"]
    setup["env_range_end"] = setup["rl_env_range_end"]

    trainer = TrainerRL(params=dict_merge(setup, params), save_rollouts_to_dataset=rl_dataset_name(run_name))

    # -------------------------------------------------------------------------------------
    for start_rl_epoch in range(10000):
        epfname = epoch_rl_filename(run_name, start_rl_epoch)
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_rl_epoch > 0:
        print(f"RLP: CONTINUING RL TRAINING FROM EPOCH: {start_rl_epoch}")
        load_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, start_rl_epoch - 1))
        trainer.set_start_epoch(start_rl_epoch)

    # -------------------------------------------------------------------------------------

    # Get the correct Stage 1 model from supervised worker when it starts up
    print("RLP: Waiting for SUP process to send epoch...")
    _, supervised_epoch = sup_process_conn.recv()
    print(f"RLP: Starting at sup epoch: {supervised_epoch}")

    print("RLP: Beginning training...")
    for rl_epoch in range(start_rl_epoch, 10000):

        new_supervised_epoch, new_stage1_model_state_dict = send_receive_rl(sup_process_conn, rl_epoch)
        supervised_epoch = new_supervised_epoch or supervised_epoch

        if new_stage1_model_state_dict:
            print(f"RLP: Loading latest Stage 1 model for epoch {supervised_epoch}")
            print(type(new_stage1_model_state_dict))
            trainer.reload_stage1(new_stage1_model_state_dict)

        #trainer.reload_static_state_on_workers(epoch_sup_filename(run_name, supervised_epoch, domain="sim"))

        train_reward, metrics = trainer.train_epoch(eval=False, envs="train")
        dev_reward, metrics = trainer.train_epoch(eval=True, envs="dev")

        print("RLP: RL Epoch", rl_epoch, "train reward:", train_reward, "dev reward:", dev_reward)
        save_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, rl_epoch))

DEBUG_SUP = False
DEBUG_RL = False

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

def train_rl_dagger_combined():
    P.initialize_experiment()
    pipe_rl_end, pipe_sup_end = Pipe()

    rlsup_params = P.get_current_parameters()["RLSUP"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")
    run_name = P.get_current_parameters()["Setup"]["run_name"]

    copy_seed_dataset(from_dataset=sim_seed_dataset, to_dataset=rl_dataset_name(run_name))

    if DEBUG_SUP:
        train_supervised_worker(pipe_sup_end)
    elif DEBUG_RL:
        train_rl_worker(pipe_rl_end)
    else:
        rl_process = Process(target=train_rl_worker, args=[pipe_rl_end])
        sup_proces = Process(target=train_supervised_worker, args=[pipe_sup_end])

        rl_process.start()
        sup_proces.start()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_rl_dagger_combined()

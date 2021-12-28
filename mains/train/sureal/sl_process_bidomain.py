import os
import ray
from time import sleep
import multiprocessing
import sys

from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.models import load_model
from data_io.paths import get_model_dir, get_dataset_dir
from data_io.instructions import get_restricted_env_id_lists

from learning.training.trainer_supervised import Trainer
from learning.training.trainer_supervised_bidomain_bidata import TrainerBidomainBidata

from mains.train.sureal import conf

import parameters.parameter_server as P


@ray.remote(num_gpus=1, num_cpus=2)
def sureal_supervised_learning(kv_store, params_name, send_model_only=False):
    #multiprocessing.set_start_method("forkserver")
    P.initialize_experiment(params_name)
    # bad hack
    sys.argv = [sys.argv[0]] + [params_name] + sys.argv[1:]
    setup = P.get_current_parameters()["Setup"]
    rlsup = P.get_current_parameters()["SuReAL"]
    setup["trajectory_length"] = setup["sup_trajectory_length"]
    run_name = setup["run_name"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]
    sup_device = rlsup.get("sup_train_device")

    train_size_cap = P.get("Data::train_size_cap", None)
    eval_size_cap = P.get("Data::eval_size_cap", None)

    print("SUPP: Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # Load the starter model and save it at epoch 0
    # Supervised worker to use GPU 1, RL will use GPU 0. Simulators run on GPU 2
    model_sim = load_model(setup["sup_model"], setup["sim_model_file"], domain="sim")[0].to(sup_device)
    model_real = load_model(setup["sup_model"], setup["real_model_file"], domain="real")[0].to(sup_device)
    model_critic = load_model(setup["sup_critic_model"], setup["critic_model_file"])[0].to(sup_device)

    print("SUPP: Initializing trainer")
    # ----------------------------------------------------------------------------------------------------------------

    print("SUPP: Initializing trainer")
    rlsup_params = P.get_current_parameters()["SuReAL"]
    sim_seed_dataset = rlsup_params.get("sim_seed_dataset")

    # TODO: Figure if 6000 or 7000 here
    model_oracle_critic = None
    trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
    train_envs_common = [e for e in train_envs if 6000 <= e < 8000]
    train_envs_sim = [e for e in train_envs if e < 8000]
    dev_envs_common = [e for e in dev_envs if 6000 <= e < 8000]
    dev_envs_sim = [e for e in dev_envs if e < 8000]
    sim_datasets = [conf.rl_dataset_name(run_name)]
    real_datasets = ["real_full"]
    trainer.set_dataset_names(sim_datasets=sim_datasets, real_datasets=real_datasets)

    # ----------------------------------------------------------------------------------------------------------------
    for start_sup_epoch in range(10000):
        epfname = conf.epoch_sup_filename(run_name, start_sup_epoch, model="stage1", domain="sim")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_sup_epoch == 0:
        save_pytorch_model(model_real, conf.epoch_sup_filename(run_name, -1, model="stage1", domain="real"))
        save_pytorch_model(model_sim, conf.epoch_sup_filename(run_name, -1, model="stage1", domain="sim"))
        save_pytorch_model(model_critic, conf.epoch_sup_filename(run_name, -1, model="stage1", domain="critic"))
    if start_sup_epoch > 0:
        print(f"SUPP: CONTINUING SUP TRAINING FROM EPOCH: {start_sup_epoch}")
        load_pytorch_model(model_real, conf.epoch_sup_filename(run_name, start_sup_epoch - 1, model="stage1", domain="real"))
        load_pytorch_model(model_sim, conf.epoch_sup_filename(run_name, start_sup_epoch - 1, model="stage1", domain="sim"))
        load_pytorch_model(model_critic,
                           conf.epoch_sup_filename(run_name, start_sup_epoch - 1, model="critic", domain="critic"))
        trainer.set_start_epoch(start_sup_epoch)

    # ----------------------------------------------------------------------------------------------------------------
    print("SUPP: Beginning training...")
    for epoch in range(start_sup_epoch, num_epochs):
        # Tell the RL process that a new Stage 1 model is ready for loading
        print("SUPP: Sending model to RL")
        latest_model_name = conf.epoch_sup_filename(run_name, epoch-1, model="stage1", domain="sim")
        kv_store.put.remote("stage1_model_file", latest_model_name)

        # Debugging RL process: just send the model to the RL process and stop
        if send_model_only:
            return

        if not sim_seed_dataset:
            ddir = get_dataset_dir(conf.rl_dataset_name(run_name))
            os.makedirs(ddir, exist_ok=True)
            while len(os.listdir(ddir)) < 20:
                print("SUPP: Waiting for rollouts to appear")
                sleep(3)

        print("SUPP: Beginning Epoch")
        train_loss = trainer.train_epoch(env_list_common=train_envs_common, env_list_sim=train_envs_sim, eval=False, data_size_cap=train_size_cap)
        test_loss = trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True, data_size_cap=eval_size_cap)
        print ("SUPP: Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        #save_pytorch_model(trainer.full_model, epoch_rl_filename(run_name, rl_epoch, model="full"))
        save_pytorch_model(model_real, conf.epoch_sup_filename(run_name, epoch, model="stage1", domain="real"))
        save_pytorch_model(model_sim, conf.epoch_sup_filename(run_name, epoch, model="stage1", domain="sim"))
        save_pytorch_model(model_critic, conf.epoch_sup_filename(run_name, epoch, model="critic", domain="critic"))

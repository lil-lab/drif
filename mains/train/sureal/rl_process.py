import os
import ray
from time import sleep

from data_io.paths import get_model_dir, get_dataset_dir
from data_io.model_io import save_pytorch_model, load_pytorch_model, load_model_state_dict

from mains.train.sureal import conf

from learning.training.trainer_rl import TrainerRL
from utils import dict_tools

import parameters.parameter_server as P


#@ray.remote
def train_rl_worker(kv_store, params_name, save_rollouts=True):
    P.initialize_experiment(params_name)
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

    trainer = TrainerRL(params=dict_tools.dict_merge(setup, params),
                        save_rollouts_to_dataset=conf.rl_dataset_name(run_name) if save_rollouts else None,
                        train_device=rl_train_device,
                        rollout_device=rollout_device)

    # -------------------------------------------------------------------------------------
    # TODO: Continue (including figure out how to initialize Supervised Stage 1 real/sim/critic and RL Stage 2 policy
    start_rl_epoch = 0
    for start_rl_epoch in range(10000):
        epfname = conf.epoch_rl_filename(run_name, start_rl_epoch, model="full")
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break
    if start_rl_epoch > 0:
        print(f"RLP: CONTINUING RL TRAINING FROM EPOCH: {start_rl_epoch}")
        load_pytorch_model(trainer.full_model, conf.epoch_rl_filename(run_name, start_rl_epoch-1, model="full"))
        trainer.set_start_epoch(start_rl_epoch)
    # Wait for supervised process to send it's model
    sleep(2)

    # -------------------------------------------------------------------------------------

    print("RLP: Beginning training...")
    for rl_epoch in range(start_rl_epoch, 10000):
        # Get the latest Stage 1 model. Halt on the first epoch so that we can actually initialize the Stage 1
        print("RLP: Checking for new Stage 1 model")
        new_stage1_path = ray.get(kv_store.get.remote("stage1_model_file"))
        while new_stage1_path is None: # The first time around, the stage1 model file may not yet be reported
            new_stage1_path = ray.get(kv_store.get.remote("stage1_model_file"))
        print(f"RLP: Re-loading latest Stage 1 model")
        trainer.reload_stage1_from_path(new_stage1_path)

        train_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=False, envs="train")
        dev_reward, metrics = trainer.train_epoch(epoch_num=rl_epoch, eval=True, envs="dev")

        print("RLP: RL Epoch", rl_epoch, "train reward:", train_reward, "dev reward:", dev_reward)
        save_pytorch_model(trainer.full_model, conf.epoch_rl_filename(run_name, rl_epoch, model="full"))
        save_pytorch_model(trainer.full_model.stage1_visitation_prediction, conf.epoch_rl_filename(run_name, rl_epoch, model="stage1"))
        save_pytorch_model(trainer.full_model.stage2_action_generation, conf.epoch_rl_filename(run_name, rl_epoch, model="stage2"))
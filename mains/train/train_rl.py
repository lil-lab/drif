import os

from data_io.paths import get_model_dir
from learning.training.trainer_rl import TrainerRL
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.instructions import get_restricted_env_id_lists
from parameters.parameter_server import initialize_experiment, get_current_parameters

from utils.dict_tools import dict_merge

def epoch_filename(fname, epoch):
    return "tmp/" + fname + "_epoch_" + str(epoch)

# Supervised learning parameters
def train_rl():
    initialize_experiment()

    setup = get_current_parameters()["Setup"]
    params = get_current_parameters()["RL"]

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    filename = "rl_" + setup["model"] + "_" + setup["run_name"]

    trainer = TrainerRL(params=dict_merge(setup, params))

    for start_epoch in range(10000):
        epfname = epoch_filename(filename, start_epoch)
        path = os.path.join(get_model_dir(), str(epfname) + ".pytorch")
        if not os.path.exists(path):
            break

    if start_epoch > 0:
        print(f"CONTINUING RL TRAINING FROM EPOCH: {start_epoch}")
        load_pytorch_model(trainer.full_model, epoch_filename(filename, start_epoch - 1))
        trainer.set_start_epoch(start_epoch)

    print("Beginning training...")
    best_dev_reward = -1e+10
    for epoch in range(start_epoch, 10000):
        train_reward, metrics = trainer.train_epoch(eval=False, envs="train")
        # TODO: Test on just a few dev environments
        # TODO: Take most likely or mean action when testing
        dev_reward, metrics = trainer.train_epoch(eval=True, envs="dev")
        #dev_reward, metrics = trainer.train_epoch(eval=True, envs="dev")
        dev_reward = 0

        #if dev_reward >= best_dev_reward:
        #    best_dev_reward = dev_reward
        #    save_pytorch_model(trainer.full_model, filename)
        #    print("Saved model in:", filename)

        print ("Epoch", epoch, "train reward:", train_reward, "dev reward:", dev_reward)
        save_pytorch_model(trainer.full_model, epoch_filename(filename, epoch))
        if hasattr(trainer.full_model, "save"):
            trainer.full_model.save(epoch)

if __name__ == "__main__":
    train_rl()

import torch
import ray
from learning.training.trainer_supervised import Trainer
from learning.training.trainer_supervised_dataparallel import TrainerDataparallel
from data_io.train_data import file_exists
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.instructions import get_restricted_env_id_lists
from data_io.env import load_env_split

import parameters.parameter_server as P


# Supervised learning parameters
#@profile
def train_supervised(test_only=False):
    P.initialize_experiment()

    setup = P.get("Setup")
    supervised_params = P.get("Supervised")
    num_epochs = supervised_params["num_epochs"]
    test_epochs = P.get("Training::test_epochs", 1)
    train_only = P.get("Training::no_eval", False)
    dataparallel = P.get("Training::use_dataparallel")

    if dataparallel:
        TrainerClass = TrainerDataparallel
        local_ray = P.get("Setup::local_ray", False)
        #num_gpus = len(set(P.get("Training::dataparallel_device_map") + [P.get("Training::dataparallel_local_device")]))
        num_gpus = 3
        ray.init(num_cpus=P.get("Training::num_dataparallel_workers", 2) + 1,
                 num_gpus=num_gpus,
                 memory=30 * (1024 ** 3),               # 40GB for workers
                 object_store_memory=16 * (1024 ** 3),  # 10GB for object store
                 local_mode=local_ray)
    else:
        TrainerClass = Trainer

    model, model_loaded = load_model()

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    if P.get("Setup::dev_on_train", False):
        dev_envs = train_envs

    filename = "supervised_" + setup["model"] + "_" + setup["run_name"]
    if not test_only:
        start_filename = "tmp/" + filename + "_epoch_" + str(supervised_params["start_epoch"])
        if supervised_params["start_epoch"] > 0:
            if file_exists(start_filename, ""):
                print(f"Loading model for start epoch: {start_filename}")
                load_pytorch_model(model, start_filename)
            else:
                print("Couldn't continue training. Model file doesn't exist at:")
                print(start_filename)
                exit(-1)

    trainer = TrainerClass(model, epoch=supervised_params["start_epoch"], name=setup["model"], run_name=setup["run_name"])

    print("Beginning training...")
    best_test_loss = 1000
    for epoch in range(num_epochs):
        if not test_only:
            train_loss = trainer.train_epoch(train_data=None, train_envs=train_envs, eval=False)
        else:
            train_loss = 0
        if not train_only:
            test_loss = trainer.train_epoch(train_data=None, train_envs=dev_envs, eval=True)
        else:
            test_loss = 0

        if not test_only:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_pytorch_model(trainer.model, filename)
                print("Saved model in:", filename)
            print("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
            save_pytorch_model(trainer.model, "tmp/" + filename + "_epoch_" + str(epoch))
            if hasattr(trainer.model, "save"):
                trainer.model.save(epoch)
            save_pretrained_weights(trainer.model, setup["run_name"])
        if test_only and epoch >= test_epochs - 1:
            break
    return test_loss, train_loss


if __name__ == "__main__":
    #torch.multiprocessing.set_sharing_strategy('file_system')
    train_supervised()

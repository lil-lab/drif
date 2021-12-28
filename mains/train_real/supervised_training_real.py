import os, sys

from drones.train_real.trainer_real import PerceptionTrainer
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.paths import get_results_dir, get_results_path
from data_io.instructions import get_env_id_lists_perception
from utils.simple_profiler import SimpleProfiler
from torch.utils.data.dataloader import DataLoader
from drones.train_real.trainer_critic import CriticTrainer

import parameters.parameter_server as P


def train_supervised():

    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    start_epoch = P.get_current_parameters()["Supervised"]["start_epoch"]
    num_epochs = P.get_current_parameters()["Supervised"]["num_epochs"]

    # inspired on train_top_down
    model, model_loaded = load_model(real=True)
    train_envs, dev_envs, test_envs = get_env_id_lists_perception(max_envs=setup["max_envs"])
    #if real_sim:
    #    train_envs_sim, dev_envs_sim, test_envs_sim = get_env_id_lists_perception(max_envs=args.max_envs, real=False)

    trainer = PerceptionTrainer(
        model,
        epoch=start_epoch,
        name=setup["model"],
        run_name=setup["run_name"])

    filename = setup["run_name"]

    print("Beginning training...")
    print("Real training environments: {}".format(train_envs))
    print("Real eval environments: {}".format(dev_envs))

    PROFILE2 = False
    prof = SimpleProfiler(torch_sync=PROFILE2, print=PROFILE2)
    prof.tick("out")
    dataset_name = P.get_current_parameters().get("Data").get("dataset_name")
    train_dataset = trainer.model.get_dataset(dataset_names=dataset_name, dataset_prefix="train", envs=train_envs, eval=False)

    prof.tick("train dataset")

    # Dataloader for training data
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=trainer.batch_size,
        shuffle=True,
        num_workers=trainer.num_loaders,
        pin_memory=False,
        timeout=0,
        drop_last=False)

    prof.tick("train dataloader")

    eval_dataset = trainer.model.get_dataset(dataset_names=dataset_name, dataset_prefix="eval", envs=dev_envs, eval=True)

    prof.tick("eval dataset")

    # Dataloader for eval data
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        batch_size=trainer.batch_size,
        shuffle=True,
        num_workers=trainer.num_loaders,
        pin_memory=False,
        timeout=0,
        drop_last=False)
    prof.tick("eval dataloader")
    prof.print_stats()

    PROFILE = False
    prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
    prof.tick("out")

    # if we use wasserstein loss, then the discriminator is a critic network
    # it is trained differently than usual discriminators (cf paper Wassertein GAN)
    # The critic's training is made in a different trainer for practical reasons (as poosed to non Wass. GAN where
    # the training is made in the same trainer ad the feature extractor).
    if model.wasserstein:
        critic_trainer = CriticTrainer(train_dataloader, model, epoch=start_epoch)

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_dataloader, eval=False, critic_trainer=critic_trainer)
        prof.tick("training")
        test_loss = trainer.train_epoch(eval_dataloader, eval=True)
        prof.tick("evaluation")
        #writer.add_dict(prefix, get_current_meters(), iter)
        #iter += 1
        if model.wasserstein:
            critic_trainer.inc_epoch()

        if epoch == 0:
            best_test_loss = test_loss + 1
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_pytorch_model(trainer.model, filename)
            print("Saved model in:", filename)
            prof.tick("saving model")

        print("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        save_pytorch_model(trainer.model, "tmp/" + filename + "_epoch_" + str(epoch))
        prof.tick("saving model in tmp")

        if hasattr(trainer.model, "save"):
            trainer.model.save(epoch)
            prof.tick("save model trainer")
        trainer.model.inc_iter()
        prof.print_stats()
    save_pretrained_weights(trainer.model, setup["run_name"])
    results_dir = get_results_dir(setup["run_name"])
    results_json_path = get_results_path(setup["run_name"])
    os.makedirs(results_dir, exist_ok=True)


if __name__ == "__main__":
    train_supervised()

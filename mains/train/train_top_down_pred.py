from learning.training.trainer_supervised import Trainer
from deprecated.parser import parse_args
from data_io.models import load_model
from data_io.model_io import save_pytorch_model
from data_io.instructions import get_restricted_env_id_lists
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.paths import get_model_dir
import pandas as pd
from multiprocessing import set_start_method

from parameters.parameter_server import initialize_experiment

# Supervised learning parameters
SUPERVISED_EPOCHS = 200
EVAL_INTERVAL = 10
BATCH_SIZE = 10


def train_top_down_pred(args, max_epoch=SUPERVISED_EPOCHS):
    initialize_experiment(args.run_name, args.setup_name)

    model, model_loaded = load_model()

    # TODO: Get batch size from global parameter server when it exists
    batch_size = 1 if \
        args.model == "top_down" or \
        args.model == "top_down_prior" or \
        args.model == "top_down_sm" or \
        args.model == "top_down_pretrain" or \
        args.model == "top_down_goal_pretrain" or \
        args.model == "top_down_nav" or \
        args.model == "top_down_cond" \
        else BATCH_SIZE

    lr = 0.001# * batch_size
    trainer = Trainer(model, epoch=args.start_epoch, name=args.model, run_name=args.run_name)

    train_envs, dev_envs, test_envs = get_restricted_env_id_lists(max_envs=args.max_envs)

    filename = "top_down_" + args.model + "_" + args.run_name

    if args.restore_weights_name is not None:
        restore_pretrained_weights(model, args.restore_weights_name, args.fix_restored_weights)

    print("Beginning training...")
    best_test_loss = 1000

    validation_loss = []

    for epoch in range(SUPERVISED_EPOCHS):
        train_loss = -1

        if not args.eval_pretrain:
            train_loss = trainer.train_epoch(train_envs=train_envs, eval=False)

        test_loss = trainer.train_epoch(train_envs=dev_envs, eval=True)
        validation_loss.append([epoch, test_loss])

        if not args.eval_pretrain:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_pytorch_model(trainer.model, filename)
                print("Saved model in:", filename)

            print ("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
            save_pytorch_model(trainer.model, "tmp/" + filename + "_epoch_" + str(epoch))
            save_pretrained_weights(trainer.model, args.run_name)

        else:
            break

        if max_epoch is not None and epoch > max_epoch:
            print ("Reached epoch limit!")
            break

    test_loss_dir = get_model_dir() + "/test_loss/" +filename + "_test_loss.csv"
    validation_loss = pd.DataFrame(validation_loss, columns=['epoch', "test_loss"])
    validation_loss.to_csv(test_loss_dir, index=False)

if __name__ == "__main__":
    args = parse_args()
    # Workaround for freezing DataLoader
    set_start_method('spawn', force=True)
    train_top_down_pred(args)

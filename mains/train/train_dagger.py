import gc
import os
import random

import data_io.instructions
import data_io.model_io
import data_io.train_data
import data_io.env
import data_io.models
from evaluation.evaluate_t_landmark_side import DataEvalLandmarkSide
from evaluation.evaluate_nl import DataEvalNL
from data_io.helpers import save_json
from data_io.train_data import filter_env_list_has_data
from learning.training.train_supervised import Trainer
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams, RolloutStrategy
from data_io.models import load_model

import parameters.parameter_server as P

# Dagger parameters
PARAMS = None


def get_model_filename_at_iteration(setup, i):
    dagger_filename = "dagger_" + setup["model"] + "_" + setup["run_name"]
    return "dagger/" + dagger_filename + "_iteration_" + str(i)


def rollout_on_env_set(roller, env_list, iteration, model_filename, test=False):
    setup = P.get_current_parameters()["Setup"]

    roll_out_params = RollOutParams() \
        .setModelName(setup["model"]) \
        .setModelFile(model_filename) \
        .setRunName(setup["run_name"]) \
        .setSetupName(P.get_setup_name()) \
        .setWriteSummaries(False) \
        .setBuildTrainData(False) \
        .setEnvList(env_list) \
        .setMaxDeviation(PARAMS["max_deviation"]) \
        .setSavePlots(True) \
        .setShowAction(False) \
        .setSegmentLevel(PARAMS["segment_level"]) \
        .setPlotDir("dagger_" + setup["run_name"] + "/" + "iteration_" + str(iteration)) \
        .setCuda(setup["cuda"]) \
        .setFlag("dagger_" + str(iteration))

    # If we are training, use a mixture policy for experience collection
    if not test:
        # At each timestep, execute oracle action with probability expert_prob
        # The average of these must supposedly go to 0, so we raise to a power a bit bigger than 1
        expert_prob = PARAMS["oracle_discount_factor"] ** (iteration + 1)
        # expert_prob = 0
        roll_out_params.setRolloutStrategy(RolloutStrategy.MIXTURE) \
            .setMixtureReferenceProbability(expert_prob) \
            .setHorizon(100)

    data_i = roller.roll_out_policy(roll_out_params)
    return data_i


def sample_n_from_list(list, n):
    indices = random.sample(range(0, len(list)), min(n, len(list)))
    sublist = [list[i] for i in indices]
    return sublist


def load_dagger_model(latest_model_filename):
    setup = P.get_current_parameters()["Setup"]
    # Load and re-save the model to the continuously updated dagger filename
    if PARAMS["restore_latest"]:
        print("Loading latest model: ", latest_model_filename)
        model, model_loaded = load_model(model_file_override=latest_model_filename)
    elif PARAMS["restore"] == 0 or setup["restore_data_only"]:
        model, model_loaded = load_model()
    elif setup["restore"] > 0 and not setup["restore_data_only"]:
        model_name = get_model_filename_at_iteration(setup, setup["dagger_restore"] - 1)
        model, model_loaded = load_model(model_file_override=model_name)
    return model


def restore_data(dagger_data_dir, all_train_data, all_test_data):
    # Roll forward
    for i in range(PARAMS["restore"]):
        print("Restoring dagger data: " + str(i))
        train_data_i = data_io.train_data.load_dataset(dagger_data_dir + "train_" + str(i))
        all_train_data += train_data_i
        try:
            test_data_i = data_io.train_data.load_dataset(dagger_data_dir + "test_" + str(i))
            all_test_data += test_data_i
        except:
            print("Error re-loading test data")


def restore_data_latest(dagger_data_dir):
    train_data_i = data_io.train_data.load_dataset(dagger_data_dir + "train_latest")
    test_data_i = data_io.train_data.load_dataset(dagger_data_dir + "test_latest")
    return train_data_i, test_data_i


def collect_iteration_data(roller, iteration, train_envs, test_envs, latest_model_filename, dagger_data_dir):
    setup = P.get_current_parameters()["Setup"]
    # Collect data with current policy
    num_train_samples = PARAMS["train_envs_per_iteration_override"][iteration] if iteration in \
                        PARAMS["train_envs_per_iteration_override"] else PARAMS["train_envs_per_iteration"]

    train_envs_i = sample_n_from_list(train_envs, num_train_samples)
    if PARAMS["test_on_train"]:
        test_envs_i = train_envs_i
    else:
        test_envs_i = sample_n_from_list(test_envs, PARAMS["test_envs_per_iteration"])

    train_data_i = rollout_on_env_set(roller, train_envs_i, iteration, latest_model_filename, test=False)
    test_data_i = rollout_on_env_set(roller, test_envs_i, iteration, latest_model_filename, test=True)

    if setup["eval_landmark_side"]:
        evaler = DataEvalLandmarkSide(setup["run_name"], save_images=False)
        evaler.evaluate_dataset(test_data_i)
        results = evaler.get_results()
        print("Results:", results)
        evaler.write_summaries(setup["run_name"], "dagger_eval", iteration)
    if setup["eval_nl"]:
        evaler = DataEvalNL(setup["run_name"], entire_trajectory=not PARAMS["segment_level"])
        evaler.evaluate_dataset(test_data_i)
        results = evaler.get_results()
        print("Results:", results)
        evaler.write_summaries(setup["run_name"], "dagger_eval", iteration)
    #TODO: Complete

    # Kill the simulators after each rollout to save CPU cycles and avoid the slowdown
    os.system("killall -9 MyProject5-Linux-Shipping")
    os.system("killall -9 MyProject5")

    #save_json(train_summary, dagger_data_dir + "dagger_train_summary_" + str(iteration) + ".json")
    #save_json(test_summary, dagger_data_dir + "dagger_test_summary_" + str(iteration) + ".json")
    data_io.train_data.save_dataset(train_data_i, dagger_data_dir + "train_" + str(iteration))
    data_io.train_data.save_dataset(test_data_i, dagger_data_dir + "test_" + str(iteration))

    return train_data_i, test_data_i


def load_latest_model(latest_model_filename):
    # If we retrain every iteration, don't load previously trained model, but train from scratch
    setup = P.get_current_parameters()["Setup"]
    if PARAMS["retrain_every_iteration"]:
        model, model_loaded = load_model(model_file_override="reset")
    # Otherwise load the latest model
    else:
        model, model_loaded = load_model(model_file_override=latest_model_filename)
    return model, model_loaded


def resample_supervised_data(train_data, all_env_ids):
    """
    :param train_data: list of training data, where integers represent supervised data environments and the rest are lists of samples
    :param all_env_ids: the env id's from which to resample the integers representing supervised data
    :return: Nothing, the list is modified in place.
    """
    # All the integers that have been mixed in with the training data represent supervised data
    eligible_env_ids = filter_env_list_has_data(all_env_ids, "supervised")
    current_supervised_envs = [env_id for env_id in train_data if type(env_id) is int]
    new_supervised_envs = random.sample(eligible_env_ids, len(current_supervised_envs))
    j = 0
    for i, env in enumerate(train_data):
        if type(env) is int:
            train_data[i] = new_supervised_envs[j]
            j += 1


def train_dagger():
    P.initialize_experiment()
    global PARAMS
    PARAMS = P.get_current_parameters()["Dagger"]
    setup = P.get_current_parameters()["Setup"]

    if setup["num_workers"] > 1:
        roller = ParallelPolicyRoller(num_workers=setup["num_workers"], first_worker=setup["first_worker"], reduce=PARAMS["segment_level"])
    else:
        roller = PolicyRoller()

    latest_model_filename = "dagger_" + setup["model"] + "_" + setup["run_name"]
    dagger_data_dir = "dagger_data/" + setup["run_name"] + "/"

    save_json(PARAMS, dagger_data_dir + "run_params.json")

    # Load less tf data, but sample dagger rollouts from more environments to avoid overfitting.
    train_envs, dev_envs, test_envs = data_io.instructions.get_all_env_id_lists(max_envs=PARAMS["max_envs_dag"])

    if PARAMS["resample_supervised_data"]:
        # Supervised data are represented as integers that will be later loaded by the dataset
        all_train_data = list(range(PARAMS["max_samples_in_memory"]))
        all_test_data = list(range(0))
    else:
        all_train_data, all_test_data = data_io.train_data.load_supervised_data(max_envs=PARAMS["max_envs_sup"], split_segments=PARAMS["segment_level"])

    resample_supervised_data(all_train_data, train_envs)
    resample_supervised_data(all_test_data, test_envs)

    print("Loaded tf data size: " + str(len(all_train_data)) + " : " + str(len(all_test_data)))

    model = load_dagger_model(latest_model_filename)
    data_io.model_io.save_pytorch_model(model, latest_model_filename)

    if PARAMS["restore_latest"]:
        all_train_data, all_test_data = restore_data_latest(dagger_data_dir)
    else:
        restore_data(dagger_data_dir, all_train_data, all_test_data)

    last_trainer_state = None

    for iteration in range(PARAMS["restore"], PARAMS["max_iterations"]):
        gc.collect()
        print("-------------------------------")
        print("DAGGER ITERATION : ", iteration)
        print("-------------------------------")

        test_data_i = all_test_data

        # If we have too many training examples in memory, discard uniformly at random to keep a somewhat fixed bound
        max_samples = PARAMS["max_samples_in_memory"]
        if max_samples > 0 and len(all_train_data) > max_samples:# and iteration != args.dagger_restore:
            num_discard = len(all_train_data) - max_samples
            print("Too many samples in memory! Dropping " + str(num_discard) + " samples")
            discards = set(random.sample(list(range(len(all_train_data))), num_discard))
            all_train_data = [sample for i, sample in enumerate(all_train_data) if i not in discards]
            print("Now left " + str(len(all_train_data)) + " samples")

        # Roll out new data at iteration i, except if we are restoring to that iteration, in which case we already have data
        if iteration != PARAMS["restore"] or iteration == 0:
            train_data_i, test_data_i = collect_iteration_data(roller, iteration, train_envs, test_envs, latest_model_filename, dagger_data_dir)

            # Aggregate the dataset
            all_train_data += train_data_i
            all_test_data += test_data_i
            print("Aggregated dataset!)")
            print("Total samples: ", len(all_train_data))
            print("New samples: ", len(train_data_i))

        data_io.train_data.save_dataset(all_train_data, dagger_data_dir + "train_latest")
        data_io.train_data.save_dataset(test_data_i, dagger_data_dir + "test_latest")

        model, model_loaded = load_latest_model(latest_model_filename)

        trainer = Trainer(model, state=last_trainer_state)

        import rollout.run_metadata as run_md
        run_md.IS_ROLLOUT = False

        # Train on the newly aggregated dataset
        num_epochs = PARAMS["epochs_per_iteration_override"][iteration] if iteration in PARAMS["epochs_per_iteration_override"] else PARAMS["epochs_per_iteration"]
        for epoch in range(num_epochs):

            # Get a random sample of all test data for calculating eval loss
            #epoch_test_sample = sample_n_from_list(all_test_data, PARAMS["num_test_samples"])
            # Just evaluate on the latest test data
            epoch_test_sample = test_data_i

            loss = trainer.train_epoch(all_train_data)
            test_loss = trainer.train_epoch(epoch_test_sample, eval=True)

            data_io.model_io.save_pytorch_model(trainer.model, latest_model_filename)
            print("Epoch", epoch, "Loss: Train:", loss, "Test:", test_loss)

        data_io.model_io.save_pytorch_model(trainer.model, get_model_filename_at_iteration(setup, iteration))
        if hasattr(trainer.model, "save"):
            trainer.model.save("dag" + str(iteration))
        last_trainer_state = trainer.get_state()


if __name__ == "__main__":
    train_dagger()


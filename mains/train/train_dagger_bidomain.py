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
from learning.training.trainer_supervised import Trainer
from learning.training.trainer_supervised_bidomain import TrainerBidomain
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams, RolloutStrategy
from data_io.models import load_model

import parameters.parameter_server as P

# Dagger parameters
PARAMS = None

# ---------------------------------------------------------------------
# IO

def get_dagger_data_dir(setup, real_drone):
    return f"dagger_data/{setup['run_name']}/{'real' if real_drone else 'simulator'}/"

def get_latest_model_filename(setup, suffix):
    return f"dagger_{setup['model']}_{setup['run_name']}_{suffix}"

def get_model_filename(setup, suffix):
    return f"dagger_{setup['model']}_{setup['model']}_{setup['run_name']}_{suffix}"

def get_model_filename_at_iteration(setup, i, suffix):
    return f"dagger/{get_model_filename(setup, suffix)}_iter_{i}"

'''
def load_dagger_model(latest_model_filename, suffix):
    setup = P.get_current_parameters()["Setup"]
    # Load and re-save the model to the continuously updated dagger filename
    if PARAMS["restore_latest"]:
        print("Loading latest model: ", latest_model_filename)
        model, model_loaded = load_model(model_file_override=latest_model_filename)
    elif PARAMS["restore"] == 0:
        model, model_loaded = load_model()
    elif PARAMS["restore"]:
        model_name = get_model_filename_at_iteration(setup, PARAMS["restore"] - 1, sufffix)
        model, model_loaded = load_model(model_file_override=model_name)
    return model
'''

def restore_data(dataset_name, dagger_data_dir, all_train_data, all_test_data):
    # TODO: Adapt this to bi-domain
    # Roll forward
    for i in range(PARAMS["restore"]):
        print("Restoring dagger data: " + str(i))
        train_data_i = data_io.train_data.load_dataset(dataset_name, dagger_data_dir + "train_" + str(i))
        all_train_data += train_data_i
        try:
            test_data_i = data_io.train_data.load_dataset(dataset_name, dagger_data_dir + "test_" + str(i))
            all_test_data += test_data_i
        except:
            print("Error re-loading test data")

def restore_data_latest(dagger_data_dir, dataset_name):
    train_data_i = data_io.train_data.load_dataset(dataset_name, dagger_data_dir + "train_latest")
    test_data_i = data_io.train_data.load_dataset(dataset_name, dagger_data_dir + "test_latest")
    return train_data_i, test_data_i

def pick_policy_roller(setup):
    if setup["num_workers"] > 1:
        roller = ParallelPolicyRoller(num_workers=setup["num_workers"], first_worker=setup["first_worker"], reduce=PARAMS["segment_level"])
    else:
        roller = PolicyRoller()
    return roller

# ---------------------------------------------------------------------
# Rollouts

def rollout_on_env_set(roller, env_list, iteration, model_filename, test=False, real_drone=False):
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
        .setSavePlots(False) \
        .setRealDrone(real_drone) \
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

def collect_iteration_data(roller, iteration, train_envs, test_envs, latest_model_filename):
    setup = P.get_current_parameters()["Setup"]

    # Collect data with current policy
    num_train_samples = PARAMS["train_envs_per_iteration"]
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
        evaler = DataEvalNL(setup["run_name"], entire_trajectory=not PARAMS["segment_level"], save_images=False)
        evaler.evaluate_dataset(test_data_i)
        results = evaler.get_results()
        print("Results:", results)
        evaler.write_summaries(setup["run_name"], "dagger_eval", iteration)

    # Kill the simulators after each rollout to save CPU cycles and avoid the slowdown
    os.system("killall -9 MyProject5-Linux-Shipping")
    os.system("killall -9 MyProject5")

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


def discard_if_too_many(data_list, max_samples):
    if len(data_list) > max_samples > 0:  # and iteration != args.dagger_restore:
        num_discard = len(data_list) - max_samples
        print("Too many samples in memory! Dropping " + str(num_discard) + " samples")
        discards = set(random.sample(list(range(len(data_list))), num_discard))
        all_train_data = [sample for i, sample in enumerate(data_list) if i not in discards]
        print("Now left " + str(len(all_train_data)) + " samples")
    return data_list


def train_dagger():
    P.initialize_experiment()
    global PARAMS
    PARAMS = P.get_current_parameters()["Dagger"]
    setup = P.get_current_parameters()["Setup"]
    roller = pick_policy_roller(setup)

    save_json(PARAMS, get_dagger_data_dir(setup, real_drone=False) + "run_params.json")

    # Load less tf data, but sample dagger rollouts from more environments to avoid overfitting.
    train_envs, dev_envs, test_envs = data_io.instructions.get_restricted_env_id_lists(max_envs=PARAMS["max_envs_dag"])

    all_train_data_real, all_dev_data_real = \
        data_io.train_data.load_supervised_data("real", max_envs=PARAMS["max_envs_sup"], split_segments=PARAMS["segment_level"])
    all_train_data_sim, all_dev_data_sim = \
        data_io.train_data.load_supervised_data("simulator", max_envs=PARAMS["max_envs_sup"], split_segments=PARAMS["segment_level"])

    print("Loaded data: ")
    print(f"   Real train {len(all_train_data_real)}, dev {len(all_dev_data_real)}")
    print(f"   Sim train {len(all_train_data_sim)}, dev {len(all_dev_data_sim)}")

    # Load and re-save models from supervised learning stage
    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
    model_real, _ = load_model(setup["model"], setup["real_model_file"], domain="real")
    model_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])
    data_io.model_io.save_pytorch_model(model_sim, get_latest_model_filename(setup, "sim"))
    data_io.model_io.save_pytorch_model(model_real, get_latest_model_filename(setup, "real"))
    data_io.model_io.save_pytorch_model(model_critic, get_latest_model_filename(setup, "critic"))

    last_trainer_state = None

    for iteration in range(0, PARAMS["max_iterations"]):
        gc.collect()
        print("-------------------------------")
        print("DAGGER ITERATION : ", iteration)
        print("-------------------------------")

        # If we have too many training examples in memory, discard uniformly at random to keep a somewhat fixed bound
        max_samples = PARAMS["max_samples_in_memory"]
        all_train_data_real = discard_if_too_many(all_train_data_real, max_samples)
        all_train_data_sim = discard_if_too_many(all_train_data_sim, max_samples)

        # Roll out new data in simulation only
        latest_model_filename_sim = get_latest_model_filename(setup, "sim")
        train_data_i_sim, dev_data_i_sim = collect_iteration_data(roller, iteration, train_envs, test_envs, latest_model_filename_sim)

        # TODO: Save
        #data_io.train_data.save_dataset(dataset_name, train_data_i, dagger_data_dir + "train_" + str(iteration))
        #data_io.train_data.save_dataset(dataset_name, test_data_i, dagger_data_dir + "test_" + str(iteration))

        # Aggregate the dataset
        all_train_data_sim += train_data_i_sim
        all_dev_data_sim += dev_data_i_sim
        print("Aggregated dataset!)")
        print("Total samples: ", len(all_train_data_sim))
        print("New samples: ", len(train_data_i_sim))

        data_io.train_data.save_dataset("sim_dagger", all_train_data_sim, get_dagger_data_dir(setup, False) + "train_latest")
        data_io.train_data.save_dataset("sim_dagger", dev_data_i_sim, get_dagger_data_dir(setup, False) + "test_latest")

        model_sim, _ = load_model(setup["model"], get_latest_model_filename(setup, "sim"), domain="sim")
        model_real, _ = load_model(setup["model"], get_latest_model_filename(setup, "real"), domain="real")
        model_critic, _ = load_model(setup["critic_model"], get_latest_model_filename(setup, "critic"))

        trainer = TrainerBidomain(model_real, model_sim, model_critic, state=last_trainer_state)

        # Hacky reset of the rollout flag after doing the rollouts
        import rollout.run_metadata as run_md
        run_md.IS_ROLLOUT = False

        # Train on the newly aggregated dataset
        num_epochs = PARAMS["epochs_per_iteration"]
        for epoch in range(num_epochs):

            loss = trainer.train_epoch(data_list_real=all_train_data_real, data_list_sim=all_train_data_sim)
            dev_loss = trainer.train_epoch(data_list_real=all_dev_data_real, data_list_sim=dev_data_i_sim, eval=True)

            data_io.model_io.save_pytorch_model(model_sim, get_latest_model_filename(setup, "sim"))
            data_io.model_io.save_pytorch_model(model_real, get_latest_model_filename(setup, "real"))
            data_io.model_io.save_pytorch_model(model_critic, get_latest_model_filename(setup, "critic"))

            print("Epoch", epoch, "Loss: Train:", loss, "Test:", dev_loss)

        data_io.model_io.save_pytorch_model(model_real, get_model_filename_at_iteration(setup, iteration, "real"))
        data_io.model_io.save_pytorch_model(model_sim, get_model_filename_at_iteration(setup, iteration, "sim"))
        data_io.model_io.save_pytorch_model(model_critic, get_model_filename_at_iteration(setup, iteration, "critic"))

        last_trainer_state = trainer.get_state()


if __name__ == "__main__":
    train_dagger()


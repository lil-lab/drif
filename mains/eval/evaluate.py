import os
import shutil
import multiprocessing

from evaluation.evaluate_t_landmark_side import DataEvalLandmarkSide
from evaluation.evaluate_nl import DataEvalNL
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams
from data_io.weights import restore_pretrained_weights
from data_io.instructions import get_correct_eval_env_id_list
from data_io.models import load_model
import data_io.paths
from data_io.train_data import load_dataset_from_path, save_dataset_to_path, load_multiple_env_data_from_dir

import parameters.parameter_server as P


def query_user_load_discard(pth):
    print(f"Dataset exists at: {pth}")
    print("You have options:")
    print("Load this dataset and continue evaluation (Y/y)")
    print("Discard dataset and start new evaluation  (D/d)")
    print("Cancel (N/n/C/c)")
    while True:
        char = input(">>>>")
        if char in ["Y", "y"]:
            return "load"
        elif char in ["D", "d"]:
            return "discard"
        elif char in ["N", "n", "C", "c"]:
            return "cancel"
        else:
            print(f"Unrecognized input: {char}")


def evaluate():
    P.initialize_experiment()
    params = P.get_current_parameters()
    setup = params["Setup"]

    models = []
    for i in range(setup["num_workers"]):
        model, model_loaded = load_model()
        models.append(model)

    eval_envs = list(sorted(get_correct_eval_env_id_list()))

    round_size = P.get_current_parameters()["Data"].get("collect_n_at_a_time")

    # TODO: Scrap RollOutParams and use parameter server JSON params instead
    roll_out_params = RollOutParams() \
                        .setModelName(setup["model"]) \
                        .setModelFile(setup["model_file"]) \
                        .setRunName(setup["run_name"]) \
                        .setSetupName(P.get_setup_name()) \
                        .setEnvList(eval_envs) \
                        .setMaxDeviation(800) \
                        .setHorizon(setup["trajectory_length"]) \
                        .setStepsToForceStop(20) \
                        .setPlot(False) \
                        .setShowAction(False) \
                        .setIgnorePolicyStop(False) \
                        .setPlotDir("evaluate/" + setup["run_name"]) \
                        .setSavePlots(False) \
                        .setRealtimeFirstPerson(False) \
                        .setSaveSamples(False) \
                        .setBuildTrainData(False) \
                        .setSegmentReset("always") \
                        .setSegmentLevel(False) \
                        .setFirstSegmentOnly(False) \
                        .setDebug(setup["debug"]) \
                        .setCuda(setup["cuda"]) \
                        .setRealDrone(setup["real_drone"])

    custom_eval = "Eval" in params and params["Eval"]["custom_eval"]
    instructions = None
    if custom_eval:
        examples = params["Eval"]["examples"]
        eval_envs, eval_sets, eval_segs, instructions = tuple(map(lambda m: list(m), list(zip(*examples))))
        print("!! Running custom evaluation with the following setup:")
        print(examples)
        roll_out_params.setEnvList(eval_envs)
        roll_out_params.setSegList(eval_segs)
        roll_out_params.setCustomInstructions(instructions)

    if setup["num_workers"] > 1:
        roller = ParallelPolicyRoller(num_workers=setup["num_workers"])
    else:
        roller = PolicyRoller()

    if round_size:
        eval_dataset_name = data_io.paths.get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
        eval_dataset_path = data_io.paths.get_dataset_dir(eval_dataset_name)

        cumulative_dataset = []
        if os.path.exists(eval_dataset_path):
            result = query_user_load_discard(eval_dataset_path)
            if result == "load":
                print("Loading dataset and continuing evaluation")
                cumulative_dataset = load_multiple_env_data_from_dir(eval_dataset_path)
            elif result == "discard":
                print("Discarding existing evaluation data")
                shutil.rmtree(eval_dataset_path)
            elif result == "cancel":
                print("Cancelling evaluation")
                return

        os.makedirs(eval_dataset_path, exist_ok=True)

        collected_envs = set([rollout[0]["env_id"] for rollout in cumulative_dataset if len(rollout) > 0])
        eval_envs = [e for e in eval_envs if e not in collected_envs]
        if setup.get("compute_results_no_rollout", False):
            eval_envs = []

        for i in range(0, len(eval_envs), round_size):
            j = min(len(eval_envs), i + round_size)
            round_envs = eval_envs[i:j]
            roll_out_params.setEnvList(round_envs)
            dataset = roller.roll_out_policy(roll_out_params)

            # Save this data
            for rollout in dataset:
                if len(rollout) == 0:
                    print("WARNING! DROPPING EMPTY ROLLOUTS! SHOULDN'T DO THIS")
                    continue
                ## rollout is a list of samples:
                env_id = rollout[0]["env_id"] if "metadata" in rollout[0] else rollout[0]["env_id"]
                if True:
                    if len(rollout) > 0:
                        save_dataset_to_path(os.path.join(eval_dataset_path, str(env_id)), rollout)
                ## rollout is a list of segments, each is a list of samples
                else:
                    if len(rollout) > 0:
                        save_dataset_to_path(os.path.join(eval_dataset_path, str(env_id)), rollout)

            cumulative_dataset += dataset
            print(f"Saved cumulative dataset to: {eval_dataset_path}")

        dataset = cumulative_dataset
    else:
        dataset = roller.roll_out_policy(roll_out_params)

    results = {}
    if setup["eval_landmark_side"]:
        evaler = DataEvalLandmarkSide(setup["run_name"], save_images=True, world_size=setup["world_size_m"])
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()
    if setup["eval_nl"]:
        evaler = DataEvalNL(setup["run_name"], save_images=True, entire_trajectory=False, custom_instr=instructions)
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()

    print("Results:", results)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    evaluate()
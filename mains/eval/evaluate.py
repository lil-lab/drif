import os
import shutil
import ray
import multiprocessing

from evaluation.evaluate_t_landmark_side import DataEvalLandmarkSide
from evaluation.evaluate_nl import DataEvalNL
#from rollout.parallel_roll_out import ParallelPolicyRoller
#from rollout.roll_out import PolicyRoller
#from rollout.roll_out_params import RollOutParams
from rollout.simple_rollout import SimplePolicyRoller
from rollout.simple_parallel_rollout import SimpleParallelPolicyRoller
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
    #print("Discard dataset and start new evaluation  (D/d)")
    print("Cancel (N/n/C/c)")
    while True:
        char = input(">>>>")
        if char in ["Y", "y"]:
            return "load"
        #elif char in ["D", "d"]:
        #    return "discard"
        elif char in ["N", "n", "C", "c"]:
            return "cancel"
        else:
            print(f"Unrecognized input: {char}")


def evaluate():
    P.initialize_experiment()
    params = P.get_current_parameters()

    model, model_loaded = load_model()
    oracle, _ = load_model("oracle")

    eval_dataset_name = data_io.paths.get_eval_tmp_dataset_name(P.get("Setup::model"), P.get("Setup::run_name"))
    eval_dataset_path = data_io.paths.get_dataset_dir(eval_dataset_name)

    if P.get("Setup::num_workers") > 1:
        local_ray = P.get("Setup::local_ray", False)
        ray.init(num_cpus=12,
                 num_gpus=1,
                 memory=40 * (1024 ** 3),
                 local_mode=local_ray,
                 ignore_reinit_error=True)
        roller = SimpleParallelPolicyRoller(policy=model,
                                            num_workers=P.get("Setup::num_workers"),
                                            oracle=oracle,
                                            device=None,
                                            dataset_save_name=eval_dataset_name,
                                            restart_every_n=1000,
                                            no_reward=False)
    else:
        roller = SimplePolicyRoller(instance_id=0,
                                    real_drone=P.get("Setup::real_drone"),
                                    policy=model,
                                    oracle=oracle,
                                    dataset_save_name=eval_dataset_name,
                                    no_reward=False)

    eval_envs = list(sorted(get_correct_eval_env_id_list()))

    round_size = P.get("Data::collect_n_at_a_time")

    custom_eval = "Eval" in params and params["Eval"]["custom_eval"]
    instructions = None
    if custom_eval:
        raise ValueError("TODO: Allow custom eval on specific segs and not only envs")
        examples = params["Eval"]["examples"]
        eval_envs, eval_sets, eval_segs, instructions = tuple(map(lambda m: list(m), list(zip(*examples))))
        print("!! Running custom evaluation with the following setup:")
        print(examples)

    # Check if dataset exists and offer to continue, abort, or delete the dataset
    cumulative_dataset = []
    if os.path.exists(eval_dataset_path):
        result = query_user_load_discard(eval_dataset_path)
        if result == "load":
            print("Loading dataset and continuing evaluation")
            cumulative_dataset = load_multiple_env_data_from_dir(eval_dataset_path, single_proc=True)
        elif result == "discard":
            print("Discarding existing evaluation data")
            shutil.rmtree(eval_dataset_path)
        elif result == "cancel":
            print("Cancelling evaluation")
            return
    os.makedirs(eval_dataset_path, exist_ok=True)

    collected_envs = set([rollout[0]["env_id"] for rollout in cumulative_dataset if len(rollout) > 0])
    eval_envs = [e for e in eval_envs if e not in collected_envs]
    if P.get("Setup::compute_results_no_rollout", False):
        eval_envs = []

    for i in range(0, len(eval_envs), round_size):
        j = min(len(eval_envs), i + round_size)
        round_envs = eval_envs[i:j]
        dataset = roller.rollout_envs(round_envs)

        # Save this data
        # Roller already saves the data
        #for rollout in dataset:
        #    ## rollout is a list of samples:
        #    env_id = rollout[0]["env_id"] if "metadata" in rollout[0] else rollout[0]["env_id"]
        #    if len(rollout) > 0:
        #        save_dataset_to_path(os.path.join(eval_dataset_path, str(env_id)), rollout)
        #    ## rollout is a list of segments, each is a list of samples

        cumulative_dataset += dataset
        print(f"Saved cumulative dataset to: {eval_dataset_path}")

    dataset = cumulative_dataset

    results = {}
    if P.get("Setup::eval_landmark_side"):
        evaler = DataEvalLandmarkSide(P.get("Setup::run_name"), save_images=True, world_size=P.get("Setup::world_size_m"))
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()
    if P.get("Setup::eval_nl"):
        evaler = DataEvalNL(f"{P.get('Setup::run_name')}-len1", save_images=False, entire_trajectory=False, custom_instr=instructions, aug_len=1)
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()
        print("Results-Length-1:", results)
        evaler = DataEvalNL(f"{P.get('Setup::run_name')}-len2", save_images=False, entire_trajectory=False, custom_instr=instructions, aug_len=2)
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()
        print("Results-Length-2:", results)



if __name__ == "__main__":
    evaluate()
    multiprocessing.set_start_method("spawn")
import os
import sys
import multiprocessing

from rollout.simple_rollout import SimplePolicyRoller
from data_io.instructions import get_correct_eval_env_id_list, get_segs_available_for_env
from data_io.models import load_model
from data_io.paths import get_eval_tmp_dataset_name, get_dataset_dir
from data_io.train_data import get_supervised_data_filename
from utils.dict_tools import dict_merge

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


def env_data_already_collected(env_id, model_name, run_name):
    dname = get_eval_tmp_dataset_name(model_name, run_name)
    dataset_path = get_dataset_dir(dname)
    data_file = os.path.join(dataset_path, get_supervised_data_filename(env_id))
    return os.path.isfile(data_file)


def check_and_prompt_if_data_exists(system_namespaces):
    existing_datasets = []
    for namespace in system_namespaces:
        P.switch_to_namespace(namespace)
        setup = P.get_current_parameters()["Setup"]
        dname = get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
        dpath = get_dataset_dir(dname)
        if os.path.exists(dpath):
            existing_datasets.append(dname)

    if len(existing_datasets) > 0:
        print("The following evaluation rollout datasets already exist:")
        print(existing_datasets)
        print("Do you want to continue evaluation and extend these datasets?")
        while True:
            char = input("(y/n)>>>>")
            if char in ["Y", "y"]:
                return
            elif char in ["N", "n"]:
                print("You may delete/move the existing datasets and run again!")
                sys.exit(0)
            else:
                print(f"Unrecognized input: {char}")


def setup_parameter_namespaces():
    global_params = P.load_parameters(None)
    systems_params = global_params["MultipleEval"]["SystemsParams"]
    setup_overlay = global_params["MultipleEval"]["SetupOverlay"]

    # ----------------------------------------------------------------------------------------
    # Initialize systems
    # ----------------------------------------------------------------------------------------

    # Set up parameters for each system that will be evaluated
    system_namespaces = []
    for system_params in systems_params:
        namespace = system_params["param_namespace"]
        local_params = P.load_parameters(system_params["param_file"])
        local_params["Setup"] = dict_merge(local_params["Setup"], setup_overlay)
        P.set_parameters_for_namespace(local_params, namespace)
        system_namespaces.append(namespace)

    P.log_experiment_start(global_params["MultipleEval"]["run_name"])
    return global_params, system_namespaces


def multiple_eval_rollout():

    params, system_namespaces = setup_parameter_namespaces()
    setup_overlay = params["MultipleEval"]["SetupOverlay"]
    domain = "real" if setup_overlay["real_drone"] else "sim"
    one_at_a_time = params["MultipleEval"]["one_at_a_time"]
    check_and_prompt_if_data_exists(system_namespaces)

    # Load the systems
    # TODO: Check how many can fit in GPU memory. If not too many, perhaps we can move them off-GPU between rounds
    policies = []
    for system_namespace in system_namespaces:
        P.switch_to_namespace(system_namespace)
        setup = P.get_current_parameters()["Setup"]
        policy, _ = load_model(setup["model"], setup["model_file"], domain)
        policies.append(policy)

    # ----------------------------------------------------------------------------------------
    # Initialize Roller
    # ----------------------------------------------------------------------------------------
    policy_roller = SimplePolicyRoller(
        instance_id=7,
        real_drone = setup_overlay["real_drone"],
        policy=None,
        oracle=None,
        no_reward=True
    )

    # ----------------------------------------------------------------------------------------
    # Collect rollouts
    # ----------------------------------------------------------------------------------------

    eval_envs = list(sorted(get_correct_eval_env_id_list()))
    count = 0

    # Loop over environments
    for env_id in eval_envs:
        seg_ids = get_segs_available_for_env(env_id, 0)
        env_ids = [env_id] * len(seg_ids)
        print("Beginning rollouts for env: {env_id}")
        if len(seg_ids) == 0:
            print("   NO SEGMENTS! Next...")
            continue

        # Loop over systems and save data
        for i, (policy, system_namespace) in enumerate(zip(policies, system_namespaces)):
            print(f"Rolling policy in namespace {system_namespace} for env: {env_id}")
            P.switch_to_namespace(system_namespace)
            setup = P.get_current_parameters()["Setup"]
            if env_data_already_collected(env_id, setup["model"], setup["run_name"]):
                print(f"Skipping env_id: {env_id}, policy: {setup['model']}")
                continue

            eval_dataset_name = get_eval_tmp_dataset_name(setup["model"], setup["run_name"])
            policy_roller.set_policy(policy)
            # when the last policy is done, we should land the drone
            policy_roller.rollout_segments(env_ids, seg_ids, None, False, 0, save_dataset_name=eval_dataset_name,
                                           rl_rollout=False, land_afterwards=(i==len(policies)-1))
            count += 1

        if one_at_a_time and count > 0:
            print("Stopping. Run again to roll-out on the next environment!")
            break

    print("Done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    multiple_eval_rollout()

import os
from evaluation.evaluate_nl import DataEvalNL
from data_io.instructions import get_correct_eval_env_id_list, get_all_instructions
from data_io.paths import get_eval_tmp_dataset_name, get_results_dir
from data_io.train_data import load_multiple_env_data

import parameters.parameter_server as P


def log(text, logdir):
    print(text)
    os.makedirs(logdir, exist_ok=True)
    with open(f"{logdir}/log.txt", "a") as fp:
        fp.write(text + "\n")


def evaluate_saved_rollouts():
    params = P.get_current_parameters()
    setup = params["Setup"]
    model_name = setup["model"]
    run_name = setup["run_name"]
    eval_dname = get_eval_tmp_dataset_name(model_name, run_name)

    eval_envs = set(list(sorted(get_correct_eval_env_id_list())))
    rollouts = load_multiple_env_data(eval_dname, single_proc=True)
    present_envs = set([rollout[0]["env_id"] for rollout in rollouts if len(rollout) > 0])
    missing_envs = eval_envs - present_envs

    logdir = get_results_dir(run_name)

    if len(missing_envs) > 0:
        print(f"Warning! {len(missing_envs)} envs missing: {missing_envs}")
        #sys.exit(1)

    log("", logdir)
    log("--------------------------------------------------------------------------------------------", logdir)
    log(f"Evaluating rollouts for run {run_name}", logdir)
    log(f"   using dataset {eval_dname}", logdir)
    log(f"   missing envs {missing_envs}", logdir)
    log("--------------------------------------------------------------------------------------------", logdir)

    SAVE_IMAGES = True

    #evaler1 = DataEvalNL(setup["run_name"]+"1-1", save_images=SAVE_IMAGES, entire_trajectory=False, aug_len=1)
    #evaler1.evaluate_dataset(rollouts)
    #results1 = evaler1.get_results()

    evaler2 = DataEvalNL(setup["run_name"]+"2-2", save_images=SAVE_IMAGES, entire_trajectory=False, aug_len=2)
    evaler2.evaluate_dataset(rollouts)
    results2 = evaler2.get_results()

    #evalerf = DataEvalNL(setup["run_name"]+"1-2", save_images=SAVE_IMAGES, entire_trajectory=False)
    #evalerf.evaluate_dataset(rollouts)
    #resultsf = evalerf.get_results()

    #log(f"Results 1-1:{results1}", logdir)
    log(f"Results 2-2:{results2}", logdir)
    #log(f"Results 1-2:{resultsf}", logdir)

    log(f" -- END EVALUATION FOR {run_name}-- ", logdir)
    log("--------------------------------------------------------------------------------------------", logdir)


if __name__ == "__main__":
    P.initialize_experiment()
    evaluate_saved_rollouts()

from evaluation.evaluate_t_landmark_side import DataEvalLandmarkSide
from evaluation.evaluate_nl import DataEvalNL
from rollout.parallel_roll_out import ParallelPolicyRoller
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams
from data_io.weights import restore_pretrained_weights
from data_io.instructions import get_correct_eval_env_id_list
from data_io.models import load_model

import parameters.parameter_server as P


def evaluate():
    P.initialize_experiment()
    params = P.get_current_parameters()
    setup = params["Setup"]

    models = []
    for i in range(setup["num_workers"]):
        model, model_loaded = load_model()
        if setup["restore_weights_name"]:
            restore_pretrained_weights(model, setup["restore_weights_name"], setup["fix_restored_weights"])
        models.append(model)

    eval_envs = get_correct_eval_env_id_list()

    roll_out_params = RollOutParams() \
                        .setModelName(setup["model"]) \
                        .setModelFile(setup["model_file"]) \
                        .setRunName(setup["run_name"]) \
                        .setSetupName(P.get_setup_name()) \
                        .setEnvList(eval_envs) \
                        .setMaxDeviation(400) \
                        .setHorizon(100) \
                        .setStepsToForceStop(10) \
                        .setPlot(False) \
                        .setShowAction(False) \
                        .setIgnorePolicyStop(False) \
                        .setPlotDir("evaluate/" + setup["run_name"]) \
                        .setSavePlots(True) \
                        .setRealtimeFirstPerson(False) \
                        .setSaveSamples(False) \
                        .setBuildTrainData(False) \
                        .setSegmentReset("always") \
                        .setSegmentLevel(True) \
                        .setFirstSegmentOnly(False) \
                        .setDebug(setup["debug"]) \
                        .setCuda(setup["cuda"])

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

    dataset = roller.roll_out_policy(roll_out_params)

    results = {}
    if setup["eval_landmark_side"]:
        evaler = DataEvalLandmarkSide(setup["run_name"])
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()
    if setup["eval_nl"]:
        evaler = DataEvalNL(setup["run_name"], save_images=True, entire_trajectory=False, custom_instr=instructions)
        evaler.evaluate_dataset(dataset)
        results = evaler.get_results()

    print("Results:", results)


if __name__ == "__main__":
    evaluate()
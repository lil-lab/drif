from mains.eval.evaluate_saved_rollouts import evaluate_saved_rollouts
from mains.eval.multiple_eval_rollout import setup_parameter_namespaces

import parameters.parameter_server as P


def evaluate_multiple_saved_rollouts():
    params, system_namespaces = setup_parameter_namespaces()

    # ----------------------------------------------------------------------------------------
    # Initialize systems
    # ----------------------------------------------------------------------------------------

    for system_namespace in system_namespaces:
        P.switch_to_namespace(system_namespace)
        evaluate_saved_rollouts()

if __name__ == "__main__":
    evaluate_multiple_saved_rollouts()

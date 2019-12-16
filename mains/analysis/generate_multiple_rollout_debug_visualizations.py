from mains.eval.multiple_eval_rollout import setup_parameter_namespaces
from mains.analysis.generate_rollout_debug_visualizations import generate_rollout_debug_visualizations

import parameters.parameter_server as P


def generate_multiple_rollout_visualizations():
    params, system_namespaces = setup_parameter_namespaces()

    for system_namespace in system_namespaces:
        P.switch_to_namespace(system_namespace)
        generate_rollout_debug_visualizations()

if __name__ == "__main__":
    generate_multiple_rollout_visualizations()

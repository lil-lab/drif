import os
import sys
import datetime
from data_io.helpers import load_json, save_json
from utils.dict_tools import dict_merge

CURRENT_PARAMS = None
CURRENT_RUN = None
SETUP_NAME = None
CURRENT_NAMESPACE = None


def _get_param_server_dir():
    pyfile = os.path.realpath(__file__)
    pydir = os.path.dirname(pyfile)
    return pydir


def _get_past_run_dir(run_name):
    pydir = _get_param_server_dir()
    past_runs_dir = os.path.join(pydir, "past_runs_pre_submission")
    run_dir = os.path.join(past_runs_dir, run_name)
    return run_dir


def _load_params(setup_name):
    pydir = _get_param_server_dir()
    paramsdir = os.path.join(pydir, "run_params")
    paramsname = setup_name + ".json"
    paramsfile = os.path.join(paramsdir, paramsname)
    params = load_json(paramsfile)

    return params


def get_state():
    global CURRENT_RUN, CURRENT_NAMESPACE, CURRENT_PARAMS
    return {"CURRENT_PARAMS": CURRENT_PARAMS,
            "CURRENT_NAMESPACE": CURRENT_NAMESPACE,
            "CURRENT_RUN": CURRENT_RUN}


def set_state(state):
    global CURRENT_RUN, CURRENT_NAMESPACE, CURRENT_PARAMS
    CURRENT_RUN = state["CURRENT_RUN"]
    CURRENT_NAMESPACE = state["CURRENT_NAMESPACE"]
    CURRENT_PARAMS = state["CURRENT_PARAMS"]


def log_experiment_start(run_name):
    global CURRENT_PARAMS, CURRENT_RUN
    rundir = _get_past_run_dir(run_name)
    paramsfile = os.path.join(rundir, "params.json")
    save_json(CURRENT_PARAMS, paramsfile)


def import_include_params(params):
    includes = params.get("@include") or []
    inherited_params = {}
    for include in includes:
        #print("Including params:", include)
        incl_params = _load_params(include)
        if incl_params is None:
            raise ValueError("No parameter file include found for: ", include)
        incl_params = import_include_params(incl_params)
        inherited_params = dict_merge(inherited_params, incl_params)

    # Overlay the defined parameters on top of the included parameters
    params = dict_merge(inherited_params, params)
    return params


def load_parameters(setup_name):
    if setup_name is None:
        assert len(sys.argv) >= 2, "The second command-line argument provided must be the setup name"
        setup_name = sys.argv[1]
    # Load the base configuration
    params = _load_params(setup_name)
    if params is None:
        print("Whoops! Parameters not found for: " + str(setup_name))
    # Load all the included parameters
    params = import_include_params(params)
    return params


def set_parameters_for_namespace(params, namespace):
    global CURRENT_PARAMS
    if CURRENT_PARAMS is None:
        CURRENT_PARAMS = {}
    CURRENT_PARAMS[namespace] = params


def switch_to_namespace(namespace):
    global CURRENT_NAMESPACE
    CURRENT_NAMESPACE = namespace


def get_current_namespace():
    global CURRENT_NAMESPACE
    return CURRENT_NAMESPACE


def initialize_experiment(setup_name=None):
    """
    :param setup_name: relative path from parameters/run_params, excluding .json extension,
    that specifies which config file to load and use. If None, uses the first argument provided to the program.
    :param namespace: load experiments in a specific namespace. Namespaces allow multiple experiment
    configs to be present at once, and switching between which one to use.
    :return:
    """
    params = load_parameters(setup_name)

    if "Setup" in params and "run_name" in params["Setup"]:
        run_name = params["Setup"]["run_name"]
    else:
        run_name = "UntitledRun"

    # Save for external access
    global CURRENT_PARAMS, CURRENT_RUN, SETUP_NAME, CURRENT_NAMESPACE
    CURRENT_NAMESPACE = "Global"
    CURRENT_PARAMS = {CURRENT_NAMESPACE: params}
    CURRENT_RUN = {CURRENT_NAMESPACE: run_name}
    SETUP_NAME = setup_name
    log_experiment_start(run_name)


def get_current_run_name():
    global CURRENT_RUN, CURRENT_NAMESPACE
    return CURRENT_RUN[CURRENT_NAMESPACE]


def get_current_parameters():
    global CURRENT_PARAMS, CURRENT_NAMESPACE
    return CURRENT_PARAMS[CURRENT_NAMESPACE]


def get_setup_name():
    print(DeprecationWarning("get_setup_name is deprecated as it is inconsistent with namespaces! To be removed..."))
    global SETUP_NAME
    return SETUP_NAME


def get_stamp():
    stamp = datetime.datetime.now().strftime("%M %d %Y - %H:%M:%S")
    return stamp

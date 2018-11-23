import os
import sys
import datetime
from data_io.helpers import load_json, save_json
from utils.dict_tools import dict_merge

CURRENT_PARAMS = None
CURRENT_RUN = None
SETUP_NAME = None


def get_param_server_dir():
    pyfile = os.path.realpath(__file__)
    pydir = os.path.dirname(pyfile)
    return pydir


def get_past_run_dir(run_name):
    pydir = get_param_server_dir()
    past_runs_dir = os.path.join(pydir, "past_runs")
    run_dir = os.path.join(past_runs_dir, run_name)
    return run_dir


def load_params(setup_name):
    pydir = get_param_server_dir()
    paramsdir = os.path.join(pydir, "run_params")
    paramsname = setup_name + ".json"
    paramsfile = os.path.join(paramsdir, paramsname)
    params = load_json(paramsfile)

    return params


def log_experiment_start(run_name, params):
    rundir = get_past_run_dir(run_name)
    paramsfile = os.path.join(rundir, "params.json")
    save_json(params, paramsfile)


def import_include_params(params):
    includes = params.get("@include") or []
    inherited_params = {}
    for include in includes:
        print("Including params:", include)
        incl_params = load_params(include)
        if incl_params is None:
            raise ValueError("No parameter file include found for: ", include)
        incl_params = import_include_params(incl_params)
        inherited_params = dict_merge(inherited_params, incl_params)

    # Overlay the defined parameters on top of the included parameters
    params = dict_merge(inherited_params, params)
    return params


def initialize_experiment(setup_name=None):
    if setup_name is None:
        assert len(sys.argv) >= 2, "The second command-line argument provided must be the setup name"
        setup_name = sys.argv[1]

    # Load the base configuration
    params = load_params(setup_name)
    if params is None:
        print("Whoops! Parameters not found for: " + str(setup_name))

    if "Setup" in params and "run_name" in params["Setup"]:
        run_name = params["Setup"]["run_name"]
    else:
        run_name = "UntitledRun"

    # Load all the included parameters
    params = import_include_params(params)

    # Save for external access
    global CURRENT_PARAMS, CURRENT_RUN, SETUP_NAME
    CURRENT_PARAMS = params
    CURRENT_RUN = run_name
    SETUP_NAME = setup_name
    log_experiment_start(run_name, params)


def get_stamp():
    stamp = datetime.datetime.now().strftime("%M %d %Y - %H:%M:%S")
    return stamp


def log(string):
    rundir = get_past_run_dir(CURRENT_RUN)
    logfile = os.path.join(rundir, "log.txt")
    stamp = get_stamp()
    logline = stamp + " " + string
    with open(logfile, "a") as fp:
        fp.write(logline + "\n")


def get_current_parameters():
    global CURRENT_PARAMS
    return CURRENT_PARAMS


def get_run_name():
    global CURRENT_RUN
    return CURRENT_RUN


def get_setup_name():
    global SETUP_NAME
    return SETUP_NAME
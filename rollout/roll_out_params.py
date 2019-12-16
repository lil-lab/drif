from enum import Enum

from policies.fancy_carrot_planner import FancyCarrotPlanner
from policies.simple_carrot_planner import SimpleCarrotPlanner
from policies.basic_carrot_planner import BasicCarrotPlanner
from data_io.models import load_model
import data_io.paths as paths

import parameters.parameter_server as P

class RolloutStrategy(Enum):
    POLICY = 1,
    REFERENCE = 2,
    POLICY_IN_REF_OUT = 3,
    MIXTURE = 4


class SwitchThresholdStrategy(Enum):
    IN_ONLY = 1,
    UNIFORM = 2


class RollOutParams:
    """
    This is just a class that holds a bunch of parameters, sort of like a C-struct.
    I did this, so that you don't have to provide 15 arguments to the policy rollout function.
        """
    def __init__(self):
        self.params = None
        self.policy_loaded = False
        self.use_policy = False
        self.flag = None                    # Flag added to the metadata of each sample
        self.ignore_policy_stop = False
        self.write_summaries = True
        self.model_name = "unnamed"
        self.model_file = None
        self.run_name = "default_run"
        self.segment_reset = "always"
        self.first_person = False
        self.plot = False
        self.show_action = False
        self.plot_dir = "."
        self.save_plots = False
        self.save_samples = False
        self.collect_data = False
        self.envs = None
        self.seg_list = None
        self.custom_instructions = None
        self.cuda = False
        self.build_train_data = False
        self.max_deviation = 100.0
        self.horizon = 80.0
        self.steps_to_kill = 50.0
        self.segment_level = False
        self.debug = False
        self.setup_name = ""
        self.first_segment_only = False
        self.wrong_path_p = 0.0

        self.switch_prob = 0.0
        self.mixture_ref_prob = 0.0
        self.rollout_strategy = RolloutStrategy.POLICY
        self.threshold_strategy = SwitchThresholdStrategy.IN_ONLY

        self.real_drone = False
        self.policy = None
        self.ref_policy = None

    # Methods for setting parameters
    def setModelName(self, model_name):
        self.model_name = model_name
        return self

    def setModelFile(self, model_file):
        self.model_file = model_file
        return self

    def setWriteSummaries(self, write_summaries):
        self.write_summaries = write_summaries
        return self

    def setRunName(self, run_name):
        self.run_name = run_name
        return self

    def setSetupName(self, setup_name):
        self.setup_name = setup_name
        return self

    def setRealtimeFirstPerson(self, first_person):
        self.first_person = first_person
        return self

    def setPlot(self, plot):
        self.plot = plot
        return self

    def setShowAction(self, show_action):
        self.show_action = show_action
        return self

    def setPlotDir(self, dir):
        self.plot_dir = dir
        return self

    def setSavePlots(self, save_plots):
        self.save_plots = save_plots
        return self

    def setSaveSamples(self, save_samples):
        self.save_samples = save_samples
        return self

    def setCollectData(self, collect_data):
        self.collect_data = collect_data
        return self

    def setEnvList(self, env_list):
        self.envs = env_list
        return self

    def setSegList(self, seg_list):
        self.seg_list = seg_list
        return self

    def setCustomInstructions(self, custom_instr):
        self.custom_instructions = custom_instr
        return self

    # segment_reset = "never", "always" or "if_reset"
    def setSegmentReset(self, segment_reset):
        self.segment_reset = segment_reset
        return self

    def setSegmentLevel(self, is_segment_level):
        self.segment_level = is_segment_level
        return self

    def setFirstSegmentOnly(self, first_seg_only):
        self.first_segment_only = first_seg_only
        return self

    def isSegmentLevel(self):
        return self.segment_level

    def setCuda(self, cuda):
        self.cuda = cuda
        return self

    def setMaxDeviation(self, deviation):
        self.max_deviation = deviation
        return self

    def setStepsToForceStop(self, steps_to_stop):
        self.steps_to_kill = steps_to_stop
        return self

    def setHorizon(self, horizon):
        self.horizon = horizon
        return self

    def setRolloutStrategy(self, strategy):
        self.rollout_strategy = strategy
        return self

    def setSwitchThresholdStrategy(self, strategy):
        self.threshold_strategy = strategy
        return self

    def setSwitchProbability(self, prob):
        self.switch_prob = prob
        return self

    def setMixtureReferenceProbability(self, prob):
        self.mixture_ref_prob = prob
        return self

    def setIgnorePolicyStop(self, ignore):
        self.ignore_policy_stop = ignore
        return self

    def setFlag(self, flag):
        self.flag = flag
        return self

    def setWrongPathP(self, p):
        self.wrong_path_p = p
        return self

    def setDebug(self, debug):
        self.debug = debug
        return self

    def setRealDrone(self, realDrone):
        self.real_drone = realDrone
        return self

    def isRealDrone(self):
        return self.real_drone

    def isDebug(self):
        return self.debug

    # Call this to load the policy based on the provided model_name and model_file
    def loadPolicy(self):
        P.initialize_experiment(self.setup_name)
        self.params = P.get_current_parameters()["Rollout"]
        if self.model_name is not None:
            print ("RollOutParams loading model")
            print ("Use cuda: " + str(self.cuda))
            self.policy, self.policy_loaded = \
                load_model(model_file_override=self.model_file)

            self.use_policy = True
            if self.policy is not None:
                print("Loaded policy: ", self.model_name)
            else:
                print("Error loading policy: ", self.model_name)
        else:
            print("Error! Requested loadPolicy, but model_name is None!")
        return self

    def initPolicyContext(self, env_id, path):
        if self.params["oracle_type"] == "SimpleCarrotPlanner":
            self.ref_policy = SimpleCarrotPlanner(path, max_deviation=self.max_deviation)
        elif self.params["oracle_type"] == "FancyCarrotPlanner":
            self.ref_policy = FancyCarrotPlanner(path, max_deviation=self.max_deviation)
        elif self.params["oracle_type"] == "BasicCarrotPlanner":
            self.ref_policy = BasicCarrotPlanner(path, max_deviation=self.max_deviation)

        else:
            raise Exception("Unknown Oracle in RollOutParams: " + str(self.params["OracleType"]))

        if self.policyRequresGroundTruth():
            self.policy.set_path(path)
        if self.wrong_path_p > 0:
            self.policy.set_wrong_path_prob(self.wrong_path_p)
        if hasattr(self.policy, "setEnvContext"):
            self.policy.setEnvContext({"env_id": env_id})

    def setCurrentSegment(self, start_idx, end_idx):
        self.ref_policy.set_current_segment(start_idx, end_idx)
        if self.policyRequresGroundTruth():
            self.policy.set_current_segment(start_idx, end_idx)

    def setBuildTrainData(self, should):
        self.build_train_data = should
        return self

    def shouldBuildTrainData(self):
        return self.build_train_data

    def policyRequresGroundTruth(self):
        return isinstance(self.policy, FancyCarrotPlanner) \
               or isinstance(self.policy, SimpleCarrotPlanner) \
                or isinstance(self.policy, BasicCarrotPlanner)

    # Getter methods
    def shouldResetAlways(self):
        return self.segment_reset == "always"

    def shouldResetIfFailed(self):
        return self.segment_reset == "if_failed"

    def shouldNeverReset(self):
        return self.segment_reset == "never"

    def hasPolicy(self):
        return self.use_policy

    def getFlag(self):
        return self.flag

    def shouldIgnorePolicyStop(self):
        return self.ignore_policy_stop

    def getSaveSamplesPath(self, env_id, instruction_set_id, seg_idx, step_num):
        image_data_path = paths.get_rollout_samples_dir()
        folder = image_data_path + "/" + self.run_name + "_" + self.model_name + "_env_" + \
                 str(env_id) + "_set_" + str(instruction_set_id) + "/"
        filename = "seg_" + str(seg_idx) + "_step_" + str(step_num)
        return folder + filename

    def getSavePlotPath(self, env_id, instruction_set_id, seg_idx):
        plot_data_path = paths.get_rollout_plots_dir()
        filaname = plot_data_path + self.plot_dir + "/"
        filename = filaname + self.run_name + "_" + self.model_name + "_env_" + \
                 str(env_id) + "_set_" + str(instruction_set_id) + "_seg_" + str(seg_idx)
        return filename
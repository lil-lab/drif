import rospy

from data_io.instructions import get_restricted_env_id_lists
from rollout.roll_out import PolicyRoller
from rollout.roll_out_params import RollOutParams

import parameters.parameter_server as P

P.initialize_experiment()
params = P.get_current_parameters()
setup = params["Setup"]

import random
#banana_ids = [2, 4, 51, 52, 62, 63, 65, 87, 92, 98]  #[random.randint(0, 499)]
#gorilla_ids = [49, 56, 69, 74, 78, 83, 88, 97]
#env_ids = gorilla_ids[:3] + banana_ids[:3]

train_envs, dev_envs, test_envs = get_restricted_env_id_lists()
env_ids = train_envs

# set infinite horizon (nb of steps before stop)
#horizon = 10 ** 6
horizon = 50

roll_out_params = RollOutParams() \
                    .setModelName(setup["model"]) \
                    .setModelFile(setup["model_file"]) \
                    .setRunName(setup["run_name"]) \
                    .setSetupName(P.get_setup_name()) \
                    .setEnvList(env_ids) \
                    .setMaxDeviation(400) \
                    .setHorizon(horizon) \
                    .setStepsToForceStop(10) \
                    .setPlot(False) \
                    .setShowAction(False) \
                    .setIgnorePolicyStop(True) \
                    .setPlotDir("evaluate/" + setup["run_name"]) \
                    .setSavePlots(True) \
                    .setCollectData(False)  \
                    .setRealtimeFirstPerson(False) \
                    .setSaveSamples(False) \
                    .setBuildTrainData(False) \
                    .setSegmentReset("always") \
                    .setSegmentLevel(True) \
                    .setDebug(setup["debug"]) \
                    .setCuda(setup["cuda"])  \
                    .setRealDrone(setup["real_drone"])
roller = PolicyRoller()
print("*********ignore policy stop ********",roll_out_params.shouldIgnorePolicyStop())

#env = ROSInterface(env_ids[0])
try:
    dataset = roller.roll_out_policy(roll_out_params)
except rospy.ROSInterruptException:
    roller.env.drone.land()
    #print("Error, exception encountered during drone controller execution!")

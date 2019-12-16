from drones.airsim_interface.droneController import DroneController as DroneControllerAirSim
try:
    from drones.aero_interface.droneController import DroneController as DroneControllerROS
except Exception as e:
    print("NO ROS")

# THIS LINE IS NEEDED:
from drones.rollout_exception import RolloutException


def drone_controller_factory(simulator):
    if simulator:
        return DroneControllerAirSim
    else:
        return DroneControllerROS

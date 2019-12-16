import os
import time
import imageio
import cv2

from data_io.paths import get_landmark_image_path, get_landmark_images_dir
from env_config.definitions.landmarks import PORTABLE_LANDMARK_RADII as LM_RADII
from env_config.definitions.landmarks import get_landmark_stage_name
from drones.droneController import drone_controller_factory

from visualization import Presenter
import parameters.parameter_server as P


def build_default_fpv_config(landmark_radii):
    config = {
        "zPos": [],
        "xPos": [],
        "isEnabled": [],
        "radius": [],
        "landmarkName": []
    }
    lm_name_to_idx = {}
    i = 0
    for lm_name, lm_radius in landmark_radii.items():
        config["xPos"].append(-500)
        config["zPos"].append(-500)
        config["isEnabled"].append(True)
        config["radius"].append(lm_radius)
        config["landmarkName"].append(lm_name)
        lm_name_to_idx[lm_name] = i
        i += 1
    return config, lm_name_to_idx


def collect_fpv_images():
    P.initialize_experiment()
    setup = P.get_current_parameters()["Setup"]
    flight_height = P.get_current_parameters()["PomdpInterface"]["flight_height"]
    drone = drone_controller_factory(simulator=True)(instance=0, flight_height=flight_height)

    os.makedirs(get_landmark_images_dir(), exist_ok=True)

    for landmark in LM_RADII.keys():
        print(f"Saving landmark: {landmark}")
        config, lm_name_to_idx = build_default_fpv_config(LM_RADII)
        idx = lm_name_to_idx[landmark]
        config["xPos"][idx] = 500
        config["zPos"][idx] = 500

        drone.set_current_env_from_config(config, instance_id=0)
        drone.reset_environment()

        time.sleep(0.5)

        drone.teleport_to([4.7-1.5, 4.7-1.5], 0.78 + 3.14159)
        drone.send_local_velocity_command([0.0, 0.0, 0.0])
        state, img = drone.get_state()

        Presenter().show_image(img, "img", waitkey=10)

        stage_name = get_landmark_stage_name(landmark)
        cv2.putText(img, stage_name, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)
        cv2.putText(img, stage_name, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        impath = get_landmark_image_path(stage_name)
        imageio.imsave(impath, img)


if __name__ == "__main__":
    collect_fpv_images()

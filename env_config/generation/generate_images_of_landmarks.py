import os

import scipy.misc

from drones.airsim_interface.rate import Rate
from env_config.definitions.landmarks import LANDMARK_RADII
from data_io.paths import get_landmark_images_dir
from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter

"""
This script is used to take pictures of various landmarks
"""
from parameters import parameter_server as P
P.initialize_experiment("nl_datacollect")

rate = Rate(0.1)

IMAGES_PER_LANDMARK_TRAIN = 1000
IMAGES_PER_LANDMARK_TEST = 200

env = PomdpInterface()

count = 0

presenter = Presenter()


def save_landmark_img(state, landmark_name, i, eval):
    data_dir = get_landmark_images_dir(landmark_name, eval)
    os.makedirs(data_dir, exist_ok=True)
    full_path = os.path.join(data_dir, landmark_name + "_" + str(i) + ".jpg")
    scipy.misc.imsave(full_path, state.image)


for landmark_name, landmark_radius in LANDMARK_RADII.items():

    for i in range(IMAGES_PER_LANDMARK_TRAIN):
        print("Generating train image " + str(i) + " for landmark '" + landmark_name + "'")
        state = env.reset_to_random_cv_env(landmark_name)
        presenter.show_image(state.image)
        save_landmark_img(state, landmark_name, i, False)

    for i in range(IMAGES_PER_LANDMARK_TEST):
        print("Generating test image " + str(i) + " for landmark '" + landmark_name + "'")
        state = env.reset_to_random_cv_env(landmark_name)
        presenter.show_image(state.image)
        save_landmark_img(state, landmark_name, i, True)

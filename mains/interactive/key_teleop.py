import threading
import numpy as np

from pykeyboard import PyKeyboardEvent

from drones.airsim_interface.rate import Rate
from data_io.instructions import get_all_instructions
from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter

from parameters.parameter_server import initialize_experiment, get_current_parameters
from utils.keyboard import KeyTeleop

initialize_experiment()


teleoper = KeyTeleop()
rate = Rate(0.1)

env = PomdpInterface(is_real=get_current_parameters()["Setup"]["real_drone"])

train_instructions, dev_instructions, test_instructions, _ = get_all_instructions()

count = 0
stuck_count = 0


def show_depth(image):
    grayscale = np.mean(image[:, :, 0:3], axis=2)
    depth = image[:, :, 3]
    comb = np.stack([grayscale, grayscale, depth], axis=2)
    comb -= comb.min()
    comb /= (comb.max() + 1e-9)
    Presenter().show_image(comb, "depth_alignment", torch=False, waitkey=1, scale=4)


for instruction_sets in train_instructions.values():
    for instruction_set in instruction_sets:
        env_id = instruction_set['env']
        env.set_environment(env_id, instruction_set["instructions"])
        env.reset(0)

        presenter = Presenter()
        cumulative_reward = 0
        for seg_idx in range(len(instruction_set["instructions"])):
            valid_segment = env.set_current_segment(seg_idx)
            if not valid_segment:
                continue

            teleoper.reset()

            while True:
                rate.sleep()
                action = teleoper.get_command()
                command = env.get_current_nl_command()
                state, reward, done = env.step(action)
                cumulative_reward += reward
                presenter.show_sample(state, action, reward, cumulative_reward, command)
                #show_depth(state.image)
                if done:
                    break
            print ("Segment finished!")
    print("Env finished!")

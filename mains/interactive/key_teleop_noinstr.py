import threading
import numpy as np

from pykeyboard import PyKeyboardEvent

from drones.airsim_interface.rate import Rate
from data_io.drone_sim import set_current_env_id
from data_io.instructions import get_all_instructions
from data_io.env import get_available_env_ids
from pomdp.pomdp_interface import PomdpInterface
from visualization import Presenter

from parameters.parameter_server import initialize_experiment

KEYFUNC = {
    "a": "left",
    "s": "backward",
    "d": "right",
    "w": "forward"
}

FWD_VEL = 1

TURN_VEL = 0.5

class MonitorSuper(PyKeyboardEvent):

    def __init__(self):
        PyKeyboardEvent.__init__(self)
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False

        self.current_vel = [0, 0, 0, 0]

    def tap(self, keycode, c, press):
        '''Monitor Super key.'''
        if c in KEYFUNC:
            #print (KEYFUNC[c])
            if KEYFUNC[c] == "left":
                if (press):
                    self.current_vel[2] = TURN_VEL
                else:
                    self.current_vel[2] = 0
            if KEYFUNC[c] == "right":
                if press:
                    self.current_vel[2] = -TURN_VEL
                else:
                    self.current_vel[2] = 0
            if KEYFUNC[c] == "forward":
                if press:
                    self.current_vel[0] = FWD_VEL
                else:
                    self.current_vel[0] = 0
            if KEYFUNC[c] == "backward":
                if press:
                    self.current_vel[0] = -FWD_VEL
                else:
                    self.current_vel[0] = 0

class KeyTeleop():

    def __init__(self):
        self.mon = MonitorSuper()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        self.mon.run()

    def get_command(self):
        return self.mon.current_vel

initialize_experiment("nl_datacollect_cage")

teleoper = KeyTeleop()
rate = Rate(0.1)

env = PomdpInterface()

env_ids = get_available_env_ids()

count = 0
stuck_count = 0


def show_depth(image):
    grayscale = np.mean(image[:, :, 0:3], axis=2)
    depth = image[:, :, 3]
    comb = np.stack([grayscale, grayscale, depth], axis=2)
    comb -= comb.min()
    comb /= (comb.max() + 1e-9)
    Presenter().show_image(comb, "depth_alignment", torch=False, waitkey=1, scale=4)


for env_id in env_ids:
    set_current_env_id(env_id)
    env.set_environment(env_id)
    env.reset(0)

    presenter = Presenter()
    cumulative_reward = 0
    while True:
        rate.sleep(quiet=True)
        action = teleoper.get_command()
        state, reward, done = env.step(action)
        rot_euler = state.get_rot_euler()
        print(rot_euler)
        cumulative_reward += reward
        presenter.show_sample(state, action, cumulative_reward, "")
        #show_depth(state.image)
        if done:
            break
    print ("Episode finished!")


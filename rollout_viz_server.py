import numpy as np
from data_io.env import load_env_img
from visualization import Presenter

import parameters.parameter_server as P
# TODO: Always convert image to numpy first, if it is in torch.
# TODO: Consider breaking apart into a rendering engine, and a thing that draws things?
H = 512
W = 512


class RolloutVizServer:
    def __init__(self):
        self.presenter = Presenter()
        self.clear()
        self.current_rollout = {}
        self.current_rollout_name = None
        self.env_image = None
        self.current_timestep = None

    def clear(self):
        self.current_rollout = {}
        self.current_rollout_name = None
        self.env_image = None
        self.current_timestep = None

    def start_rollout(self, env_id, set_idx, seg_idx, domain, dataset, suffix=""):
        rollout_name = f"{env_id}:{set_idx}:{seg_idx}:{domain}:{dataset}:{suffix}"
        self.current_rollout = {
            "top-down": []
        }
        self.current_rollout_name = rollout_name
        self.env_image = load_env_img(512, 512, alpha=True)

    def start_timestep(self, timestep):
        self.current_timestep = timestep
        # Add top-down view image for the new timestep
        self.current_rollout["top-down"].append()
        self.current_rollout["top-down"].append(self.env_image.copy())

    def set_drone_state(self, timestep, state):
        drone_pose = state.get_cam_pose()
        # Draw drone sprite on top_down image
        tdimg = self.current_rollout["top-down"][timestep]
        tdimg_n = self.presenter.draw_drone(tdimg, drone_pose, P.get_current_parameters()["Setup"]["world_size_m"])
        self.current_rollout["top-down"][timestep] = tdimg_n


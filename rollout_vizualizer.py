import os
import numpy as np
import moviepy.editor as mpy
from data_io.env import load_env_img, get_landmark_locations_airsim
from visualization import Presenter
from transformations import poses_m_to_px

import parameters.parameter_server as P
# TODO: Always convert image to numpy first, if it is in torch.
# TODO: Consider breaking apart into a rendering engine, and a thing that draws things?

class RolloutVisualizer:
    def __init__(self, resolution=512):
        self.presenter = Presenter()
        self.clear()
        self.current_rollout = {}
        self.current_rollout_name = None
        self.env_image = None
        self.current_timestep = None
        self.world_size_m = P.get_current_parameters()["Setup"]["world_size_m"]

        self.resolution = resolution

    def clear(self):
        self.current_rollout = {}
        self.current_rollout_name = None
        self.env_image = None
        self.current_timestep = None

    def _draw_landmarks(self, image, env_id):
        lm_names, lm_idx, lm_pos = get_landmark_locations_airsim(env_id=env_id)
        image = self.presenter.draw_landmarks(image, lm_names, lm_pos, self.world_size_m)
        return image

    def load_video_clip(self, env_id, seg_idx, rollout, domain, cam_name, rollout_dir):
        video_path = os.path.join(rollout_dir, f"rollout_{cam_name}_{env_id}-0-{seg_idx}.mkv")
        try:
            #if os.path.getsize(video_path) > 1024 * 1024 * 30
            print("Loading video: ", video_path)
            clip = mpy.VideoFileClip(video_path)
        except Exception as e:
            return None
        return clip


    def grab_frames(self, env_id, seg_idx, rollout, domain, frame_name, scale=1):
        frames = []
        for sample in rollout:
            if frame_name == "image":
                frame = sample["state"].image
            else:
                frame = sample[frame_name][0, :3, :, :].transpose((1, 2, 0))
            frame -= frame.min()
            frame = frame / (frame.max() + 1e-9)
            if scale != 1:
                frame = self.presenter.scale_image(frame, scale)
            frames.append(frame)
        return frames

    def action_visualization(self, env_id, seg_idx, rollout, domain, frame_name="action"):
        frames = []
        for sample in rollout:
            action = sample[frame_name]
            frame = np.ones((200, 200, 3), dtype=np.uint8)
            self.presenter.draw_action(frame, (1, 159), action)
            frames.append(frame)
        return frames

    def top_down_visualization(self, env_id, seg_idx, rollout, domain, params, replace_vdist=None):
        if params.get("draw_topdown", False):
            # TODO: flipdiag should be false for CoRL 2019
            bg_image = load_env_img(env_id, self.resolution, self.resolution, real_drone=True if domain=="real" else False, origin_bottom_left=True, flipdiag=True, alpha=True)
        else:
            bg_image = np.zeros((self.resolution, self.resolution, 3))
        if params.get("draw_landmarks", False):
            bg_image = self._draw_landmarks(bg_image, env_id)

        # Initialize stuff
        frames = []
        poses_m = []
        poses_px = []
        for sample in rollout:
            sample_image = bg_image.copy()
            frames.append(sample_image)
            state = sample["state"]
            #pose_m = state.get_cam_pose()
            pose_m = state.get_drone_pose()
            pose_px = poses_m_to_px(pose_m, self.resolution, self.resolution, self.world_size_m, batch_dim=False)
            poses_px.append(pose_px)
            poses_m.append(pose_m)

        instruction = rollout[0]["instruction"]
        print("Instruction: ")
        print(instruction)

        # Draw visitation distributions if requested:
        if params.get("include_vdist", False):
            import cv2
            print("Drawing visitation distributions")
            for i, sample in enumerate(rollout):
                # This was for CoRL 2019. CoRL 2020 has a bug that model_state only has the last frame
                # v_dist_w = sample["model_state"].tensor_store.get("v_dist_w")
                # This is for CoRL 2020
                # TODO: This kinda duplicates the visualization function in Partial2DDistribution
                if replace_vdist is not None:
                    v_dist_w = replace_vdist[i:i+1]
                else:
                    v_dist_w = sample["model_state"].tensor_store.get_inputs_batch("log_v_dist_w")[i].softmax()
                v_dist_w_inner = v_dist_w.inner_distribution[0]
                v_dist_w_inner = self.presenter.prep_image(v_dist_w_inner)
                v_dist_w_inner[:, :, 0] /= (np.percentile(v_dist_w_inner[:, :, 0], 98) + 1e-10)
                v_dist_w_inner[:, :, 1] /= (np.percentile(v_dist_w_inner[:, :, 1], 98) + 1e-10)
                v_dist_w_inner = np.clip(v_dist_w_inner, 0, 1)
                v_dist_w_outer = v_dist_w.outer_prob_mass[0]
                f = self.presenter.overlaid_image(frames[i], v_dist_w_inner, strength=0.6, bg_strength=0.8, interpolation=cv2.INTER_LINEAR)
                f = self.presenter.draw_prob_bars(f, v_dist_w_outer)
                frames[i] = f

        if params.get("include_layer", False):
            layer_name = params["include_layer"]
            print(f"Drawing first 3 channels of layer {layer_name}")
            for i, sample in enumerate(rollout):
                layer = sample[layer_name]
                if len(layer.shape) == 4:
                    layer = layer[0,:,:,:]
                layer = layer.transpose((2,1,0))
                layer = np.flipud(layer)
                frames[i] = self.presenter.overlaid_image(frames[i], layer[:,:,:3])

        if params.get("include_instr", False):
            print("Drawing instruction")
            for i, sample in enumerate(rollout):
                frames[i] = self.presenter.overlay_text(frames[i], sample["instruction"])

        # Draw trajectory history
        if params.get("draw_trajectory", False):
            print("Drawing trajectory")
            for i, sample in enumerate(rollout):
                history = poses_px[:i+1]
                position_history = [h.position for h in history]
                frames[i] = self.presenter.draw_trajectory(frames[i], position_history, self.world_size_m, convert=True)

        # Draw drone
        if params.get("draw_drone", False):
            print("Drawing drone")
            for i, sample in enumerate(rollout):
                frames[i] = self.presenter.draw_drone(frames[i], poses_m[i], self.world_size_m, convert=True)

        # Draw observability mask:
        if params.get("draw_fov", True):
            print("Drawing FOV")
            for i, sample in enumerate(rollout):
                frames[i] = self.presenter.draw_observability(frames[i], poses_m[i], self.world_size_m, 84)

        # Visualize
        if False:
            for i, sample in enumerate(rollout):
                self.presenter.show_image(frames[i], "sample_image", scale=1, waitkey=True)

        return frames

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


import os
import cv2
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

    def _auto_contrast(self, image):
        import cv2
        image_c = np.clip(image, 0.0, 1.0)
        hsv_image = cv2.cvtColor(image_c, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] *= 1.2
        image_out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        image_out = np.clip(image_out, 0.0, 1.0)
        #print(image_out.min(), image_out.)
        print(image_out[:, :, 1].min(), image_out[:, :, 1].max())
        return image

    def _integrate_mask(self, frames):
        frames_out = [frames[0]]
        for frame in frames[1:]:
            new_frame_out = np.maximum(frames_out[-1], frame)
            frames_out.append(new_frame_out)
        return frames_out

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
            elif frame_name == "action":
                action = sample["action"]
                bg = np.zeros((400, 400, 3))
                frame = self.presenter.draw_action(bg, offset=(0, 0), action=action)
            elif frame_name == "v_dist_r_inner":
                frame_t = sample[frame_name][:3, :, :].transpose((1, 2, 0))
                # TODO: These should come from params
                map_size = 64
                crop_size = 16
                gap = int((map_size - crop_size) / 2)
                crop_l = gap
                crop_r = map_size - gap
                frame_t = frame_t[crop_l:crop_r, crop_l:crop_r, :]
                frame_t[:, :, 0] /= (np.percentile(frame_t[:, :, 0], 99) + 1e-9)
                frame_t[:, :, 1] /= (np.percentile(frame_t[:, :, 1], 99) + 1e-9)
                frame_t = np.clip(frame_t, 0.0, 1.0)
                shp = list(frame_t.shape)
                shp[2] = 3
                frame = np.zeros(shp)
                frame[:, :, :2] = frame_t
                frame = cv2.resize(frame, dsize=(self.resolution, self.resolution))
            elif frame_name == "map_struct":
                frame_t = sample[frame_name][:3, :, :].transpose((1, 2, 0))
                shp = list(frame_t.shape)
                shp[2] = 3
                frame = np.zeros(shp)
                frame[:, :, :2] = frame_t
                frame = cv2.resize(frame, dsize=(self.resolution, self.resolution))
            elif frame_name == "ego_obs_mask":
                frame_t = sample["map_struct"][:3, :, :].transpose((1, 2, 0))
                shp = list(frame_t.shape)
                shp[2] = 3
                canvas = np.zeros(shp)
                canvas[:, :, :] = 1 - frame_t[:, :, 0:1]
                canvas[:, :, :] -= frame_t[:, :, 1:2]
                canvas = np.clip(canvas, 0.0, 1.0)
                frame = cv2.resize(canvas, dsize=(self.resolution, self.resolution))
            else:
                frame = sample[frame_name][0, :3, :, :].transpose((1, 2, 0))
            if frame_name in ["image", "v_dist_r_inner"]:
                frame -= frame.min()
                frame = frame / (frame.max() + 1e-9)
            else:
                frame -= np.percentile(frame, 0)
                frame /= (np.percentile(frame, 95) + 1e-9)
                frame = np.clip(frame, 0.0, 1.0)
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

    def overlay_frames(self, under_frames, over_frames, strength=0.5):
        overlaid_frames = [self.presenter.overlaid_image(u, o, strength=strength) for u, o in zip(under_frames, over_frames)]
        return overlaid_frames

    def top_down_visualization(self, env_id, seg_idx, rollout, domain, params):
        fd = domain == "real"
        obl = domain in ["simulator", "sim"]
        print(domain, obl)
        if params["draw_topdown"]:
            bg_image = load_env_img(env_id, self.resolution, self.resolution, real_drone=True if domain=="real" else False, origin_bottom_left=obl, flipdiag=False, alpha=True)
        else:
            bg_image = np.zeros((self.resolution, self.resolution, 3))
        if params["draw_landmarks"]:
            bg_image = self._draw_landmarks(bg_image, env_id)

        # Initialize stuff
        frames = []
        poses_m = []
        poses_px = []
        for sample in rollout:
            sample_image = bg_image.copy()
            frames.append(sample_image)
            state = sample["state"]
            pose_m = state.get_drone_pose()
            pose_px = poses_m_to_px(pose_m, self.resolution, self.resolution, self.world_size_m, batch_dim=False)
            poses_px.append(pose_px)
            poses_m.append(pose_m)

        instruction = rollout[0]["instruction"]
        print("Instruction: ")
        print(instruction)

        # Draw visitation distributions if requested:
        if params["include_vdist"]:
            print("Drawing visitation distributions")
            if params["ego_vdist"]:
                inner_key = "v_dist_r_inner"
                outer_key = "v_dist_r_outer"
            else:
                inner_key = "v_dist_w_inner"
                outer_key = "v_dist_w_outer"
            for i, sample in enumerate(rollout):
                v_dist_w_inner = np.flipud(sample[inner_key].transpose((2, 1, 0)))
                # Expand range of each channel separately so that stop entropy doesn't affect how trajectory looks
                v_dist_w_inner[:, :, 0] /= (np.percentile(v_dist_w_inner[:, :, 0], 99.5) + 1e-9)
                v_dist_w_inner[:, :, 1] /= (np.percentile(v_dist_w_inner[:, :, 1], 99.5) + 1e-9)
                v_dist_w_inner = np.clip(v_dist_w_inner, 0.0, 1.0)
                v_dist_w_outer = sample[outer_key]
                if bg_image.max() - bg_image.min() > 1e-9:
                    f = self.presenter.blend_image(frames[i], v_dist_w_inner)
                else:
                    f = self.presenter.overlaid_image(frames[i], v_dist_w_inner, strength=1.0)
                f = self.presenter.draw_prob_bars(f, v_dist_w_outer)
                frames[i] = f

        if params["include_layer"]:
            layer_name = params["include_layer"]
            print(f"Drawing first 3 channels of layer {layer_name}")
            accumulate = False
            invert = False
            gray = False
            if layer_name == "M_W_accum":
                accumulate = True
                layer_name = "M_W"
            if layer_name == "M_W_accum_inv":
                invert = True
                accumulate = True
                layer_name = "M_W"

            if layer_name.endswith("_Gray"):
                gray = True
                layer_name = layer_name[:-len("_Gray")]

            for i, sample in enumerate(rollout):
                layer = sample[layer_name]
                if len(layer.shape) == 4:
                    layer = layer[0, :, :, :]
                layer = layer.transpose((2,1,0))
                layer = np.flipud(layer)
                if layer_name in ["S_W", "F_W"]:
                    layer = layer[:, :, :3]
                else:
                    layer = layer[:, :, :3]
                if layer_name in ["S_W", "R_W", "F_W"]:
                    if gray:
                        layer -= np.percentile(layer, 1)
                        layer /= (np.percentile(layer, 99) + 1e-9)
                    else:
                        layer /= (np.percentile(layer, 97) + 1e-9)
                    layer = np.clip(layer, 0.0, 1.0)

                if layer_name in ["M_W"]:
                    # Having a 0-1 mask does not encode properly with the codec. Add a bit of imperceptible gaussian noise.
                    layer = layer.astype(np.float32)
                    layer = np.tile(layer, (1,1,3))

                if accumulate and i > 0:
                    layer = np.maximum(layer, prev_layer)

                prev_layer = layer
                if invert:
                    layer = 1 - layer
                if frames[i].max() > 0.01:
                    frames[i] = self.presenter.blend_image(frames[i], layer[:, :, :3])
                    #frames[i] = self.presenter.overlaid_image(frames[i], layer[:, :, :3])
                else:
                    scale = (int(self.resolution / layer.shape[0]), int(self.resolution / layer.shape[1]))
                    frames[i] = self.presenter.prep_image(layer[:, :, :3], scale=scale)

        if params["include_instr"]:
            print("Drawing instruction")
            for i, sample in enumerate(rollout):
                frames[i] = self.presenter.overlay_text(frames[i], sample["instruction"])

        # Draw trajectory history
        if params["draw_trajectory"]:
            print("Drawing trajectory")
            for i, sample in enumerate(rollout):
                history = poses_px[:i+1]
                position_history = [h.position for h in history]
                frames[i] = self.presenter.draw_trajectory(frames[i], position_history, self.world_size_m)

        # Draw drone
        if params["draw_drone"]:
            print("Drawing drone")
            for i, sample in enumerate(rollout):
                frames[i] = self.presenter.draw_drone(frames[i], poses_m[i], self.world_size_m)

        # Draw observability mask:
        if params["draw_fov"]:
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


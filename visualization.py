import itertools
import json
import zipfile
import os
import shutil
import string
import cv2
import functools
from imageio import imread

import moviepy.editor as mpy

from data_io.paths import get_sprites_dir
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import scipy.misc
from sklearn.metrics import confusion_matrix
from data_io.instructions import debug_untokenize_instruction
import imageio

#from data_io.units import UnrealUnits
from env_config.definitions.landmarks import get_landmark_stage_name
from data_io.env import load_env_img
from data_io.paths import get_env_image_path
from transformations import cf_to_img, poses_m_to_px, get_affine_rot_2d, get_affine_scale_2d, get_affine_trans_2d
from geometry import clip_angle

import parameters.parameter_server as P

FWD_MULTIPLIER = 0.5
ANG_MULTIPLIER = 2.0
ACTION_OPACITY = 160

# Flag for generating paper figures
DONT_DRAW_TEXT = False


class Presenter:

    def __init__(self):
        self.headless = P.get_current_parameters()["Environment"].get("headless", False)
        self.drone_image = None
        self.coord_grid = None

    def _load_drone_img(self):
        if self.drone_image is not None:
            return self.drone_image
        drone_path = os.path.join(get_sprites_dir(), "drone_img_u.png")
        drone_img = imread(drone_path).astype(np.float64) / 255
        self.drone_image = drone_img
        return self.drone_image

    def show_instruction(self, instruction_str):
        if self.headless:
            return
        cv2.namedWindow("instruction", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        win_width = 1349 * 2
        win_height = 40 * 2
        char_width_px = 10 * 2
        offset = 30 * 2
        fontsize = 0.62 * 2

        expected_width_px = len(instruction_str) * char_width_px
        left_padding = int(((win_width - expected_width_px) / 2))
        img = np.zeros((win_height, win_width))
        img.fill(0.92)
        cv2.putText(img, instruction_str, (left_padding, offset),
                    cv2.FONT_HERSHEY_DUPLEX, fontsize, (0, 0, 0), 2, cv2.LINE_AA)

        img = cv2.resize(img, (int(win_width/2), int(win_height/2)))

        cv2.imshow("instruction", img)
        cv2.waitKey(5)

    def draw_landmarks(self, image, lm_names, lm_pos, world_size_m):
        image = image.copy()
        for i, name in enumerate(lm_names):
            stage_name = get_landmark_stage_name(name)
            x_m, y_m, _ = lm_pos[i]
            x_px = int(x_m * image.shape[1] / world_size_m)
            y_px = int(y_m * image.shape[0] / world_size_m)
            x_px = image.shape[0] - x_px
            color = (1.0, 1.0, 1.0)
            #cv2.circle(image, (y_px, x_px), 5, color, -1)
            cv2.putText(image, stage_name, (y_px - len(stage_name)*4, x_px), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def draw_trajectory(self, image, positions, world_size_m, color="c"):
        h = image.shape[0]
        w = image.shape[1]

        x_targets, y_targets = list(zip(*positions))
        fig = Figure(figsize=(float(h)/100, float(w)/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.axis('off')
        ax.imshow(image, extent=(0, h, 0, w))
        ax.plot(x_targets, y_targets, color)
        ax.plot(x_targets[-1], y_targets[-1], color+"o")
        ax.axis('image')
        ax.set_autoscaley_on(False)
        ax.set_autoscalex_on(False)
        ax.set_xlim((0,w))
        ax.set_ylim((0,h))
        ax.set_xbound((0,w))
        ax.set_ybound((0,h))
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        canvas.draw()  # draw the canvas, cache the renderer
        image_out = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        image_out = image_out.reshape([h, w, 3])
        image_out = (image_out.astype(np.float64) / 255)
        return image_out

    def _transform_img_to_pose(self, image_out, img_to_transform, pose, img_scale):
        img_size_px = image_out.shape[1]
        yaw = pose.orientation
        desired_drone_size = img_size_px * img_scale
        scale = desired_drone_size / img_to_transform.shape[1]
        # Transforms, should be applied in order that they are defined
        # Scale it down to desired size
        scale_transform = get_affine_scale_2d(np.asarray([scale, scale]))
        # Translate it so that drone is centered around the origin
        trans_reset_transform = get_affine_trans_2d(np.asarray([-desired_drone_size / 2, -desired_drone_size / 2]))
        # Rotate it so that it faces the correct way
        rot_transform = get_affine_rot_2d(yaw)
        # Translate it so that it is centered around the correct position
        trans_transform = get_affine_trans_2d(pose.position)
        # First scale down, then rotate correctly, then translate to starting position
        transform = np.dot(trans_transform, np.dot(rot_transform, np.dot(trans_reset_transform, scale_transform)))

        # Transform:
        img_t = cv2.warpAffine(img_to_transform, transform[:2, :], (img_size_px, img_size_px))
        img_t_mask = (img_t > 1e-10).astype(np.int64)
        return img_t, img_t_mask

    # TODO: Generalize to draw_sprite_at_pose
    def draw_drone(self, image, pose_m, world_size_m):
        img_size_px = image.shape[1]
        pose_px = poses_m_to_px(as_pose=pose_m, img_size_px=img_size_px, world_size_px=img_size_px,
                                world_size_m=world_size_m, batch_dim=False)
        pose_px.position[1] = img_size_px - pose_px.position[1]
        drone_img = self._load_drone_img()
        DRONE_SIZE_FRACTIONAL = 0.1
        drone_img_t, drone_img_mask = self._transform_img_to_pose(image, drone_img, pose_px, DRONE_SIZE_FRACTIONAL)
        overlaid_image = image[:,:,:3] * (1 - drone_img_mask[:,:,:3]) + drone_img_t[:,:,:3] * drone_img_mask[:,:,:3]
        return overlaid_image

    def draw_observability(self, image, pose_m, world_size_m, h_fov):
        image = image.copy()
        img_size_px = image.shape[1]
        pose_px = poses_m_to_px(as_pose=pose_m, img_size_px=img_size_px, world_size_px=img_size_px,
                                world_size_m=world_size_m, batch_dim=False)
        pose_px.position[1] = img_size_px - pose_px.position[1]
        drone_yaw = -pose_px.orientation + 3.14159

        if self.coord_grid is None:
            lspace = np.linspace(0, image.shape[0] - 1, image.shape[0])
            coord_grid = np.meshgrid(lspace, lspace)
            coord_grid = [c[:,:,np.newaxis] for c in coord_grid]
            coord_grid = np.concatenate(coord_grid, axis=2)
            self.coord_grid = coord_grid

        heading = self.coord_grid - pose_px.position[np.newaxis, np.newaxis, :]
        yaws = np.arctan2(heading[:, :, 0], heading[:, :, 1])

        diff = yaws - drone_yaw
        toobig = diff > np.pi
        toosmall = diff < -np.pi
        diff[toobig] -= np.pi * 2
        diff[toosmall] += np.pi * 2
        diff = np.fabs(diff)

        visible_mask = diff < np.deg2rad(h_fov)/2
        invisible_mask = np.logical_not(visible_mask)
        image[invisible_mask] *= 0.8
        return image

    def draw_prob_bars(self, image, probabilities):
        GAP = 2
        origin_x = int(image.shape[0] * 0.6)
        origin_y = int(image.shape[1] * 0.8)
        area_width = image.shape[1] - origin_y
        area_height = image.shape[0] - origin_x
        bar_width = int(area_width / len(probabilities)) - GAP
        bar_height = area_height
        for i, prob in enumerate(probabilities):
            if i == 0: continue
            bar_active_height = int(bar_height * prob)
            bar_bottom = image.shape[0] - GAP
            bar_top = image.shape[0] - bar_height
            bar_left = origin_y + i * (bar_width + GAP)
            bar_right = bar_left + bar_width
            active_bar_top = bar_bottom - bar_active_height
            # draw rectangle around the bar:
            image[bar_bottom, bar_left:bar_right, :] = 1.0
            image[bar_top, bar_left:bar_right, :] = 1.0
            image[bar_top:bar_bottom, bar_left, :] = 1.0
            image[bar_top:bar_bottom, bar_right, :] = 1.0
            # fill the bar:
            image[active_bar_top:bar_bottom, bar_left:bar_right, i] += 0.5
            image = image.clip(0,1)
        return image

    def draw_action(self, image, offset, action):
        img2 = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
        offset = np.asarray(offset)

        rect_width = int(image.shape[0] * 0.2)
        h_rect_length = int(image.shape[0] * 0.5) - 2
        v_rect_length = int(image.shape[0] - rect_width - 2)

        left_rect_p1 = offset + np.asarray((0, 0))
        left_rect_p2 = offset + np.asarray((h_rect_length, rect_width))
        right_rect_p1 = offset + np.asarray((h_rect_length, 0))
        right_rect_p2 = offset + np.asarray((2*h_rect_length, rect_width))
        top_rect_p1 = offset + np.asarray((h_rect_length - rect_width / 2, 0))
        top_rect_p2 = offset + np.asarray((h_rect_length + rect_width / 2, -v_rect_length))

        turn_percent = action[2] * ANG_MULTIPLIER
        fwd_percent = max(action[0] * FWD_MULTIPLIER, 0)

        turn_p1 = offset + np.asarray((h_rect_length, 0))
        turn_p2 = offset + np.asarray((h_rect_length + h_rect_length * turn_percent, rect_width))
        turn_p1 = turn_p1.astype(int)
        turn_p2 = turn_p2.astype(int)

        fwd_p1 = offset + np.asarray((h_rect_length - rect_width / 2, 0))
        fwd_p2 = offset + np.asarray((h_rect_length - rect_width / 2 + rect_width,
                                      -v_rect_length * fwd_percent))
        fwd_p1 = fwd_p1.astype(int)
        fwd_p2 = fwd_p2.astype(int)

        turn_color = (255, 100, 100, ACTION_OPACITY)
        cv2.rectangle(img2, tuple(turn_p1), tuple(turn_p2), turn_color, thickness=-1)
        cv2.rectangle(img2, tuple(fwd_p1), tuple(fwd_p2), turn_color, thickness=-1)

        border_color = (255, 255, 255, ACTION_OPACITY)
        cv2.rectangle(img2, tuple(left_rect_p1.astype(int)), tuple(left_rect_p2.astype(int)), border_color)
        cv2.rectangle(img2, tuple(right_rect_p1.astype(int)), tuple(right_rect_p2.astype(int)), border_color)
        cv2.rectangle(img2, tuple(top_rect_p1.astype(int)), tuple(top_rect_p2.astype(int)), border_color)

        cv2.addWeighted(image, 1, img2, 0.4, 0, image)
        return image

    def scale_image(self, image, scale):
        image_o = cv2.resize(image, dsize=tuple(reversed([int(x) for x in np.asarray(image.shape)[:2]*scale])))
        return image_o

    def save_gif(self, frames, filepath, fps=2.0):
        frames = [filter_for_gif(f) for f in frames]
        imageio.mimsave(filepath, frames, "GIF-FI", fps=fps, quantizer="nq")

    def _make_make_frame(self, frames, fps):
        def make_frame_partial(frames, fps, t):
            frame_no = int(t * fps + 1e-5)
            frame_no = frame_no % len(frames)
            return frames[frame_no] * 255
        return functools.partial(make_frame_partial, frames, fps)

    def save_video(self, frames, filepath, fps=2.0):
        if isinstance(frames, list):
            duration = len(frames) / fps
            clip = mpy.VideoClip(self._make_make_frame(frames, fps), duration=duration)
        else:
            clip = frames
        clip.write_videofile(filepath, fps=fps)

    def get_all_file_paths_in_dir(self, directory):
        # initializing empty file paths list
        file_paths = []
        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory):
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
                # returning all file paths
        return file_paths

    def save_frames(self, frames, framedir):

        if isinstance(frames, list):
            os.makedirs(framedir, exist_ok=True)
            for i, frame in enumerate(frames):
                self.save_image(frame, name=str(i), folder=framedir)

            filepaths = self.get_all_file_paths_in_dir(framedir)
            zippath = f"{framedir}.zip"
            # writing files to a zipfile
            with zipfile.ZipFile(zippath, 'w') as zip:
                # writing each file one by one
                for file in filepaths:
                    zip.write(file, arcname=os.path.basename(file))

            shutil.rmtree(framedir)
        # This is a video
        else:
            pass

    def split_lines(self, string, maxchars=50):

        if len(string) < maxchars:
            return [string]
        else:
            words = string.split(" ")
            letter_count = 0
            split_word = 0
            for num, word in enumerate(words):
                if letter_count > maxchars:
                    break
                split_word = num
                letter_count += len(word)

            string1 = " ".join(words[:split_word])
            string2 = " ".join(words[split_word:])
            return [string1] + self.split_lines(string2)

    def plot_pts_on_torch_image(self, image, pts):
        """
        :param image: CxHxW image
        :param pts: Nx2 points - (H,W) coords in the image
        :return:
        """
        image_np = image.cpu().data.numpy()
        image_np = image_np.transpose((1, 2, 0))
        pts = pts.cpu().data.numpy()
        image_np[:, :, 0] = 0.0
        for pt in pts:
            image_np[pt[0], pt[1], 0] = 1.0
        return image_np


    def save_sample(self, path, drone_state, image, action, reward, command):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        mpimg.imsave(path + ".png", image[:,:,:3])
        data_dict = {
            "drone_state": list(drone_state),
            "action": list(action),
            "reward": reward,
            "command": command
        }
        file = open(path + ".json", "w")
        json.dump(data_dict, file)
        file.close()

    def show_action(self, action, name="action"):
        if self.headless:
            return
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.draw_action(image, (0, 100), action)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def show_sample(self, state, action, reward, cumulative_reward, command):
        if self.headless:
            return
        drone_state = state.state
        image = state.image

        if command is None:
            command = ""
        image = image[:,:,0:3]
        height, width = image.shape[:2]
        dst = cv2.resize(image, (6*width, 6*height), interpolation=cv2.INTER_CUBIC)

        start_point = [dst.shape[0]-1, dst.shape[1]/2]
        vec = np.asarray([-action[0], -action[2]])
        end_point = start_point + vec

        #cv2.line(dst, start_point, end_point)
        #Draw the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        instructionScale = 0.5
        instructionColor = (255, 255, 255)
        rewardScale = 1.0
        rewardColor = (255, 100, 100)
        lineType = 2
        rewardLineType = 2

        textOrgReward = (dst.shape[1] - 500, 40)

        command = ''.join(ch if ch not in set(string.punctuation) else "" for ch in command).strip().lower()
        wordlist = command.split(" ")
        wordlist = [word for word in wordlist if word != " "]
        command = " ".join(wordlist)

        lines = self.split_lines(command, maxchars=45)
        for i, line in enumerate(reversed(lines)):
            textOrg = (10, dst.shape[0] - 10 - int(35 * instructionScale * i))
            cv2.putText(dst, line, textOrg, font, instructionScale, instructionColor, lineType)

        reward_str = "Reward: {:5.1f}  Return: {:5.1f}".format(reward, cumulative_reward)
        cv2.putText(dst, reward_str, textOrgReward, font, rewardScale, rewardColor, rewardLineType)

        # Draw the action
        self.draw_action(dst, (1100, 300), action)

        cv2.imshow("live", cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def prep_image(self, image, scale=(1.0, 1.0)):
        import cv2
        if type(scale) is int:
            scale = (scale, scale)

        is_torch = hasattr(image, "cpu")
        if is_torch:
            image = image.detach().cpu().numpy()
            image = image.squeeze()
            if len(image.shape) == 3:
                image = image.transpose((1, 2, 0))

        image = image - np.min(image)
        image = image / (np.max(image) + 1e-9)

        # Only 2 channels - add another one
        if len(image.shape) == 3 and image.shape[2] == 2:
            newshape = list(image.shape)
            newshape[2] = 3
            new_img = np.zeros(newshape)
            new_img[:, :, 0:2] = image
            image = new_img

        # If we have too many channels, only show 3 of them
        if len(image.shape) > 2 and image.shape[2] > 3:
            image = image[:, :, 0:3]

        if scale != 1.0:
            width = image.shape[1]
            height = image.shape[0]
            image = cv2.resize(image, (int(scale[0]*width), int(scale[1]*height)), interpolation=cv2.INTER_LINEAR)

        if image.dtype == np.float64:
            image = image.astype(np.float32)

        #if len(image.shape) > 2 and (image.shape[2] == 3 or image.shape[2] == 4):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def is_torch(self, list_np_or_tensor):
        if hasattr(list_np_or_tensor, "cuda"):
            return True

    def overlay_pts_on_image(self, image, pts):
        """
        :param image: CxHxW image
        :param pts: Nx2 points - (H,W) coords in the image
        :return:
        """
        image = self.prep_image(image)
        if self.is_torch(pts):
            pts = pts.cpu().data.numpy()

        image[:, :, 0] = 0.0
        if pts is not None:
            for pt in pts:
                image[int(pt[0]), int(pt[1]), 0] = 1.0

        return image

    def blend_image(self, background, overlay):
        background = self.prep_image(background)
        overlay_t = self.prep_image(overlay)

        scale_x = int(background.shape[0] / overlay_t.shape[0])
        scale_y = int(background.shape[1] / overlay_t.shape[1])

        overlay = self.prep_image(overlay, scale=(scale_x, scale_y))
        out = background.copy()

        if len(overlay.shape) == 2:
            overlay = np.expand_dims(overlay, 2)

        alpha = np.clip(overlay.mean(2, keepdims=True) * 2, 0, 0.8)

        overlay = overlay * alpha
        out = out * (1 - alpha)

        overlay_channels = overlay.shape[2]

        out[:, :, 0:overlay_channels] += overlay

        return out

    def overlaid_image(self, background, overlay, gray_bg=False, channel=None, strength=0.7):
        background = self.prep_image(background)
        overlay_t = self.prep_image(overlay)

        scale_x = int(background.shape[0] / overlay_t.shape[0])
        scale_y = int(background.shape[1] / overlay_t.shape[1])

        overlay = self.prep_image(overlay, scale=(scale_x, scale_y))
        out = np.zeros_like(background)
        # Place the background
        if gray_bg:
            out[:, :, :] = np.expand_dims(np.mean(background, axis=2) * (1-strength), 2)
        else:
            out[:, :, :] = background * (1-strength)

        if len(overlay.shape) == 2:
            overlay = np.expand_dims(overlay, 2)

        overlay_channels = overlay.shape[2]

        # Add the overlay:
        if channel is None:
            out += overlay * strength
        else:
            out[:, :, channel:channel + overlay_channels] += overlay * strength

        return out

    def overlay_text(self, image, text_str):
        img_out = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.001 * image.shape[0]
        #fontColor = (255, 255, 255)
        fontColor = (1.0, 1.0, 1.0)
        lineType = 1

        lines = self.split_lines(text_str, maxchars=45)
        for i, line in enumerate(lines):
            textOrg = (15, 15 + int(35 * fontScale * i))
            cv2.putText(img_out, line, textOrg, font, fontScale, fontColor, lineType)
        return img_out

    def show_image(self, image, name="live", torch=False, waitkey=False, scale=1.0):
        if self.headless:
            return

        import cv2

        image = self.prep_image(image, scale)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(name, image)
        if type(waitkey) is int:
            cv2.waitKey(waitkey)
        elif waitkey:
            cv2.waitKey(0)
        else:
            cv2.waitKey(10)

    def save_image(self, image, name="live", torch=False, draw_point=None, scale=1.0, folder=""):

        image = self.prep_image(image, scale)

        if draw_point is not None:
            image[int(draw_point[0]), int(draw_point[1]), :] = np.array([1.0, 0, 1.0])

        if folder != "":
            os.makedirs(folder, exist_ok=True)

        scipy.misc.imsave(folder + "/" + name + ".png", image)

    def save_action(self, action, filename, folder):
        img = np.ones((420, 420, 3)).astype(np.uint8)
        self.draw_action(img, (10, 300), action)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
            folder += "/"
        scipy.misc.imsave(filename, img)

    def save_instruction(self, instruction, filename, torch=False, folder=""):
        if torch:
            instruction = debug_untokenize_instruction(instruction)

        if folder != "":
            os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, filename), "w") as fp:
            fp.write(instruction)

    def save_env_image(self, env_id, filename, folder):
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        try:
            shutil.copy(get_env_image_path(env_id), os.path.join(folder, filename))
        except Exception as e:
            print("Error saving env image!")
            print(e)

    def plot_path(self, env_id, paths, interactive=False, show=True, bg=True):
        if interactive:
            plt.ion()
            plt.clf()
        else:
            plt.ioff()

        #config_size = UnrealUnits().get_config_size()

        if bg:
            try:
                img = load_env_img(env_id)
                plt.imshow(img, extent=(0, config_size[0], 0, config_size[1]))
            except Exception as e:
                print("Error in loading and plotting path!")
                print(e)

        colors = ["r", "g", "b", "y", "c", "m"]

        for path, color in zip(paths, colors):
            # Note that x and y are swapped
            x_targets, y_targets = list(zip(*path))

            y_targets = config_size[1] - y_targets

            plt.plot(y_targets, x_targets, color)
            plt.plot(y_targets[-1], x_targets[-1], color+"o")

        plt.axis([0, config_size[0], 0, config_size[1]])

        if show:
            plt.show()
            plt.pause(0.0001)

    def plot_paths(self,
                   segment_dataset,
                   world_size,
                   segment_path=None,
                   file=None,
                   interactive=False,
                   bg=True,
                   texts=[],
                   entire_trajectory=False,
                   real_drone=False):

        if interactive:
            plt.ion()
        else:
            plt.ioff()

        if len(segment_dataset) == 0:
            print("Empty segment. Not plotting!")
            return

        path_key = "path" if entire_trajectory else "seg_path"

        md = segment_dataset[0]["metadata"] if "metadata" in segment_dataset[0] else segment_dataset[0]

        env_id = md["env_id"]
        if segment_path is None:
            segment_path = md[path_key]

        segment_path_px = (segment_path * 512 / world_size).astype(np.int32)
        #segment_path_px[:,0] = 512 - segment_path_px[:,0]

        if entire_trajectory:
            instructions = [segment_dataset[i]["instruction"] for i in range(len(segment_dataset))]
            unique_instructions = [instructions[0]]
            for instruction in instructions:
                if instruction != unique_instructions[-1]:
                    unique_instructions.append(instruction)
            instruction = "; ".join(unique_instructions)
        else:
            instruction = segment_dataset[0]["instruction"]

        if bg:
            try:
                img = load_env_img(env_id, width=512, height=512, real_drone=real_drone, origin_bottom_left=True)
                plt.imshow(img, extent=(0, 512, 0, 512))
            except Exception as e:
                print("Error in plotting paths!")
                print(e)

        plt.plot(segment_path_px[:,1], segment_path_px[:,0], "r")
        plt.plot(segment_path_px[-1,1], segment_path_px[-1,0], "ro")

        actual_path = []
        for sample in segment_dataset:
            actual_path.append(sample["state"].state[0:2])
        actual_path_px = (np.asarray(actual_path) * 512 / world_size).astype(np.int32)
        #actual_path_px[:,0] = 512 - actual_path_px[:,0]

        plt.plot(actual_path_px[:,1], actual_path_px[:, 0], "b")
        plt.plot(actual_path_px[-1,1], actual_path_px[-1, 0], "bo")

        plt.axis([0, 512, 0, 512])

        instruction_split = self.split_lines(instruction, maxchars=40)
        title = "\n".join(instruction_split)
        plt.title("env: " + str(env_id) + " - " + title)

        x = 10
        y = 5
        gap = 20
        for text in texts:
            if not DONT_DRAW_TEXT:
                plt.text(x, y, text)
            y += gap

        y += len(instruction_split) * gap
        for line in instruction_split:
            if not DONT_DRAW_TEXT:
                plt.text(x, y, line)
            y -= gap

        if interactive:
            plt.show()
            plt.pause(0.0001)

    def save_plot(self, filename):
        dir = os.path.dirname(filename)
        os.makedirs(dir, exist_ok=True)

        #print ("Saving plot in: ", filename + ".png")
        plt.savefig(filename + ".png")
        plt.clf()
        plt.close()

    def print_tokenized_instruction(self, instruction):
        instr_str = debug_untokenize_instruction(instruction)
        print("instruction: " + str(instr_str))

    def plot_confusion_matrix(self, predictions, labels, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(labels, predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(30, 20))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("confusion_matrix.jpg")
        print ("saved confusion matrix!")


def filter_for_gif(image):
    image = image - np.min(image)
    image = image / (np.max(image) + 1e-9)
    img_new = image * 255
    img_new = img_new.astype(np.uint8)
    img_new = np.clip(img_new, 0, 255)
    return img_new
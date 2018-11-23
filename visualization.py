import itertools
import json
import os
import shutil
import string
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.misc
from sklearn.metrics import confusion_matrix
from data_io.instructions import clean_instruction, debug_untokenize_instruction

from drones.airsim_interface.units import UnrealUnits
from data_io.env import load_env_img
from data_io.paths import get_env_image_path
from transformations import cf_to_img

FWD_MULTIPLIER = 0.5
ANG_MULTIPLIER = 2.0
ACTION_OPACITY = 160

DONT_DRAW_TEXT = True


class Presenter:

    def __init__(self):
        pass

    def show_instruction(self, instruction_str):
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

    def draw_action(self, image, offset, action):
        img2 = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
        offset = np.asarray(offset)

        rect_width = 40
        rect_length = 200

        left_rect_p1 = offset + np.asarray((0, 0))
        left_rect_p2 = offset + np.asarray((rect_length, rect_width))
        right_rect_p1 = offset + np.asarray((rect_length, 0))
        right_rect_p2 = offset + np.asarray((2*rect_length, rect_width))
        top_rect_p1 = offset + np.asarray((rect_length - rect_width / 2, 0))
        top_rect_p2 = offset + np.asarray((rect_length + rect_width / 2, -rect_length))

        turn_percent = -action[2] * ANG_MULTIPLIER
        fwd_percent = action[0] * FWD_MULTIPLIER

        turn_p1 = offset + np.asarray((rect_length, 0))
        turn_p2 = offset + np.asarray((rect_length + rect_length * turn_percent, rect_width))
        turn_p1 = turn_p1.astype(int)
        turn_p2 = turn_p2.astype(int)

        fwd_p1 = offset + np.asarray((rect_length - rect_width / 2, -rect_length / 2))
        fwd_p2 = offset + np.asarray((rect_length - rect_width / 2 + rect_width,
                                      -rect_length / 2 - rect_length * fwd_percent))
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
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.draw_action(image, (0, 100), action)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def show_sample(self, state, action, reward, command):
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
        instructionScale = 1.5
        instructionColor = (255, 255, 255)
        rewardScale = 1.0
        rewardColor = (255, 100, 100)
        lineType = 2
        rewardLineType = 2

        textOrgReward = (dst.shape[1] - 300, 40)

        command = ''.join(ch if ch not in set(string.punctuation) else "" for ch in command).strip().lower()
        wordlist = command.split(" ")
        wordlist = [word for word in wordlist if word != " "]
        command = " ".join(wordlist)

        lines = self.split_lines(command, maxchars=45)
        for i, line in enumerate(reversed(lines)):
            textOrg = (10, dst.shape[0] - 10 - int(35 * instructionScale * i))
            cv2.putText(dst, line, textOrg, font, instructionScale, instructionColor, lineType)

        reward_str = "Reward: {:5.1f}".format(reward)
        #cv2.putText(dst, reward_str, textOrgReward, font, rewardScale, rewardColor, rewardLineType)

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
            image = image.cpu().numpy()
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
            image = cv2.resize(image, (int(scale[0]*width), int(scale[1]*height)), interpolation=cv2.INTER_CUBIC)

        if image.dtype == np.float64:
            image = image.astype(np.float32)

        if len(image.shape) > 2 and (image.shape[2] == 3 or image.shape[2] == 4):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def overlaid_image(self, background, overlay, gray_bg=False, channel=None):
        background = self.prep_image(background)
        overlay = self.prep_image(overlay)

        scale_x = int(background.shape[0] / overlay.shape[0])
        scale_y = int(background.shape[1] / overlay.shape[1])

        overlay = self.prep_image(overlay, scale=(scale_x, scale_y))
        out = np.zeros_like(background)
        # Place the background
        if gray_bg:
            out[:, :, :] = np.expand_dims(np.mean(background, axis=2) * 0.7, 2)
        else:
            out[:, :, :] = background * 0.7

        if len(overlay.shape) == 2:
            overlay = np.expand_dims(overlay, 2)

        overlay_channels = overlay.shape[2]

        # Add the overlay:
        if channel is None:
            out += overlay
        else:
            out[:, :, channel:channel + overlay_channels] += overlay

        return out

    def overlay_text(self, image, text_str):
        img_out = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        lines = self.split_lines(text_str, maxchars=45)
        for i, line in enumerate(lines):
            textOrg = (30, 30 + int(35 * fontScale * i))
            cv2.putText(img_out, line, textOrg, font, fontScale, fontColor, lineType)
        return img_out

    def show_image(self, image, name="live", torch=False, waitkey=False, scale=1.0):
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

        config_size = UnrealUnits().get_config_size()

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

    def plot_paths(self, segment_dataset, segment_path=None, file=None, interactive=False, bg=True, texts=[], entire_trajectory=False):
        if interactive:
            plt.ion()
        else:
            plt.ioff()

        if len(segment_dataset) == 0:
            print("Empty segment. Not plotting!")
            return

        path_key = "path" if entire_trajectory else "seg_path"

        env_id = segment_dataset[0]["metadata"]["env_id"]
        if segment_path is None:
            segment_path = segment_dataset[0]["metadata"][path_key]

        config_size = UnrealUnits().get_config_size()
        y_targets, x_targets = list(zip(*cf_to_img(segment_path, [512, 512])))
        y_targets = np.asarray(y_targets) * config_size[1] / 512
        x_targets = np.asarray(x_targets) * config_size[0] / 512
        y_targets = config_size[1] - y_targets
        #x_targets = CONFIG_SIZE[1] - x_targets

        # Note that x and y are swapped
        #x_targets, y_targets = list(zip(*segment_path))

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
                img = load_env_img(env_id)
                plt.imshow(img, extent=(0, config_size[0], 0, config_size[1]))
            except Exception as e:
                print("Error in plotting paths!")
                print(e)

        #y_targets = CONFIG_SIZE[1] - y_targets
        plt.plot(x_targets, y_targets, "r")
        plt.plot(x_targets[-1], y_targets[-1], "ro")

        # Plot segment endpoints
        #for segment in segment_dataset:
        #    end = segment.metadata["seg_path"][-1]
        #    end_x = end[0]
        #    end_y = CONFIG_SIZE[1] - end[1]
        #    plt.plot(end_y, end_x, "ro")

        x_actual = []
        y_actual = []
        for sample in segment_dataset:
            x_actual.append(sample["state"].state[0])
            y_actual.append(sample["state"].state[1])
        x_actual = np.asarray(x_actual)
        y_actual = np.asarray(y_actual)

        """if len(segment_dataset) > 0:
            instruction, drone_states, actions, rewards, finished = zip(*segment_dataset)
            drone_states = np.asarray(drone_states)
            x_actual = drone_states[:, 0]
            y_actual = drone_states[:, 1]"""

        plt.plot(x_actual, y_actual, "b")
        plt.plot(x_actual[-1], y_actual[-1], "bo")

        plt.axis([0, config_size[0], 0, config_size[1]])

        instruction_split = self.split_lines(instruction, maxchars=40)
        title = "\n".join(instruction_split)

        plt.title("env: " + str(env_id) + " - " + title)

        x = 10
        y = 5
        for text in texts:
            if not DONT_DRAW_TEXT:
                plt.text(x, y, text)
            y += 40

        y += len(instruction_split) * 40 + 40
        for line in instruction_split:
            if not DONT_DRAW_TEXT:
                plt.text(x, y, line)
            y -= 40

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
import sys
import os
import threading
import rospy
import numpy as np
from time import sleep

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from data_io.env import load_and_convert_env_config, load_env_img
from data_io.paths import get_ceiling_cam_calibration_path
from drones.aero_interface.ros_node import init_node_if_necessary
from drones.aero_interface.landmark_colors import color_names, colors

from utils.text2speech import say, repeat, t2s

from visualization import Presenter

from pykeyboard import PyKeyboardEvent

# This allows skipping environment configuration for debugging purposes
SKIP_CONFIGURATION = False


class EnterMonitor(PyKeyboardEvent):

    def __init__(self):
        PyKeyboardEvent.__init__(self)
        self.tapped = False

    def tap(self, keycode, c, press):
        '''Monitor Super key.'''
        print("TAP")
        print(c)
        if c == "Return":
            self.tapped = True

class MonitorRunner():

    def __init__(self, monitor):
        self.mon = monitor
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        self.mon.run()

class LandmarkConfigurator():
    """
    This class prompts the user to place landmarks in correct locations before returning control to the drone
    """
    def __init__(self):
        init_node_if_necessary()
        self.subscriber = None
        self.state_positioning = False
        self.state_instructions_printed = False
        self.env_config = None
        self.last_prompt_time = rospy.Time(0)
        self.image_to_show = None
        self.new_image = False

        self.enter_monitor = None
        self.monitor_runner = None

        # Calibration of rectified ceiling camera
        self.calib_s = 470.0
        self.m_s = 4.7
        self.calib_coords_m = [[0.0, 0.0], [0.0, self.m_s], [self.m_s, 0.0], [self.m_s, self.m_s]]
        self.calib_coords_px = [[0.0, 0.0], [0.0, self.calib_s], [self.calib_s, 0.0], [self.calib_s, self.calib_s]]
        self.calib_coords_px_in = []
        self.lines_to_draw = []

        # Loading calibration results:
        # TODO: Save and load from file
        self.img_topic = "/ceiling_cam/image_rect_color"
        path = get_ceiling_cam_calibration_path()
        if os.path.exists(path + "_img2img.npy"):
            self.F_cam_p1_to_img_p2 = np.load(path + "_img2img.npy")
            self.F_world_m_to_img_p2 = np.load(path + "_world2img.npy")
            print("Loaded camera calibration: ")
            print(self.F_cam_p1_to_img_p2)
        else:
            self.F_cam_p1_to_img_p2 = None
            self.F_world_m_to_img_p2 = None
            print("Ceiling cam calibration missing. Computing!")
            self.compute_and_save_f_matrix()

    def compute_and_save_f_matrix(self):
        cv2.namedWindow("ui")
        cv2.setMouseCallback("ui", self._opencv_click_callback)

        self.subscriber = rospy.Subscriber(self.img_topic, Image, self._calibration_image_callback)
        current_calib_len = -1
        while True:
            if self.new_image:
                if self.F_cam_p1_to_img_p2 is None:
                    for pt1,pt2 in self.lines_to_draw:
                        cv2.line(self.image_to_show, tuple(pt1), tuple(pt2), color=(255,0,255))
                    cv2.imshow("ui", self.image_to_show)
                    cv2.waitKey(20)
                    if len(self.calib_coords_px_in) > current_calib_len:
                        if len(self.calib_coords_px_in) == len(self.calib_coords_m):
                            p1 = np.asarray(self.calib_coords_px_in).astype(np.float32)
                            p2 = np.asarray(self.calib_coords_px).astype(np.float32)
                            p3 = np.asarray(self.calib_coords_m).astype(np.float32)
                            self.F_cam_p1_to_img_p2 = cv2.getPerspectiveTransform(p1, p2)
                            self.F_world_m_to_img_p2 = cv2.getPerspectiveTransform(p3, p2)
                            print("Calibration img2img:")
                            print(self.F_cam_p1_to_img_p2)
                            print("Calibration world2img:")
                            print(self.F_world_m_to_img_p2)
                            path = get_ceiling_cam_calibration_path()
                            np.save(path + "_img2img.npy", self.F_cam_p1_to_img_p2)
                            np.save(path + "_world2img.npy", self.F_world_m_to_img_p2)
                            print(f"Saved to {path}")
                        else:
                            current_calib_len = len(self.calib_coords_px_in)
                            print(f"Click on the image the world point: {self.calib_coords_m[current_calib_len]}")
                else:
                    # Show the calibration results
                    proj = self.project_image(self.image_to_show, self.F_cam_p1_to_img_p2)
                    cv2.imshow("result", proj)
                    cv2.waitKey(0)
                    break


    def project_image(self, img_in, F):
        img_out = cv2.warpPerspective(img_in, F, (int(self.calib_s), int(self.calib_s)))
        return img_out

    def configure_landmarks(self, env_id):
        self.env_config = load_and_convert_env_config(env_id)
        self.state_positioning = True
        self.state_instructions_printed = False

        self.subscriber = rospy.Subscriber(self.img_topic, Image, self._image_callback)

        self.enter_monitor = EnterMonitor()
        self.monitor_runner = MonitorRunner(self.enter_monitor)

        env_sim_img = load_env_img(env_id, width=400, height=400, real_drone=False, origin_bottom_left=False)

        new = True
        while True:
            if self.new_image:
                Presenter().show_image(self.image_to_show, "Landmark Positioning", scale=2, waitkey=10)
                Presenter().show_image(env_sim_img, "Sim Image", scale=2, waitkey=10)
                if new:
                    cv2.moveWindow("Landmark Positioning", 20,20)
                    cv2.moveWindow("Sim Image", 1000,20)
                    new = False
            if self.enter_monitor.tapped or SKIP_CONFIGURATION:
                break

        sleep(1)
        cv2.destroyWindow('Landmark Positioning')
        cv2.destroyWindow("Sim Image")
        self.subscriber.unregister()
        return self.image_to_show

    def _opencv_click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.calib_coords_px_in.append([x,y])
        elif event == cv2.EVENT_MOUSEMOVE:
            # Draw polygon connecting this new point with each of the existing points:
            self.lines_to_draw = []
            for existing_pt in self.calib_coords_px_in:
                self.lines_to_draw.append([existing_pt, [x,y]])

    def _pad_image(self, img):
        PADDING = 200
        new_shape = list(img.shape)
        new_shape[0] += 2 * PADDING
        new_shape[1] += 2 * PADDING
        img_array_out = np.zeros(new_shape).astype(img.dtype)
        img_array_out[PADDING:new_shape[0]-PADDING, PADDING:new_shape[1]-PADDING, :] = img
        return img_array_out

    def _calibration_image_callback(self, img):
        bridge = CvBridge()
        image_array = np.asarray(bridge.imgmsg_to_cv2(img, "rgb8"))
        self.image_to_show = self._pad_image(image_array)
        self.new_image = True

    def _image_callback(self, img):
        if not rospy.is_shutdown():
            bridge = CvBridge()
            image_array = np.asarray(bridge.imgmsg_to_cv2(img, "rgb8"))
            image_array = self._pad_image(image_array)
            image_array = self.project_image(image_array, self.F_cam_p1_to_img_p2)

            names_and_colors = []
            for i,name in enumerate(self.env_config["landmarkName"]):
                x_m, y_m = self.env_config["x_pos_as"][i], self.env_config["y_pos_as"][i]
                x_px = int(x_m * self.calib_s / self.m_s)
                y_px = int(y_m * self.calib_s / self.m_s)
                #lm_pos_m = np.asarray([x, y, 1])
                #lm_pos_px = np.dot(self.F_world_m_to_img_p2, lm_pos_m).astype(np.int32)
                color = colors[i]
                color_name = color_names[i]
                names_and_colors.append((name, color_name))
                cv2.circle(image_array, (y_px, x_px), 5, color, -1)
                cv2.putText(image_array, name, (y_px, x_px), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            prompt = ". \n".join([f"{lm} is {c}" for lm,c in names_and_colors])

            if not self.state_instructions_printed:
                print(prompt)
                #t2s(prompt)
                self.state_instructions_printed = True
            if (rospy.get_rostime() - self.last_prompt_time).to_sec() > 10:
                #repeat(dontblock=True)
                self.last_prompt_time = rospy.get_rostime()

            self.image_to_show = image_array
            self.new_image = True

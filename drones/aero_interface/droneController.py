#!/home/drone/catkin_ws_py3/venv/bin/python
import rospy
from time import sleep
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry import vec_to_yaw

import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

from cv_bridge import CvBridge
import cv2

from utils.simple_profiler import SimpleProfiler
import drones.aero_interface.ros_node as rn
from drones.aero_interface.landmarkConfigurator import LandmarkConfigurator
from drones.droneControllerBase import DroneControllerBase
from drones.aero_interface.rviz import RvizInterface
from data_io.env import configs_equal, load_and_convert_env_config, save_env_img, load_env_img
from drones.aero_interface.find_landing_location import find_safe_landing_location
from drones.aero_interface.camcorder import Camcorder
from drones.rollout_exception import RolloutException

import parameters.parameter_server as P

model = None
PROFILE = True

MAX_LATENCY_TO_START_ROLLOUT = 0.2
MAX_LATENCY_DURING_ROLLOUT = 0.5

instructions_printed = False

class DroneController(DroneControllerBase):

    def __init__(self, instance=0, flight_height=1.0):
        super(DroneController, self).__init__()
        rn.init_node_if_necessary()
        self.instance = instance
        self.flight_height = flight_height
        self.env_id = -1
        self.env_config = None
        self.prof = SimpleProfiler(print=PROFILE)
        self.count = 0
        self.last_img = None
        self.last_img_delay = None
        self.last_drn_pos = None
        self.last_drn_rot = None
        self.landed = True
        self.seg_idx = None

        self.clock_speed = self._read_clock_speed()
        if self.clock_speed != 1.0:
            print(f"WARNING: Using real drone but clock speed is set to {self.clock_speed} not 1.0")
            raise ValueError("Not going to fly with wrong clockspeed")

        self.landmark_configurator = LandmarkConfigurator()

        self.camcorder1 = Camcorder(instance=1)

        self.passive_mode = P.get_current_parameters()["Setup"].get("passive_mode")
        self.img_w = P.get_current_parameters()["AirSim"]["CaptureSettings"][0]["Width"]
        self.img_h = P.get_current_parameters()["AirSim"]["CaptureSettings"][0]["Height"]

        self.dataloader_rollout = None
        self.rviz = RvizInterface(
            base_name="/drone_controller/",
            posearray_topics=["flight_history", "cmd_history_lin", "cmd_history_ang"],
            markerarray_topics=["env_config"]
        )

        # ROS pub/sub
        self.tf_listener = self._init_tf_listener()
        rospy.Subscriber("/fpv_cam/image_rect_color", Image, self._image_callback, queue_size=1)
        #self.fpv_image_pub = rospy.Publisher("/drone_controller/fpv_img_rcv", Image, queue_size=1)
        self.fpv_image_pub = False
        self.velocity_target_pub = rospy.Publisher("/drone_controller_safe/cmd_vel", TwistStamped, queue_size=1)
        self.landing_publisher = rospy.Publisher("/drone_controller_safe/landing", Bool, queue_size=1)
        self.pose_publisher = rospy.Publisher("/drone_controller_safe/pose_setpoint", PoseStamped, queue_size=1)

        self._wait_ros_boot()

    def _read_clock_speed(self):
        speed = 1.0
        if "ClockSpeed" in P.get_current_parameters()["AirSim"]:
            speed = P.get_current_parameters()["AirSim"]["ClockSpeed"]
        print("Read clock speed: " + str(speed))
        return speed

    def _init_tf_listener(self):
        """
        Initialize ROS TF Listener and wait until the full transform chain from /map_ned to the camera frame
        is available (e.g. wait for other systems to boot, or for transform history to accumulate in the listener)
        :return:
        """
        tf_listener = tf.TransformListener()
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            r.sleep()
            try:
                tf_listener.lookupTransform('/map_ned', '/camera_airsim_base_link', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("Waiting for TF: /map_ned -> /camera_airsim_base_link")
                continue
        return tf_listener

    def _wait_ros_boot(self):
        rate = rospy.Rate(2)
        while not self._is_ros_system_healty():
            print("Waiting for ROS setup to be online")
            rate.sleep()

    def _is_ros_system_healty(self):
        """
        Check that ROS system is up and functional.
        :return: True if it is, False if problems were found
        """
        healthy = True
        if self.last_img is None:
            print("No image received from drone")
            healthy = False
        return healthy

    def _image_callback(self, img):
        delay = (rospy.get_rostime() - img.header.stamp).to_sec()
        if delay > 0.3:
            print(f"IMAGE RECV DELAY: {delay}")
        self.last_img = img
        self.last_img_delay = delay
        if self.fpv_image_pub:
            self.fpv_image_pub.publish(img)

    def _is_close(self, position, yaw, target_position, target_yaw, pos_thres=0.2, yaw_thres=0.05):
        """
        Check if current pose of the drone is close enough to target pose before
        """
        # maximum distance between current position and target position
        max_dist = pos_thres if not self.passive_mode else 4
        pos_is_close = np.linalg.norm(np.array(position) - np.array(target_position)) < max_dist
        z_diff = position[2] - target_position[2]
        z_diff_abs = max(z_diff, -z_diff)

        pos_is_close = pos_is_close and z_diff_abs < 0.1

        # maximum difference between current yaw and target yaw
        max_yaw_diff = yaw_thres if not self.passive_mode else 4 #1
        angle_is_close = min((2 * np.pi) - np.abs(yaw - target_yaw), np.abs(yaw - target_yaw)) < max_yaw_diff

        return pos_is_close, angle_is_close

    def _get_converted_image(self):
        bridge = CvBridge()
        np_image = bridge.imgmsg_to_cv2(self.last_img, "rgb8")
        correct_size_img = cv2.resize(np_image, dsize=(self.img_w, self.img_h))
        delay = (rospy.get_rostime() - self.last_img.header.stamp).to_sec()
        if delay > 0.2:
            print(f"IMAGE OUT DELAY: {delay}")
        #image_pil = PIL_Image.fromarray()
        #image_tensor = to_tensor(image_pil)
        return correct_size_img

    def _get_pos(self):
        current_trans, current_rot = self.tf_listener.lookupTransform('/map_ned', '/base_link', rospy.Time(0))
        #if current_trans[3] > 0.2 and self.landed:
        #    print("Drone not on ground! Assuming it is flying!")
        #    self.landed = False
        return current_trans

    def _get_yaw(self):
        current_trans, current_rot = self.tf_listener.lookupTransform('/map_ned', '/base_link', rospy.Time(0))
        #_, current_rot = self.tf_listener.lookupTransform('/map_ned', '/base_link', rospy.Time(0))
        current_rot_euler = euler_from_quaternion(current_rot)
        current_yaw = current_rot_euler[2]
        return current_yaw

    def _get_state_vector(self):
        #(trans, rot) = listener.lookupTransform('/map_ned', '/base_link', img.header.stamp)
        #position = torch.tensor(trans)
        #orientation = torch.tensor([rot[3]] + rot[:3])
        # TODO: Figure out if we need map or map_ned frame
        (trans_cam, cam_rot) = self.tf_listener.lookupTransform('/map_ned', '/camera_airsim_base_link', rospy.Time(0))
        (trans_drn, drn_rot) = self.tf_listener.lookupTransform('/map_ned', '/base_link', rospy.Time(0))
        #print("'/map_ned', '/camera_airsim_base_link'", trans_cam)

        self.rviz.add_pose_and_publish_array("flight_history", trans_drn, drn_rot)
        self.last_drn_pos = trans_drn
        self.last_drn_rot = drn_rot

        #drn_rot = [drn_rot[3]] + drn_rot[:3]
        drn_rot_euler = euler_from_quaternion(drn_rot)
        # Convert quaternion from ROS xyzw to wxyz used elsewhere
        cam_rot = [cam_rot[3]] + cam_rot[:3]
        # TODO: Grab velocity estimate
        drn_vel = [0, 0, 0]

        drone_state = np.concatenate((trans_drn, drn_rot_euler, drn_vel, trans_cam, cam_rot), axis=0)
        return drone_state

    def get_real_time_rate(self):
        return self.clock_speed

    def visualize_twist_as_pose(self, cmd_vel_twist):
        last_pos = np.asarray(self.last_drn_pos)
        vel_vec = np.asarray([cmd_vel_twist.linear.x, cmd_vel_twist.linear.y, 0])
        next_pos = last_pos + vel_vec
        yaw = vec_to_yaw(next_pos - last_pos)
        yaw_rot = yaw + 0.5* cmd_vel_twist.angular.z
        quat = quaternion_from_euler(0, 0, yaw)
        quat_rot = quaternion_from_euler(0, 0, yaw_rot)
        self.rviz.add_pose_and_publish_array("cmd_history_lin", last_pos, quat)
        self.rviz.add_pose_and_publish_array("cmd_history_ang", last_pos, quat_rot)

    def send_local_velocity_command(self, cmd_vel):
        # Convert the drone reference frame command to a global command as understood by AirSim
        #print(f"send_local_velocity_command command: {cmd_vel}")
        cmd_vel = self._action_to_global(cmd_vel)
        #print(f"            global velocity command: {cmd_vel}")

        cmd_linear = cmd_vel[:2]
        cmd_yaw = cmd_vel[2]

        # AirSim uses yaw in angles (why!), ROS uses yaw in radians
        #cmd_yaw = cmd_yaw * 3.14159 / 180

        cmd_vel_twist = TwistStamped()
        cmd_vel_twist.header.frame_id = "/map_ned"
        cmd_vel_twist.header.stamp = rospy.get_rostime()
        cmd_vel_twist.twist.linear.x = cmd_linear[0]   #cmd_vel[0]
        cmd_vel_twist.twist.linear.y = cmd_linear[1]  #cmd_vel[1]
        cmd_vel_twist.twist.linear.z = 0

        cmd_vel_twist.twist.angular.x = 0
        cmd_vel_twist.twist.angular.y = 0
        cmd_vel_twist.twist.angular.z = cmd_yaw  #cmd_vel[2]

        self.visualize_twist_as_pose(cmd_vel_twist.twist)
        #print("sending velocity command: ")
        #print(cmd_vel_twist)
        self.landed = False

        if self.last_img_delay > MAX_LATENCY_DURING_ROLLOUT:
            cmd_vel_twist.twist.linear.x = 0
            cmd_vel_twist.twist.linear.y = 0
            cmd_vel_twist.twist.linear.z = 0
            if not self.passive_mode:
                self.velocity_target_pub.publish(cmd_vel_twist)
            raise RolloutException(f"Image Latency {self.last_img_delay} exceeded limit of {MAX_LATENCY_DURING_ROLLOUT}")

        if not self.passive_mode:
            self.velocity_target_pub.publish(cmd_vel_twist)

    def land(self, new_env_config=None):
        # If we're actually in the air, first fly to a safe landing location away from landmarks
        cpos = self._get_pos()
        height = -cpos[2]
        if height > 0.2:
            # Find a safe landing position that does not land on top of any of the landmarks in old or new configs
            # Edit: It's ok to land on top of landmarks in the new configuration. The drone is not heavy to move.
            loc = find_safe_landing_location(self.env_config, None, current_pos=self._get_pos())
            self.teleport_to(loc, 0, allow_latency=True)

        # Then land
        land_msg = Bool()
        land_msg.data = True

        self.landing_publisher.publish(land_msg)
        print("SENT LANDING SIGNAL")
        self.landed = True

    def set_current_env_id(self, env_id, instance_id=None):
        # If the new environment has a different arrangement of landmarks, we need the user to physically move them
        new_env_config = load_and_convert_env_config(env_id)
        if self.env_config is None:
            self.env_config = new_env_config
        self.rviz.publish_env_config("env_config", new_env_config)
        if not configs_equal(env_id, self.env_id) and \
                not P.get_current_parameters()["Setup"].get("dont_place_landmarks"):
            # Land the drone first, otherwise it's unsafe to walk inside the cage
            if not self.landed:
                self.land(new_env_config)
            # Wait for the user to place the landmarks
            env_img = self.landmark_configurator.configure_landmarks(env_id)
        else:
            env_img = load_env_img(self.env_id, real_drone=True, origin_bottom_left=False)
        save_env_img(env_id, env_img, real_drone=True)

        self.env_id = env_id
        self.env_config = new_env_config

    def set_current_seg_idx(self, seg_idx, instance_id=None):
        self.seg_idx = seg_idx

    def rollout_begin(self, instruction):
        self.camcorder1.start_recording_rollout(P.get_current_parameters()["Setup"]["run_name"], self.env_id, 0, self.seg_idx, caption=instruction)

    def rollout_end(self):
        self.camcorder1.stop_recording_rollout()

    def teleport_to(self, position, yaw, allow_latency=False):
        self.landed = False
        # TODO: Remove override
        #position = [500., 500., self.flight_height]
        position = list(position)
        if len(position) == 2:
            position.append(-self.flight_height)

        ps = PoseStamped()
        ps.header.stamp = rospy.get_rostime()
        ps.header.frame_id = "/map_ned"
        ps.pose.position.x = position[0]
        ps.pose.position.y = position[1]
        ps.pose.position.z = position[2] - 0.1

        cam_rot_quat = quaternion_from_euler(0, 0, float(yaw))

        ps.pose.orientation.x = cam_rot_quat[0]
        ps.pose.orientation.y = cam_rot_quat[1]
        ps.pose.orientation.z = cam_rot_quat[2]
        ps.pose.orientation.w = cam_rot_quat[3]

        #print("target position", ps)
        if not self.passive_mode:
            self.pose_publisher.publish(ps)

        sleep(1)
        ps.pose.position.z = position[2]

        freq = 5
        #rospy.sleep(2)
        rate = rospy.Rate(freq)
        print("Teleporting ...")
        while not rospy.is_shutdown():
            self.pose_publisher.publish(ps)
            #print("wait for position")
            cam_trans, cam_rot = self.tf_listener.lookupTransform('/map_ned', '/camera_airsim_base_link', rospy.Time(0))
            drn_trans, drn_rot = self.tf_listener.lookupTransform("/map_ned", "/base_link", rospy.Time(0))
            rate.sleep()
            drn_rot_euler = tf.transformations.euler_from_quaternion(drn_rot)
            drn_yaw = drn_rot_euler[2]
            #pos_is_nearby, _ = self._is_close(drn_trans, drn_yaw, position, yaw, pos_thres=0.5)
            pos_is_close, angle_is_close = self._is_close(drn_trans, drn_yaw, position, yaw)
            #print("Position is close: {}. Angle is close: {}".format(pos_is_close, angle_is_close))
            #print("Position",cam_trans, position)
            #print("Yaw", drn_yaw, yaw)

            if pos_is_close and angle_is_close:
                print("Teleport complete")
                sleep(1)
                if self.last_img_delay > MAX_LATENCY_TO_START_ROLLOUT:
                    print(f"Image latency {self.last_img_delay} too high to begin rollout. Hovering and waiting for it to drop below {MAX_LATENCY_TO_START_ROLLOUT}")
                else:
                    return True

        self.landed = False

    def teleport_3d(self, position, rpy, pos_in_airsim=True, fast=False):
        raise NotImplementedError("Teleport 3D not available on the real drone. Move it yourself.")

    def get_state(self, depth=True, segmentation=False):
        drone_state = self._get_state_vector()
        img_state = self._get_converted_image()
        return drone_state, img_state

    def reset_environment(self):
        self.rviz.reset_pose_array("flight_history")
        self.rviz.reset_pose_array("cmd_history_lin")
        self.rviz.reset_pose_array("cmd_history_ang")

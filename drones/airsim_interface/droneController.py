import json
import subprocess
import msgpackrpc.error

from data_io.env import load_env_config
from data_io.helpers import save_json
from drones.droneControllerBase import DroneControllerBase
from drones.airsim_interface.airsimClientNew import *
from data_io.units import DEPTH_SCALE
from transforms3d import euler

import gc
import data_io.paths as paths
from drones.airsim_interface.rate import Rate
import parameters.parameter_server as P

# note - airsim/simpleflight do not move anything if under 1.5 dist in any axis!.
# note2 - 0 is above ground, +3 is ground.

# Y-NED range is [0, 30]
# X-NED range is [0. 30]
from drones.rollout_exception import RolloutException

PORTS = {
    0: 10000,
    1: 10001,
    2: 10002,
    3: 10003,
    4: 10004,
    5: 10005,
    6: 10006,
    7: 10007,
    8: 10008,
    9: 10009,
    10: 10010,
    11: 10011,
    12: 10012,
    13: 10013,
    14: 10014,
    15: 10015
}


def spawn_headless_worker(instance_id, port):
    print(f" . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' ")
    print(f"    >>>>> Spawning a Headless Worker: {instance_id}")
    print(f" . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' . ' ")
    command = paths.get_sim_executable_path() + " WorkerID " + str(instance_id) + " ApiServerPort " + str(port)
    subprocess.Popen(command, env=os.environ, shell=True)


def spawn_worker(instance_id, port):
    res_x = 640
    res_y = 480
    # Set resolution from config
    if "Simulator" in P.get_current_parameters():
        res_x = P.get_current_parameters()["Simulator"].get("window_x") or res_x
        res_y = P.get_current_parameters()["Simulator"].get("window_y") or res_y
    pos_x = (res_x + 20) * instance_id
    res_x = str(res_x)
    pos_x = str(pos_x)
    res_y = str(res_y)

    command = 'gnome-terminal -x ' + paths.get_sim_executable_path() + " -WINDOWED -ResX=" + res_x + " -ResY=" + res_y + \
              " -FPS=10" + " -WinX=" + pos_x + " -WinY=50"
    command += " WorkerID " + str(instance_id) + " ApiServerPort " + str(port)
    subprocess.Popen(command, env=os.environ, shell=True)


def spawn_workers(num_workers):
    for i in range(num_workers):
        port = PORTS[i]
        if P.get_current_parameters()["Environment"].get("headless"):
             spawn_headless_worker(i, port)
        else:
            spawn_worker(i, port)


def startAirSim(controller, instance_id, port):
    try:
        controller._start()
    except Exception as e:
        try:
            print(e)
            print("Failed to connect to client on port " + str(port) + "! Starting new AirSim...")
            if P.get_current_parameters()["Environment"].get("headless"):
                spawn_headless_worker(instance_id, port)
            else:
                spawn_worker(instance_id, port)

            time.sleep(20)
            controller._start()
        except Exception as e:
            killAirSim(do_kill=True)
            print("Failed to connect to client on port " + str(port) + "! Starting new AirSim...")
            if P.get_current_parameters()["Environment"].get("headless"):
                spawn_headless_worker(instance_id, port)
            else:
                spawn_worker(instance_id, port)
            time.sleep(20)
            controller._start()


def killAirSim(do_kill=False):
    if do_kill:
        os.system("killall -9 MyProject5-Linux-Shipping")


# All functions with the @safeCall decorator are called from outside and use the logical (config) units
# All methods without the @safeCall decorator are called from DroneController and use AirSim units
class DroneController(DroneControllerBase):
    def __init__(self, instance=0, flight_height=100.0):
        super(DroneController, self).__init__()
        self.instance = instance
        self.port = PORTS[instance]
        self.clock_speed = self._read_clock_speed()
        self.control_interval = 5.0
        self.flight_height = -flight_height

        print("DroneController Initialize")
        self.rate = Rate(0.1 / self.clock_speed)
        self.height_offset = P.get_current_parameters()["DroneController"].get("start_height_offset")

        self.samples = []

        killAirSim()
        self._write_airsim_settings()
        startAirSim(self, instance, self.port)

    def _get_config(self, name, i=None, instance_id=None):
        config = load_env_config(id)
        return config

    def _set_config(self, json_data, subfolder, name, i=None, instance_id=None):
        folder = paths.get_current_config_folder(i, instance_id)
        id = "" if i is None else "_" + str(i)
        path = os.path.join(paths.get_sim_config_dir(), folder, subfolder, name + id + ".json")

        if not os.path.isdir(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except Exception:
                pass

        if os.path.isfile(path):
            os.remove(path)

        # print("Dumping config in: " + path)
        with open(path, 'w') as fp:
            json.dump(json_data, fp)

        subprocess.check_call(["touch", path])

    def _load_drone_config(self, i, instance_id=None):
        conf_json = load_env_config(i)
        self._set_config(conf_json, "", "random_config", i=None, instance_id=instance_id)

    def _start_call(self):
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception as e:
            import traceback
            from utils.colors import print_error
            print_error("Error encountered during policy rollout!")
            print_error(e)
            print_error(traceback.format_exc())

            startAirSim(self, self.instance, self.port)

            raise RolloutException("Exception encountered during start_call. Rollout should be re-tried")

    def _start(self):
        self.client = MultirotorClient('127.0.0.1', self.port)
        self.client.confirmConnection()
        self._start_call()

        # got to initial height
        pos_as = self.client.getPosition()
        pos_as = [pos_as.x_val, pos_as.y_val, pos_as.z_val]
        pos_as[2] = self.flight_height - self.height_offset
        #pos = self.units.pos3d_from_as(pos_as)
        #pos[2] = self.units.height_to_as(self.flight_height)
        print ("Telporting to initial position in cfg: ", pos_as)
        self.teleport_3d(pos_as, [0, 0, 0])

    def _read_clock_speed(self):
        speed = 1.0
        if "ClockSpeed" in P.get_current_parameters()["AirSim"]:
            speed = P.get_current_parameters()["AirSim"]["ClockSpeed"]
        print("Read clock speed: " + str(speed))
        return speed

    def _write_airsim_settings(self):
        airsim_settings = P.get_current_parameters()["AirSim"]
        airsim_settings_path = P.get_current_parameters()["Environment"]["airsim_settings_path"]
        airsim_settings_path = os.path.expanduser(airsim_settings_path)
        save_json(airsim_settings, airsim_settings_path)
        print("Wrote new AirSim settings to " + str(airsim_settings_path))

    """def _action_to_global(self, action):
        drone_yaw = self.get_yaw()
        action_global = np.zeros(3)
        # TODO: Add action[1]
        action_global[0] = action[0] * math.cos(drone_yaw)
        action_global[1] = action[0] * math.sin(drone_yaw)
        action_global[2] = action[2]
        return action_global
    """

    def _try_teleport_to(self, position, yaw, pitch=None, roll=None):
        #pos_as = self.units.pos2d_to_as(position[:2])
        #yaw_as = self.units.yaw_to_as(yaw)
        #tgt_h = self.units.height_to_as(self.flight_height)

        pos_as = position[:2]
        yaw_as = yaw
        tgt_h = self.flight_height - self.height_offset

        tele_orientation = self.client.toQuaternion(0, 0, yaw_as)
        self.send_local_velocity_command([0, 0, 0])

        tele_pos = Vector3r()
        tele_pos.x_val = pos_as[0]
        tele_pos.y_val = pos_as[1]

        uphigh = True
        if uphigh:
            self.rate.sleep_n_intervals(3)
            pos1 = self.client.getPosition()
            current_z = pos1.z_val
            pos1.z_val = tgt_h - 50
            self.client.simSetPose(pos1, tele_orientation)
            self.rate.sleep_n_intervals(2)
            tele_pos.z_val = tgt_h - 50
            self.client.simSetPose(tele_pos, tele_orientation)
            self.rate.sleep_n_intervals(2)

        tele_pos.z_val = tgt_h
        self.client.simSetPose(tele_pos, tele_orientation)
        self.send_local_velocity_command([0, 0, 0])
        self.rate.sleep_n_intervals(10)
        #time.sleep(0.2)

    def _get_yaw(self):
        o = self.client.getOrientation()
        roll, pitch, yaw = euler.quat2euler([o.w_val, o.x_val, o.y_val, o.z_val])
        return yaw

    def get_real_time_rate(self):
        return self.clock_speed

    def send_local_velocity_command(self, cmd_vel):
        """
        :param cmd_vel: x in m/s and yaw rate in rad/s
        :return:
        """
        try:
            # Convert the drone reference frame command to a global command as understood by AirSim
            cmd_vel = self._action_to_global(cmd_vel)
            if len(cmd_vel) == 3:
                vel_x = cmd_vel[0]
                vel_y = cmd_vel[1]
                yaw_rate = cmd_vel[2] * 180 / 3.14
                yaw_mode = YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                tgt_h = self.flight_height - self.height_offset
                self.client.moveByVelocityZ(vel_x, vel_y, tgt_h, self.control_interval, yaw_mode=yaw_mode)
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")

    def teleport_to(self, position, yaw):
        #pos_as = self.units.pos2d_to_as(position[0:2])
        #tgt_h = self.units.height_to_as(self.flight_height)
        try:
            pos_as = position[0:2]
            tgt_h = self.flight_height - self.height_offset

            teleported = False
            for i in range(10):
                self._try_teleport_to(position, yaw)
                curr_pos = self.client.getPosition()
                curr_pos_np = np.asarray([curr_pos.x_val, curr_pos.y_val])

                if np.linalg.norm(pos_as - curr_pos_np) < 1.0 and math.fabs(tgt_h - curr_pos.z_val) < 1.0:
                    teleported = True
                    break
            if not teleported:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Failed to teleport after 10 attempts!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.client.moveToPosition(velocity=1.0, x=0.0, y=0.0, z=-10.0)
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")


    def teleport_3d(self, position, rpy, fast=False):
        try:
            pos_as = position
            tele_rot = self.client.toQuaternion(rpy[0], rpy[1], rpy[2])
            self.send_local_velocity_command([0, 0, 0])
            self.rate.sleep_n_intervals(1)
            tele_pos = Vector3r()
            tele_pos.x_val = pos_as[0]
            tele_pos.y_val = pos_as[1]
            tele_pos.z_val = pos_as[2]

            self.client.simSetPose(tele_pos, tele_rot)
            self.send_local_velocity_command([0, 0, 0])
            if not fast:
                time.sleep(0.2)
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")


    def get_state(self, depth=False, segmentation=False):
        try:
            # Get images
            request = [ImageRequest(0, AirSimImageType.Scene, False, False)]
            if depth:
                request.append(ImageRequest(0, AirSimImageType.DepthPlanner, True))
            if segmentation:
                request.append(ImageRequest(0, AirSimImageType.Segmentation, False, False))

            response = self.client.simGetImages(request)

            pos_dict = response[0].camera_position
            rot_dict = response[0].camera_orientation

            cam_pos = [pos_dict[b"x_val"], pos_dict[b"y_val"], pos_dict[b"z_val"] + self.height_offset]
            cam_rot = [rot_dict[b"w_val"], rot_dict[b"x_val"], rot_dict[b"y_val"], rot_dict[b"z_val"]]

            img_data = np.frombuffer(response[0].image_data_uint8, np.uint8)
            image_raw = img_data.reshape(response[0].height, response[0].width, 4)
            image_raw = image_raw[:, :, 0:3]

            concat = [image_raw]

            if depth:
                # Depth is cast to uint8, because otherwise it takes up a lot of space and we don't need such fine-grained
                # resolution
                depth_data = np.asarray(response[1].image_data_float)
                depth_raw = depth_data.reshape(response[1].height, response[1].width, 1)
                depth_raw = np.clip(depth_raw, 0, 254 / DEPTH_SCALE)
                depth_raw *= DEPTH_SCALE
                depth_raw = depth_raw.astype(np.uint8)
                concat.append(depth_raw)

            if segmentation:
                seg_data = np.frombuffer(response[2].image_data_uint8, np.uint8)
                seg_raw = seg_data.reshape(response[2].height, response[2].width, 4)
                concat.append(seg_raw)

            img_state = np.concatenate(concat, axis=2)

            # cam_pos, cam_rot are in ned coordinate frame in the simulator. They're left as-is
            # drones position gets converted to config coordinates instead

            # Get drone's physical location, orientation and velocity
            pos = self.client.getPosition()
            vel = self.client.getVelocity()
            o = self.client.getOrientation()

            # Convert to numpys
            drn_pos = np.asarray([pos.x_val, pos.y_val, pos.z_val + self.height_offset]) # xyz
            drn_vel = np.asarray([vel.x_val, vel.y_val, vel.z_val]) # xyz
            #drn_rot = np.asarray([o.w_val, o.x_val, o.y_val, o.z_val]) #wxyz
            drn_rot = np.asarray(euler.quat2euler([o.w_val, o.x_val, o.y_val, o.z_val]))

            # Turn it into a nice 9-D vector with euler angles
            """
            roll, pitch, yaw = self.units.euler_from_as(r_as, p_as, y_as)
            pos_3d_as = np.asarray([pos.x_val, pos.y_val, pos.z_val])
            pos_state = self.units.pos3d_from_as(pos_3d_as)
            vel_state = self.units.vel3d_from_as(np.asarray([vel.x_val, vel.y_val, vel.z_val]))
            ori_state = [roll, pitch, yaw]
            """

            drone_state = np.concatenate((drn_pos, drn_rot, drn_vel, cam_pos, cam_rot), axis=0)

            return drone_state, img_state
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")

    def set_current_env_from_config(self, config, instance_id=None):
        """
        Store the provided dict representation of a random_config.json into the
        correct folder for instance_id simulator instance to read it when resetting it's pomdp.
        :param config: Random pomdp configuration to store
        :param instance_id: Simulator instance for which to store it.
        """
        try:
            self._set_config(config, "", "random_config", instance_id=instance_id)
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")

    def set_current_env_id(self, env_id, instance_id=None):
        try:
            self._load_drone_config(env_id, instance_id)
            gc.collect()
        except msgpackrpc.error.TimeoutError as e:
            startAirSim(self, self.instance, self.port)
            raise RolloutException("Simulator Died")

    def set_current_seg_idx(self, seg_idx, instance_id=None):
        pass

    def rollout_begin(self, instruction):
        pass

    def rollout_end(self):
        pass

    def reset_environment(self):
        self._start_call()
        self.client.simResetEnv()
        self._start_call()

    def land(self):
        pass

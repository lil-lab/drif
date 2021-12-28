#!/home/drone/catkin_ws_py3/venv/bin/python
import sys
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool, Header
import geometry_msgs
from geometry_msgs.msg import Pose,PoseStamped, Twist

import tf
import numpy as np
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import cv2
import copy
from data_io.instructions import get_restricted_env_id_lists, get_env_id_lists_perception
from parameters.parameter_server import initialize_experiment, get_current_parameters
from data_io.models import load_model
#from drones.airsim_interface.airsimClientNew import *

from PIL import Image as PIL_Image

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor

from utils.simple_profiler import SimpleProfiler
from data_io.env import pose_ros_enu_to_airsim_ned

import parameters.parameter_server as P


dataloader_rollout = None
model = None
PROFILE = True

class MappingNoRollout():
    def __init__(self, env_id=0, scale=1.0, instance=0):
        rospy.init_node('drone_controller')
        self.instance = instance
        self.env_id = env_id
        self.prof = SimpleProfiler(print=PROFILE)
        self.listener = tf.TransformListener()
        self.count = 0

    def compute_action(self, img):
        self.prof.tick("out")
        #print("enter compute actions")
        (trans, rot) = self.listener.lookupTransform('/map_ned', '/camera_airsim_base_link', img.header.stamp)
        #print(trans, rot)
        position = torch.tensor(trans)
        orientation= torch.tensor([rot[3]] + rot[:3])
        state = torch.cat((torch.zeros(9), position.float(), orientation.float()))
        self.prof.tick("pose")

        bridge = CvBridge()
        image_pil = PIL_Image.fromarray(bridge.imgmsg_to_cv2(img, "rgb8")).resize((256, 144))
        image_tensor = to_tensor(image_pil)
        self.prof.tick("image to tensor")

        #if count>=0:
        self.rollout_batch["images"] = torch.stack([image_tensor.squeeze()]).unsqueeze(0)
        self.rollout_batch["states"] = torch.stack([state]).unsqueeze(0)
        #else:
        #    rollout_batch["images"] = torch.cat((rollout_batch["images"],
        #                        torch.stack([image_tensor.squeeze()]).unsqueeze(0)), 1)
        #    rollout_batch["states"] = torch.cat((rollout_batch["states"],
        #                        torch.stack([state]).unsqueeze(0)), 1)

        #batch_mix["md"] = [[rollout_batch["md"][0][0]]]
        self.rollout_batch["md"] = [[self.md[0][0]]] #[[rollout_batch["md"][0][:count+1]]]
        #print(self.rollout_batch["images"].shape)
        #print(self.rollout_batch["states"].shape)
        self.prof.tick("create batch")
        #print("start batch")
        loss = model.sup_loss_on_batch(self.rollout_batch, eval=True, return_actions=True)
        #print("action {}".format(loss))
        self.prof.tick("compute maps + loss")
        #print("action computed")
        self.count +=1
        self.prof.print_stats(1)


    def run_drone_controller(self):
        rospy.init_node('drone_controller')
        global listener
        listener = tf.TransformListener()
        global position_target_pub, velocity_target_pub
        velocity_target_pub = rospy.Publisher("/drone_controller/cmd_vel", Twist, queue_size=1)

        # Initialize environment for the model
        initialize_experiment("test_forward_cage")

        setup = get_current_parameters()["Setup"]
        #supervised_params = get_current_parameters()["Supervised"]
        global model
        model, model_loaded = load_model()
        train_envs_rollout, dev_envs_rollout, test_envs_rollout = get_restricted_env_id_lists(max_envs=setup["max_envs"])

        dataset_name = P.get_current_parameters().get("Data").get("dataset_name")
        rollout_dataset = model.get_dataset(data=None, envs=train_envs_rollout[0:1], dataset_prefix=dataset_name, dataset_prefix="supervised", eval=True)

        self.dataloader_rollout = DataLoader(
            rollout_dataset,
            collate_fn=rollout_dataset.collate_fn,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # self.num_loaders
            pin_memory=False,
            timeout=0,
            drop_last=False)
        rate = rospy.Rate(1)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.compute_action)

        while not(rospy.is_shutdown()):
            iter_rollout = iter(self.dataloader_rollout)
            self.rollout_batch = iter_rollout.next()
            self.md = copy.copy(self.rollout_batch["md"])
            pass

if __name__=="__main__":
    try:
        MappingNoRollout(0).run_drone_controller()
    except rospy.ROSInterruptException:
        pass

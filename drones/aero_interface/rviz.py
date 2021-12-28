import rospy
import numpy as np
import drones.aero_interface.ros_node as rn
from drones.aero_interface.landmark_colors import color_names, colors

import random
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point32
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker

class RvizInterface():

    def __init__(self,
                 base_name="/visualizer/",
                 pose_topics=None,
                 posearray_topics=None,
                 path_topics=None,
                 markerarray_topics=None,
                 map_topics=None,
                 voxel_topics=None):

        rn.init_node_if_necessary()
        self.base_name = base_name
        if pose_topics:
            for topic in pose_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), PoseStamped, queue_size=1))

        if posearray_topics:
            for topic in posearray_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), PoseArray, queue_size=1))
                self._set_pub(topic+"_path", rospy.Publisher(self._topic(topic+"_path"), Path, queue_size=1))

        if path_topics:
            for topic in path_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), Path, queue_size=1))

        if markerarray_topics:
            for topic in markerarray_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), MarkerArray, queue_size=1))

        if map_topics:
            for topic in map_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), PointCloud, queue_size=1))

        if voxel_topics:
            for topic in voxel_topics:
                self._set_pub(topic, rospy.Publisher(self._topic(topic), PointCloud, queue_size=1))

        self.pose_arrays = {}
        self.paths = {}

    def _topic(self, key):
        return self.base_name + key

    def _publisher_attr(self, topic):
        return topic + "_publisher"

    def _get_pub(self, topic):
        return self.__getattribute__(self._publisher_attr(topic))

    def _set_pub(self, topic, pub):
        self.__setattr__(self._publisher_attr(topic), pub)

    def _make_pose_msg(self, pos_vec, rot_quat, wxyz):
        pose = Pose()
        pose.position.x = pos_vec[0]
        pose.position.y = pos_vec[1]
        pose.position.z = pos_vec[2]
        if rot_quat is not None:
            if wxyz:
                pose.orientation.w = rot_quat[0]
                pose.orientation.x = rot_quat[1]
                pose.orientation.y = rot_quat[2]
                pose.orientation.z = rot_quat[3]
            else:
                pose.orientation.x = rot_quat[0]
                pose.orientation.y = rot_quat[1]
                pose.orientation.z = rot_quat[2]
                pose.orientation.w = rot_quat[3]
        return pose

    def publish_map(self, topic, map_data_np, map_size_m, frame="/map_ned"):
        msg = PointCloud()
        msg.header.frame_id = frame
        msg.header.stamp = rospy.get_rostime()

        # Shift and scale to 0-1 range
        map_data_np = map_data_np[:,:,:3]
        map_data_np = map_data_np - np.min(map_data_np)
        map_data_np /= (np.max(map_data_np) + 1e-9)

        c = np.linspace(0, map_size_m, map_data_np.shape[1])
        cgrid = np.asarray(np.meshgrid(c, c))

        r = ChannelFloat32()
        r.name = "r"
        g = ChannelFloat32()
        g.name = "g"
        b = ChannelFloat32()
        b.name = "b"
        msg.channels.append(r)
        msg.channels.append(g)
        msg.channels.append(b)

        for x in range(map_data_np.shape[0]):
            for y in range(map_data_np.shape[1]):
                p = Point32()
                p.x = cgrid[0,x,y]
                p.y = cgrid[1,x,y]
                p.z = 0.05
                r.values.append(float(map_data_np[x,y,0]))
                g.values.append(float(map_data_np[x,y,1]))
                if map_data_np.shape[2] > 2:
                    b.values.append(float(map_data_np[x,y,2]))
                else:
                    b.values.append(0)
                msg.points.append(p)

        self._get_pub(topic).publish(msg)

    def create_path_from_2d_array(self, topic, array, height=0, frame="/map_ned", publish=True):
        self.clear_path(topic)
        for pt in array:
            pos = [pt[0], pt[1], height]
            self.add_point_to_path(topic, pos, frame=frame)
        if publish:
            self.publish_path(topic)

    def add_point_to_path(self, topic, pos_vec, rot_quat=None, frame="/map_ned", wxyz=False):
        if topic not in self.paths:
            path = Path()
            path.header.frame_id = frame
            self.paths[topic] = path
        else:
            path = self.paths[topic]
        path.header.stamp = rospy.get_rostime()
        pose = self._make_pose_msg(pos_vec, rot_quat, wxyz=wxyz)
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = rospy.get_rostime()
        ps.pose = pose
        path.poses.append(ps)

    def clear_path(self, topic):
        if topic in self.paths:
            del self.paths[topic]

    def publish_instruction_text(self, topic, text):
        delarray = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        delarray.markers.append(delete)
        self._get_pub(topic).publish(delarray)

        markerarray = MarkerArray()

        t = Marker()
        t.ns = "instruction_text"
        t.id = 0
        t.action = Marker.ADD
        t.header.frame_id = "/map_ned"
        t.header.stamp = rospy.get_rostime()
        t.type = Marker.TEXT_VIEW_FACING
        t.text = text
        t.pose.position.x = 5.0
        t.pose.position.y = 2.35
        t.pose.position.z = -0.4
        t.pose.orientation.w = 1
        t.scale.x = 0.2
        t.scale.y = 0.2
        t.scale.z = 0.2
        t.color.a = 1.0
        t.color.r = 1.0
        t.color.g = 1.0
        t.color.b = 1.0

        markerarray.markers.append(t)
        self._get_pub(topic).publish(markerarray)

    def publish_env_config(self, topic, env_config):
        delarray = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        delarray.markers.append(delete)
        self._get_pub(topic).publish(delarray)

        markerarray = MarkerArray()

        bg = Marker()
        bg.ns = "env_config_bg"
        bg.id = 0
        bg.action = Marker.ADD
        bg.header.frame_id = "/map_ned"
        bg.header.stamp = rospy.get_rostime()
        bg.type = Marker.CUBE
        bg.pose.position.x = 2.35
        bg.pose.position.y = 2.35
        bg.pose.position.z = 0.01
        bg.pose.orientation.w = 1.0
        bg.scale.x = 4.7
        bg.scale.y = 4.7
        bg.scale.z = 0.02
        bg.color.a = 1.0
        bg.color.r = 0.2
        bg.color.g = 0.6
        bg.color.b = 0.3
        markerarray.markers.append(bg)

        for i, lmname in enumerate(env_config["landmarkName"]):
            x, y = env_config["x_pos_as"][i], env_config["y_pos_as"][i]
            m = Marker()
            m.ns = "env_config_balls"
            m.id = i
            m.action = Marker.ADD
            m.header.frame_id = "/map_ned"
            m.header.stamp = rospy.get_rostime()
            m.type = Marker.SPHERE
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.orientation.w = 1
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 1.0
            m.color.r = float(colors[i][0]) / 255
            m.color.g = float(colors[i][1]) / 255
            m.color.b = float(colors[i][2]) / 255

            t = Marker()
            t.ns = "env_config_text"
            t.id = i
            t.action = Marker.ADD
            t.header.frame_id = "/map_ned"
            t.header.stamp = rospy.get_rostime()
            t.type = Marker.TEXT_VIEW_FACING
            t.text = lmname
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = -0.4
            t.pose.orientation.w = 1
            t.scale.x = 0.4
            t.scale.y = 0.4
            t.scale.z = 0.4
            t.color.a = 1.0
            t.color.r = float(colors[i][0]) / 255
            t.color.g = float(colors[i][1]) / 255
            t.color.b = float(colors[i][2]) / 255

            markerarray.markers.append(m)
            markerarray.markers.append(t)
        self._get_pub(topic).publish(markerarray)

    def tensor_2d_to_pointcloud(self, ndarray, axis0_size_m, frame="/map_ned"):
        msg = PointCloud()
        msg.header.frame_id = frame
        msg.header.stamp = rospy.get_rostime()

        # Shift and scale to 0-1 range
        map_data_np = ndarray[:, :, :3]
        map_data_np = map_data_np - np.min(map_data_np)
        map_data_np /= (np.max(map_data_np) + 1e-9)

        c = np.linspace(0, axis0_size_m, map_data_np.shape[1])
        cgrid = np.asarray(np.meshgrid(c, c))

        r = ChannelFloat32()
        r.name = "r"
        g = ChannelFloat32()
        g.name = "g"
        b = ChannelFloat32()
        b.name = "b"
        msg.channels.append(r)
        msg.channels.append(g)
        msg.channels.append(b)

        for x in range(map_data_np.shape[0]):
            for y in range(map_data_np.shape[1]):
                p = Point32()
                p.x = cgrid[0, x, y]
                p.y = cgrid[1, x, y]
                p.z = 0.05
                r.values.append(float(map_data_np[x, y, 0]))
                g.values.append(float(map_data_np[x, y, 1]))
                b.values.append(float(map_data_np[x, y, 2]))
                msg.points.append(p)
        return msg

    def tensor_3d_to_pointcloud(self, ndarray, axis0_size_m, frame="/map_ned"):
        msg = PointCloud()
        msg.header.frame_id = frame
        msg.header.stamp = rospy.get_rostime()

        # Shift and scale to 0-1 range
        map_data_np = ndarray[:, :, :, :3]
        map_data_np = map_data_np - np.min(map_data_np)
        map_data_np /= (np.max(map_data_np) + 1e-9)

        c = np.linspace(0, axis0_size_m, map_data_np.shape[1])
        cgrid = np.asarray(np.meshgrid(c, c, c))

        r = ChannelFloat32()
        r.name = "r"
        g = ChannelFloat32()
        g.name = "g"
        b = ChannelFloat32()
        b.name = "b"
        msg.channels.append(r)
        msg.channels.append(g)
        msg.channels.append(b)

        for x in range(map_data_np.shape[0]):
            for y in range(map_data_np.shape[1]):
                for z in range(map_data_np.shape[2]):
                    col = map_data_np[x, y, z, :]
                    # TODO: Clear up this hack
                    if np.linalg.norm(col) < 0.1:
                        continue

                    # Add multiple overlapping points to increase alpha value
                    intensity = int((np.linalg.norm(col)) * 10)
                    for i in range(intensity):
                        p = Point32()
                        p.x = cgrid[0, x, y, z]
                        p.y = cgrid[1, x, y, z]
                        p.z = cgrid[2, x, y, z]
                        r.values.append(min(float(col[0]) * 10, 1.0))
                        g.values.append(min(float(col[1]) * 10, 1.0))
                        b.values.append(min(float(col[2]) * 10, 1.0))
                        msg.points.append(p)
        print(f"Generated pointcloud with {len(msg.points)} points")
        return msg

    def publish_tensor(self, topic, ndarray, axis0_size_m, frame="/map_ned"):
        if len(ndarray.shape) == 3:
            pc = self.tensor_2d_to_pointcloud(ndarray, axis0_size_m, frame=frame)
        elif len(ndarray.shape) == 4:
            pc = self.tensor_3d_to_pointcloud(ndarray, axis0_size_m, frame=frame)

        self._get_pub(topic).publish(pc)

    def publish_path(self, topic):
        if topic in self.paths:
            path = self.paths[topic]
            self._get_pub(topic).publish(path)

    def publish_pose(self, topic, pos_vec, rot_quat, frame="/map_ned", wxyz=False):
        """
        :param topic: ROS topic to publish to.
        :param pos_vec: 3-dimensional indexable position
        :param rot_quat: wxyz quaternion, indexable
        :param frame: frame_id to publish to. Default: /map_ned
        :return:
        """
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = rospy.get_rostime()
        ps.pose = self._make_pose_msg(pos_vec, rot_quat, wxyz=wxyz)
        self._get_pub(topic).publish(ps)


    def add_pose_and_publish_array(self, topic, pos_vec, rot_quat, frame="/map_ned", wxyz=False):
        """
        :param topic: ROS topic to publish to.
        :param pos_vec: 3-dimensional indexable position
        :param rot_quat: wxyz quaternion, indexable
        :param frame: frame_id to publish to. Default: /map_ned
        :return:
        """
        pose = self._make_pose_msg(pos_vec, rot_quat, wxyz=wxyz)

        if topic not in self.pose_arrays:
            pa = PoseArray()
            pa.header.frame_id = frame
            path = Path()
            path.header.frame_id = frame
            self.pose_arrays[topic] = pa
            self.paths[topic] = path
        else:
            pa = self.pose_arrays[topic]
            path = self.paths[topic]

        pa.header.stamp = rospy.get_rostime()
        path.header.stamp = pa.header.stamp

        pa.poses.append(pose)

        ps = PoseStamped()
        ps.header = pa.header
        ps.pose = pose
        path.poses.append(ps)

        self._get_pub(topic).publish(pa)
        self._get_pub(topic+"_path").publish(path)

    def reset_pose_array(self, topic):
        if topic in self.pose_arrays:
            del self.pose_arrays[topic]
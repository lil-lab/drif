class RvizInterface():

    def __init__(self, base_name="/visualizer/", pose_topics=None, posearray_topics=None):
        pass

    def publish_pose(self, topic, pos_vec, rot_quat, frame="/map_ned", wxyz=False):
        pass

    def add_pose_and_publish_array(self, topic, pos_vec, rot_quat, frame="/map_ned", wxyz=False):
        pass

    def reset_pose_array(self, topic):
        pass
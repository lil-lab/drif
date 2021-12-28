import rospy

INITIALIZED = False

def init_node_if_necessary():
    global INITIALIZED
    if not INITIALIZED:
        rospy.init_node("py_drone_controller")
        print("Initialized ROS node")
        INITIALIZED = True
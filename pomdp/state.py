class DroneState:
    def __init__(self, image=None, state=None):
        self.image = image
        self.state = state

    def get_pos(self):
        return self.state[0:2]

    def get_pos_3d(self):
        return self.state[0:3]

    def get_cam_pos_3d(self):
        return self.state[9:12]

    def get_cam_rot(self):
        return self.state[12:16]

    def get_rot_euler(self):
        return self.state[3:6]

    def get_depth_image(self):
        return self.image[:, :, 3]

    def get_rgb_image(self):
        return self.image[:, :, 0:3]
class AbstractReward:
    def __init__(self, path):
        self.path = path

    def set_current_segment(self, start_idx, end_idx):
        pass

    def get_reward(self, state, action):
        pass

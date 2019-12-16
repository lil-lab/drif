class DummySummaryWriter():

    def __init__(self, log_dir="runs", restore=False, spy=False):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None):
        pass

    def add_dict(self, prefix, dict, global_step=None):
        pass
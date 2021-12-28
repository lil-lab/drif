import torch
import torch.nn as nn
import numpy as np
from transforms3d.euler import euler2quat

from learning.inputs.pose import Pose
from learning.inputs.vision import standardize_images, standardize_image
from learning.modules.generic_model_state import GenericModelState

from data_io.model_io import load_model_state_dict
from data_io.paths import get_logging_dir
from learning.inputs.pose import Pose
from utils.logging_summary_writer import LoggingSummaryWriter
from utils.dummy_summary_writer import DummySummaryWriter


class NavigationModelComponentBase(nn.Module):

    def __init__(self, run_name="", domain="sim", model_name="", nowriter=False):
        super(NavigationModelComponentBase, self).__init__()
        self.run_name = run_name
        self.domain = domain
        self.model_name = model_name
        self.nowriter = nowriter
        self.writer = self._make_writer()

        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.loaded_pretrained_modules = False
        self.loaded_pref_pretrained_modules = False

        self.model_state = None

    def set_model_state(self, model_state):
        self.model_state = model_state

    def _make_writer(self):
        if self.nowriter:
            writer = DummySummaryWriter()
        else:
            writer = LoggingSummaryWriter.make_singleton(log_dir=f"{get_logging_dir()}/runs/{self.run_name}-{self.domain}-{self.model_name}")
        return writer

    def make_picklable(self):
        """
        The SummaryWriter from TensorboardX has thread locks, which make the model non-picklable, which means it
        can't be sent between processes. This function replaces it with a dummy mock writer that doesn't do anything,
        but allows the model to be sent between processes
        :return:
        """
        self.writer = DummySummaryWriter()

    def enable_logging(self):
        """
        This function can re-enable logging after a call to make_picklable() has removed the summarywriter.
        :return:
        """
        self.writer = self._make_writer()

    def steal_cross_domain_modules(self, other_self):
        self.iter = other_self.iter

    def both_domain_parameters(self, other_self):
        # This function iterates and yields parameters from this module and the other module, but does not yield
        # shared parameters twice.
        # First yield all of the other module's parameters
        for p in other_self.parameters():
            yield p
        return

    def get_iter(self):
        iter = int(self.iter.data[0])
        return iter

    def inc_iter(self):
        self.iter += 1

    def load_state_dict(self, state_dict, strict=True):
        super(NavigationModelComponentBase, self).load_state_dict(state_dict, strict)
        # Load pre-trained models - invalidate, because we may have overwritten them
        self.loaded_pref_pretrained_modules = False
        self.loaded_pretrained_modules = False

        if not self.loaded_pref_pretrained_modules:
            self.load_pref_pretrained_modules()
            self.loaded_pref_pretrained_modules = True
        if not self.loaded_pretrained_modules:
            self.load_pretrained_modules()
            self.loaded_pretrained_modules = True

    def load_from_path(self, path):
        print("LOADING SELF FROM PATH: ", path)
        device = next(self.parameters()).device
        state_dict = load_model_state_dict(path)
        self.load_state_dict(state_dict)
        print("MOVING SELF TO DEVICE: ", device)
        self.to(device)

    def load_pref_pretrained_modules(self):
        ...

    def load_pretrained_modules(self):
        ...

    def init_weights(self):
        pass

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def drone_poses_from_states(self, states):
        drn_pos = states[:, 0:3]
        drn_rot_euler = states[:, 3:6].detach().cpu().numpy()
        quats = [euler2quat(a[0], a[1], a[2]) for a in drn_rot_euler]
        quats = torch.from_numpy(np.asarray(quats)).to(drn_pos.device)
        pose = Pose(drn_pos, quats)
        return pose

    def poses_from_states(self, states):
        USE_DRONE_POSES = True
        if USE_DRONE_POSES:
            return self.drone_poses_from_states(states)
        else:
            return self.cam_poses_from_states(states)

    def states_to_torch(self, state):
        states_np = state.state[np.newaxis, :]
        images_np = state.image[np.newaxis, :]
        images_np = standardize_images(images_np, out_np=True)
        images_fpv = torch.from_numpy(images_np).float()
        states = torch.from_numpy(states_np)
        return states, images_fpv

    def preprocess_batch(self, batch):
        """
        Called within dataset - opportunity for model to pre-compute certain CPU-heavy computation results and
        add them to the batch within DataLoader to make use of multiple processes.
        :param batch: dictonary of data
        :return: updated batch with additional data
        """
        return batch

    def forward(self, *args, **kwargs):
        ...

    def to_model_device(self, tensor):
        device = next(self.parameters()).device
        return tensor.to(device)

    def get_model_device(self):
        return next(self.parameters()).device

    def unbatch(self, batch):
        ...

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval):
        ...

    def get_dataset(self, *args, **kwargs):
        ...

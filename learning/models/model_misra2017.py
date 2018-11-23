import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from learning.datasets.segment_dataset_simple import SegmentDataset
import learning.datasets.aux_data_providers as aup

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.inputs.pose import Pose
from learning.inputs.sequence import none_padded_seq_to_tensor, len_until_nones
from learning.inputs.vision import standardize_image
from learning.modules.module_with_auxiliaries_base import ModuleWithAuxiliaries
from learning.modules.action_loss import ActionLoss

from learning.modules.dipandrew.image_cnn_emnlp import ImageCnnEmnlp
from learning.modules.dipandrew.text_simple_module import TextSimpleModule
from learning.modules.dipandrew.action_simple_module import ActionSimpleModule
from learning.modules.dipandrew.incremental_multimodal_emnlp import IncrementalMultimodalEmnlp

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter

from learning.meters.meter_server import get_current_meters

from parameters.parameter_server import get_current_parameters

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class ModelMisra2017(ModuleWithAuxiliaries):

    def __init__(self, run_name=""):

        super(ModelMisra2017, self).__init__()
        self.model_name = "misra2017"
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir="runs/" + run_name)

        self.params = get_current_parameters()["Model"]
        self.trajectory_len = get_current_parameters()["Setup"]["trajectory_length"]

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        # CNN over images - using what is essentially SimpleImage currently
        self.image_module = ImageCnnEmnlp(
            image_emb_size=self.params["image_emb_dim"],
            input_num_channels=3*5, # 3 channels per image - 5 images in history
            image_height=self.params["img_h"],
            image_width=self.params["img_w"])

        # LSTM to embed text
        self.text_module = TextSimpleModule(
            emb_dim=self.params["word_emb_dim"],
            hidden_dim=self.params["emb_size"],
            vocab_size=self.params["vocab_size"])

        # Action module to embed previous action+block
        self.action_module = ActionSimpleModule(
            num_actions=self.params["num_actions"],
            action_emb_size=self.params["action_emb_dim"]
        )

        # Put it all together
        self.final_module = IncrementalMultimodalEmnlp(
            image_module=self.image_module,
            text_module=self.text_module,
            action_module=self.action_module,
            input_embedding_size=self.params["lstm_emb_dim"] + self.params["image_emb_dim"] + self.params["action_emb_dim"],
            output_hidden_size=self.params["h1_hidden_dim"],
            blocks_hidden_size=self.params["blocks_hidden_dim"],
            directions_hidden_size=self.params["action_hidden_dim"],
            max_episode_length = self.trajectory_len
        )

        self.action_loss = ActionLoss()

        self.env_id = None
        self.prev_instruction = None
        self.seq_step = 0
        self.model_state = None
        self.image_emb_seq = None
        self.state_feature = None

    # TODO: Try to hide these in a superclass or something. They take up a lot of space:
    def cuda(self, device=None):
        ModuleWithAuxiliaries.cuda(self, device)
        self.image_module.cuda(device)
        self.text_module.cuda(device)
        self.final_module.cuda(device)
        self.action_module.cuda(device)
        self.action_loss.cuda(device)
        return self

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.final_module.init_weights()

    def reset(self):
        # TODO: This is error prone. Create a class StatefulModule, iterate submodules and reset all stateful modules
        super(ModelMisra2017, self).reset()
        self.seq_step = 0
        self.model_state = None
        pass

    def setEnvContext(self, context):
        print("Set env context to: " + str(context))
        self.env_id = context["env_id"]

    def get_action(self, state, instruction):
        """
        Given a DroneState (from PomdpInterface) and instruction, produce a numpy 4D action (x, y, theta, pstop)
        :param state: DroneState object with the raw image from the simulator
        :param instruction: Tokenized instruction given the corpus
        #TODO: Absorb corpus within model
        :return:
        """
        # TODO: Simplify this
        self.eval()
        images_np_pure = state.image
        state_np = state.state

        #print("Act: " + debug_untokenize_instruction(instruction))

        images_np = standardize_image(images_np_pure)
        image_fpv = Variable(none_padded_seq_to_tensor([images_np]))
        state = Variable(none_padded_seq_to_tensor([state_np]))
        # Add the batch dimension

        first_step = True
        if instruction == self.prev_instruction:
            first_step = False
        self.prev_instruction = instruction

        img_in_t = image_fpv
        img_in_t.volatile = True

        instr_len = [len(instruction)] if instruction is not None else None
        instruction = torch.LongTensor(instruction).unsqueeze(0)
        instruction = cuda_var(instruction, self.is_cuda, self.cuda_device)

        state.volatile = True

        if self.is_cuda:
            img_in_t = img_in_t.cuda(self.cuda_device)
            state = state.cuda(self.cuda_device)

        self.seq_step += 1

        action = self(img_in_t, instruction, instr_len)

        output_action = action.squeeze().data.cpu().numpy()
        stop_prob = output_action[3]
        output_stop = 1 if stop_prob > 0.5 else 0
        output_action[3] = output_stop

        #print("action: ", output_action)

        return output_action

    def deterministic_action(self, action_mean, action_std, stop_prob):
        batch_size = action_mean.size(0)
        action = Variable(empty_float_tensor((batch_size, 4), self.is_cuda, self.cuda_device))
        action[:, 0:3] = action_mean[:, 0:3]
        action[:, 3] = stop_prob
        return action

    def sample_action(self, action_mean, action_std, stop_prob):
        action = torch.normal(action_mean, action_std)
        stop = torch.bernoulli(stop_prob)
        return action, stop

    # This is called before beginning an execution sequence
    def start_sequence(self):
        self.seq_step = 0
        self.reset()
        print("RESETTED!")
        return

    # TODO: Move this somewhere and standardize
    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def instructions_to_dipandrew(self, instructions, instr_lengths):
        out = []
        for i in range(len(instructions)):
            instr_i = instructions[i:i+1, 0:instr_lengths[i]]
            out.append(instr_i)
        return out

    def forward(self, images, instructions, instr_lengths):

        seq_len = len(images)

        instr_dipandrew = self.instructions_to_dipandrew(instructions, instr_lengths)

        # Add sequence dimension, since we're treating batches as sequences
        images = images.unsqueeze(0)

        all_actions = []
        for i in range(seq_len):
            time_in = np.asarray([self.seq_step])
            time_in = Variable(self.maybe_cuda(torch.from_numpy(time_in).long()))
            action_i, self.model_state = self.final_module(
                images[0:1, i:i+1], instr_dipandrew[i], time_in, self.model_state)

            self.seq_step += 1
            all_actions.append(action_i)

        actions = torch.cat(all_actions, dim=0)
        return actions

    def maybe_cuda(self, tensor):
        if self.is_cuda:
            return tensor.cuda()
        else:
            return tensor

    def cuda_var(self, tensor):
        return cuda_var(tensor, self.is_cuda, self.cuda_device)

    # Forward pass for training (with batch optimizations
    def sup_loss_on_batch(self, batch, eval):
        self.prof.tick("out")

        action_loss_total = Variable(empty_float_tensor([1], self.is_cuda, self.cuda_device))

        if batch is None:
            print("Skipping None Batch")
            return action_loss_total

        images = self.maybe_cuda(batch["images"])
        instructions = self.maybe_cuda(batch["instr"])
        instr_lengths = batch["instr_len"]
        actions = self.maybe_cuda(batch["actions"])

        metadata = batch["md"]

        batch_size = images.size(0)
        count = 0

        # Loop thru batch
        for b in range(batch_size):
            self.reset()
            self.prof.tick("out")
            b_seq_len = len_until_nones(metadata[b])

            # TODO: Generalize this
            # Slice the data according to the sequence length
            b_metadata = metadata[b][:b_seq_len]
            b_images = images[b][:b_seq_len]
            b_instructions = instructions[b][:b_seq_len]
            b_instr_len = instr_lengths[b][:b_seq_len]
            b_actions = actions[b][:b_seq_len]

            # ----------------------------------------------------------------------------

            self.prof.tick("inputs")

            actions = self(b_images, b_instructions, b_instr_len)

            action_losses, _ = self.action_loss(b_actions, actions, batchreduce=False)

            self.prof.tick("call")
            action_losses = self.action_loss.batch_reduce_loss(action_losses)
            action_loss = self.action_loss.reduce_loss(action_losses)
            action_loss_total = action_loss
            count += b_seq_len

            self.prof.tick("loss")

        action_loss_avg = action_loss_total / (count + 1e-9)

        self.prof.tick("out")

        prefix = self.model_name + ("/eval" if eval else "/train")

        self.writer.add_dict(prefix, get_current_meters(), self.get_iter())
        self.writer.add_scalar(prefix + "/action_loss", action_loss_avg.data.cpu()[0], self.get_iter())

        total_loss = action_loss_avg

        self.inc_iter()

        self.prof.loop()
        self.prof.print_stats(1)

        return total_loss

    def get_dataset(self, data=None, envs=None, dataset_name=None, eval=False):
        # TODO: Maybe use eval here
        #if self.fpv:
        return SegmentDataset(data=data, env_list=envs, dataset_name=dataset_name, aux_provider_names=[], segment_level=True)
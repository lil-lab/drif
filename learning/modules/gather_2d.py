import torch
import torch.nn.functional as F
from torch.autograd import Variable

from visualization import Presenter

from learning.modules.cuda_module import CudaModule as ModuleBase

"""
Given a 4-D image (batch x channels x X x Y) and a 'list' of 2-D coordinates, extract the channel-vector for each
of the 2D coordinates and return as a 'list' of channels
"""
class Gather2D(ModuleBase):

    def __init__(self):
        super(Gather2D, self).__init__()
        self.is_cuda = False
        self.cuda_device = None

    def cuda(self, device=None):
        self.is_cuda = True
        self.cuda_device = device

    def init_weights(self):
        pass

    def dbg_viz(self, image, coords_in_features):
        image = image.data.cpu()
        image[0, :, :] = 0.0
        image -= torch.min(image)
        image /= (torch.max(image) + 1e-9)
        for coord in coords_in_features:
            c = coord.long()
            x = c.data[0]
            y = c.data[1]
            image[0, x, y] = 1.0
        Presenter().show_image(image, "gather_dbg", torch=True, scale=2, waitkey=True)

    def forward(self, image, coords_in_features, axes=(2, 3)):

        # Get rid of the batch dimension
        # TODO Handle batch dimension properly
        if len(coords_in_features.size()) > 2:
            coords_in_features = coords_in_features[0]
            #image = image[0]

        #assert type(coords_in_features.data) is torch.LongTensor or type(coords_in_features.data) is torch.cuda.LongTensor

        assert coords_in_features.data.type() == 'torch.LongTensor' or\
            coords_in_features.data.type() == 'torch.cuda.LongTensor'

        # Coords in features are represented as B x X x Y
        #assert coords_in_features.size(1) == len(axes)

        # TODO: Handle additional batch axis. Currently batch axis must be of dimension 1
        assert len(axes) == 2

        if False:
            self.dbg_viz(image[0], coords_in_features)


        # Gather the full feature maps for each of the 2 batches
        gather_x = coords_in_features[:, 0].contiguous().view([-1, 1, 1, 1])
        gather_y = coords_in_features[:, 1].contiguous().view([-1, 1, 1, 1])

        gather_img_x = gather_x.expand([-1, image.size(1), 1, image.size(3)])
        gather_img_y = gather_y.expand([-1, image.size(1), 1, 1])

        # Make enough
        img_size = list(image.size())
        img_size[0] = coords_in_features.size(0)
        image_in = image.expand(img_size)

        #image_in = image_in.squeeze()

        #print(image_in.shape)
        #print(gather_img_x.shape)
        #print(gather_img_x)
        #print(gather_img_y)
        vec_y = torch.gather(image_in, 2, gather_img_x)
        vec = torch.gather(vec_y, 3, gather_img_y)
        vec = torch.squeeze(vec, 3)
        vec = torch.squeeze(vec, 2)

        return vec
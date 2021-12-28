import torch.nn as nn
import torch.nn.functional as F


class ImageCnnEmnlp(nn.Module):
    """
    pytorch module for image part of model
    """

    def __init__(self, image_emb_size, input_num_channels,
                 image_height, image_width):
        super(ImageCnnEmnlp, self).__init__()
        self.input_dims = (input_num_channels, image_height, image_width)

        ### FROM ORIGINAL CODDE
        #        self.output_size = output_size
        #        height = image_dim
        #        width = image_dim
        #        channels = 3 * 5

        # conv + affine + relu
        # conv + affine + relu
        # conv + affine + relu

        # affine transform

        # Create 4 images for padding

        ### END FROM ORIGINAL CODDE

        k1, s1 = 8, 4
        self.conv1 = nn.Conv2d(input_num_channels, 32, kernel_size=k1, stride=s1)
        self.bn1 = nn.BatchNorm2d(32)
        h = round_up((image_height - k1 + 1) * 1.0 / s1)
        w = round_up((image_width - k1 + 1) * 1.0 / s1)

        k2, s2 = 4, 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=k2, stride=s2)
        self.bn2 = nn.BatchNorm2d(32)
        h = round_up((h - k2 + 1) * 1.0 / s2)
        w = round_up((w - k2 + 1) * 1.0 / s2)

        k3, s3 = 4, 2
        self.conv3 = nn.Conv2d(32, 32, kernel_size=k3, stride=s3)
        self.bn3 = nn.BatchNorm2d(32)
        h = round_up((h - k3 + 1) * 1.0 / s3)
        w = round_up((w - k3 + 1) * 1.0 / s3)

        k4, s4 = 2, 1
        self.conv4 = nn.Conv2d(32, 32, kernel_size=k4, stride=s4)
        self.bn4 = nn.BatchNorm2d(32)
        h = round_up((h - k4 + 1) * 1.0 / s4)
        w = round_up((w - k4 + 1) * 1.0 / s4)

        self.head = nn.Linear(h * w * 32, image_emb_size)

    def forward(self, x):
        num_batches = x.size()[0]
        input_size = (num_batches,) + self.input_dims
        x = x.view(input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(num_batches, -1))

# what the f is this?
def round_up(x):
    return int(x + 0.99999)

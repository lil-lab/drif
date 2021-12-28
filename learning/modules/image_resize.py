import torch
import numpy as np
import torch.nn.functional as F

DEFAULT_SPATIAL_SIZE = (32, 32)


class ImageResizer(torch.nn.Module):
    def __init__(self, target_spatial_size=DEFAULT_SPATIAL_SIZE):
        super(ImageResizer, self).__init__()
        self.target_spatial_size = target_spatial_size

    def normalize_image_batch(self, image_batch):
        batch_size = image_batch.shape[0]
        channels = image_batch.shape[1]
        h = image_batch.shape[2]
        w = image_batch.shape[3]
        means = image_batch.view([batch_size, channels * h * w]).mean(dim=1)
        vars = image_batch.view([batch_size, channels * h * w]).std(dim=1)
        image_batch_norm = (image_batch - means[:, np.newaxis, np.newaxis, np.newaxis]) / (
                vars[:, np.newaxis, np.newaxis, np.newaxis] + 1e-10)
        return image_batch_norm

    def resize_to_target_size(self, image_list):
        """
        Takes a list of images, each in a different spatial size, and resizes them to self.crop_size by self.crop_size.
        :param image_list: B-length list of images, each a 3xHixWi tensor.
        :return: Bx3xExE tensor, where E is self.crop_size
        """
        images_s = [F.interpolate(img.unsqueeze(0), size=self.target_spatial_size, mode="bilinear", align_corners=False)[0] for img in image_list]
        images_s = torch.stack(images_s, dim=0)
        return images_s

    def resize_to_target_size_and_normalize(self, image_list):
        """
        Takes a list of images, each in a different spatial size, and resizes them to self.crop_size by self.crop_size,
        and normalizes each image to zero mean and unit variance.
        :param image_list: B-length list of images, each a 3xHixWi tensor.
        :return: Bx3xExE tensor, where E is self.crop_size
        """
        image_stack = self.resize_to_target_size(image_list)
        image_stack_norm = self.normalize_image_batch(image_stack)
        return image_stack_norm

    def resize_back_from_target_size(self, image_tensor, size_list):
        """
        Takes a batch of images with the same spatial size, and resizes each to a different size according to size_list
        :param image_tensor: Bx3xHxW batch of images
        :param size_list: List of (Hi, Wi) spatial dimensions.
        :return: List of 3xHixWi images, where (Hi, Wi) is the i-th element in size_list
        """
        images_out = [F.interpolate(image.unsqueeze(0), size=out_size, mode="bilinear", align_corners=False)[0] for image, out_size in zip(image_tensor, size_list)]
        return images_out

    def forward(self, input):
        raise NotImplementedError("This module does not support forward")

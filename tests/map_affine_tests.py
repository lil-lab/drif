import numpy as np
import math
import torch
import math
from torch.autograd import Variable
from transforms3d import euler, quaternions
from utils.simple_profiler import SimpleProfiler

from learning.inputs.vision import standardize_image
from learning.inputs.pose import Pose
from learning.models.semantic_map.map_affine import MapAffine
from learning.modules.affine_2d import Affine2D
from data_io.env import load_env_img
from visualization import Presenter


def affine_2d_test():

    img = load_env_img(2, 128, 128)
    img = standardize_image(img)
    img = torch.from_numpy(img).float().unsqueeze(0)

    px = 64
    py = 64
    theta = 0.5

    c = math.cos(theta)
    s = math.sin(theta)

    t_p = torch.FloatTensor([
        [1, 0, px],
        [0, 1, py],
        [0, 0, 1]
    ]).unsqueeze(0)

    t_r = torch.FloatTensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ]).unsqueeze(0)

    mat_np = np.dot(t_p.squeeze().numpy(), t_r.squeeze().numpy())
    mat_np_t = torch.from_numpy(mat_np).unsqueeze(0)

    # For some forsaken reason rightmultiplying seems to mean applying the transformation second
    mat = torch.bmm(t_p, t_r)
    #mat1 = t_p
    #mat2 = t_r

    affine_2d = Affine2D()

    res1 = affine_2d(Variable(img), Variable(t_r))

    res2 = affine_2d(res1, Variable(t_p))

    res3 = affine_2d(img, Variable(mat))

    res4 = affine_2d(img, Variable(mat_np_t))

    res3_big = affine_2d(img, Variable(mat), out_size=[512,512])

    res3_small = affine_2d(img, Variable(mat), out_size=[128, 128])

    Presenter().show_image(res1.data[0], "res_1", torch=True, waitkey=False, scale=4)
    Presenter().show_image(res2.data[0], "res_2", torch=True, waitkey=False, scale=4)
    Presenter().show_image(res3.data[0], "res_3", torch=True, waitkey=False, scale=4)
    Presenter().show_image(res3_big.data[0], "res3_big", torch=True, waitkey=False, scale=4)
    Presenter().show_image(res3_small.data[0], "res3_small", torch=True, waitkey=False, scale=4)
    Presenter().show_image(res4.data[0], "res_4", torch=True, waitkey=True, scale=4)

    print("res2 should be the same as res_3 and res_4")


def map_affine_test():
    img = load_env_img(2, 128, 128)
    img = standardize_image(img)
    img = torch.from_numpy(img).float().unsqueeze(0)

    pos = np.asarray([15, 15, 0])
    quat = euler.euler2quat(0, 0, 0)
    pose0 = Pose(pos[np.newaxis, :], quat[np.newaxis, :])

    theta1 = 0.5
    pos = np.asarray([15, 15, 0])
    quat = euler.euler2quat(0, 0, theta1)
    pose1 = Pose(pos[np.newaxis, :], quat[np.newaxis, :])

    D = 10.0
    pos = np.asarray([15 + D*math.cos(theta1), 15 + D*math.sin(theta1), 0])
    quat = euler.euler2quat(0, 0, theta1)
    pose2 = Pose(pos[np.newaxis, :], quat[np.newaxis, :])

    affine = MapAffine(128, 128, 128)
    res1 = affine(img, pose0, pose1)
    res2 = affine(res1, pose1, pose2)
    res3 = affine(img, pose0, pose2)

    prof = SimpleProfiler(torch_sync=True, print=True)
    affinebig = MapAffine(128, 256, 128)
    prof.tick("init")
    res3big = affinebig(img, pose0, pose2)
    prof.tick("affinebig")

    img = load_env_img(2, 32, 32)
    img = standardize_image(img)
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()
    affines = MapAffine(32, 64, 32).cuda()
    torch.cuda.synchronize()
    prof.tick("init")
    res3s = affines(img, pose0, pose2)
    prof.tick("affines")

    prof.print_stats()


    print("Start pose: ", pose0)
    print("    Pose 1: ", pose1)
    print("    Pose 2: ", pose2)

    print("Res2, Res3 and Res3Big should align!")

    Presenter().show_image(img[0], "img", torch=True, waitkey=False, scale=2)
    Presenter().show_image(res1.data[0], "res_1", torch=True, waitkey=False, scale=2)
    Presenter().show_image(res2.data[0], "res_2", torch=True, waitkey=False, scale=2)
    Presenter().show_image(res3.data[0], "res_3", torch=True, waitkey=False, scale=2)
    Presenter().show_image(res3big.data[0], "res3big", torch=True, waitkey=True, scale=2)


if __name__ == "__main__":
    #affine_2d_test()
    map_affine_test()
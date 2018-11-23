import torch
import numpy as np
from torch.autograd import Variable

from learning.modules.img_to_map.fpv_to_global_map import FPVToGlobalMap
from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjectionModuleGlobal, PinholeCameraProjection
from learning.models.semantic_map.sm_params import CAM_FOV, CAM_MAP_X, CAM_MAP_Y
from learning.inputs.pose import Pose
from learning.inputs.vision import standardize_image


def test_data_path():
    import os
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    test_data_file = os.path.join(test_data_dir, "proj_test_data.pickle")
    return test_data_file


def cam_pose_from_state(state):
    cam_pos = state[9:12]
    cam_rot = state[12:16]
    pose = Pose(cam_pos, cam_rot)
    return pose


def extract_test_data():
    from data_io.train_data import load_single_env_supervised_data
    import pickle
    images = []
    cam_poses = []
    for i in range(100):
        superdata = load_single_env_supervised_data(i)
        for sample in superdata:
            image = sample.state.image[:, :, 0:3]
            state = sample.state.state
            cam_pose = cam_pose_from_state(state)
            images.append(image)
            cam_poses.append(cam_pose)

    test_data = {
        "images": images,
        "cam_poses": cam_poses
    }
    with open(test_data_path(), "wb") as fp:
        pickle.dump(test_data, fp)

    print("Saved test data of length: ", len(test_data["images"]))

def coord_grid_test():
    from utils.simple_profiler import SimpleProfiler
    prof = SimpleProfiler()
    print("Coord grid test:")
    projector = PinholeCameraProjection()
    prof.tick(".")
    grid = projector.get_coord_grid(32, 32, True, True)
    prof.tick("a")
    print(grid)

    print("Coord grid fastL")
    prof.tick(".")
    grid_fast = projector.get_coord_grid_fast(32, 32)
    prof.tick("b")

    print(grid_fast)

    prof.print_stats()

    assert (grid == grid_fast).all()


def proj_map_test():
    from utils.simple_profiler import SimpleProfiler
    prof = SimpleProfiler()
    print("Coord grid test:")
    projector = PinholeCameraProjectionModuleGlobal()
    map = projector.get_projection_mapping_local()


def global_proj_test():
    import pickle
    import cv2
    with open(test_data_path(), "rb") as fp:
        test_data = pickle.load(fp)

    for i in range(len(test_data["images"])):
        image = test_data["images"][i]
        pose = test_data["cam_poses"][i]

        cv2.imshow("fpv_image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

        projector = PinholeCameraProjection(
                     map_size=32,
                     map_world_size=32,
                     world_size=30,
                     img_x=256,
                     img_y=144,
                     use_depth=False,
                     cam_fov=CAM_FOV)

        proj_map = projector.get_projection_mapping(pose.position, pose.orientation, local_frame=False, range1=True)

        proj_show = proj_map - proj_map.min()
        proj_show /= (proj_show.max() + 1e-9)

        proj_show = np.concatenate([proj_show, np.zeros_like(proj_show[:, :, 0:1])], axis=2)

        proj_show = cv2.resize(proj_show, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("projection_map", proj_show)
        cv2.waitKey()


def runtest_fpv_to_global_map():
    img_to_map = FPVToGlobalMap(
        source_map_size=32, world_size_px=32, world_size=30, img_w=256, img_h=144,
        res_channels=3, map_channels=3, img_dbg=True)

    import pickle
    import cv2
    with open(test_data_path(), "rb") as fp:
        test_data = pickle.load(fp)

    for i in range(len(test_data["images"])):
        image = test_data["images"][i]
        pose = test_data["cam_poses"][i]

        cv2.imshow("fpv_image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        image = standardize_image(image)
        image_t = Variable(torch.from_numpy(image))
        pose_t = pose.to_torch().to_var()
        pose_t = Pose(pose_t.position.unsqueeze(0), pose_t.orientation.unsqueeze(0))
        image_t = image_t.unsqueeze(0)

        projected, poses = img_to_map(image_t, pose_t, None, show="yes")
        print("Ding")
        print("globalish poses: ", poses)


if __name__ == "__main__":
    #extract_test_data()
    runtest_fpv_to_global_map()
    #global_proj_test()
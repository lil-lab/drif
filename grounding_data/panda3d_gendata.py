import os
import json
import numpy as np
import random
import imageio
from transforms3d import quaternions, euler

from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from direct.gui.OnscreenImage import OnscreenImage
from direct.task import Task
from panda3d.core import *
import grounding_data.multi_loader as ml
import grounding_data.shapenet_loader as sl

from grounding_data.panda_utils import setupGround
import grounding_data.generate_random_config as grc

from data_io.train_data import load_multiple_env_data_from_dir, split_into_segs
import parameters.parameter_server as P

#rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/empty/simulator_empty"
#data_out_dir = "/media/clic/BigStore/grounding_data_sim_8/raw"
rollout_dir = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/data/empty/real_empty"
data_out_dir = "/media/clic/BigStore/grounding_data_rss2020/grounding_data_real_8/raw"

DATA_SPLIT = "train"
#DATA_SPLIT = "test"


center = [2.35, 2.35, 0.0]

ground_color = [0.5, 0.5, 0.5, 1.0]
superbright_color = [1000, 1000, 1000, 1]

NUM_LIGHTS = 4
#SKIP_EVERY_N = 3
SKIP_EVERY_N = 3

NUM_TRAIN_SCENES = 20000
NUM_TRAJECTORIES_PER_SCENE = 2

# For matching data:
#NUM_TRAIN_SCENES = 50000
#NUM_TRAJECTORIES_PER_SCENE = 4

NUM_TINY_TEST_SCENES = 10
NUM_TEST_SCENES = 1000

landmark_positions = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]
IMAGE_H = 96*6
IMAGE_W = 128*6


MATCHING_DATA_ONLY = False


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        # TODO: Generate scene based on an environment configuration from generate_random_config, except with larger number of landmarks.
        self.landmarks = []
        self.landmark_ids = []
        self.landmark_colors = []
        self.current_config = None
        self.focus_object = None
        self.object_radii = {}

        self.spotlights = []
        self.ambient_light = None
        self.ambient_light_path = None
        self.current_mode = "full"

        self.blank_light = None
        self.blank_light_path = None
        self.superbright_light = None
        self.superbright_light_path = None

        self.setBackgroundColor(0, 0, 0, 0)
        self.background_image_path = "test_grass.jpg"
        self.background = OnscreenImage(parent=self.render2dp, image=self.background_image_path)
        self.background.reparentTo(self.render2dp)
        self.background_pos = self.background.getPos()
        self._enable_background(True)

        # TODO: Filters mess with the display region, and mess up the sort of render and render2d
        self.filters = CommonFilters(self.win, self.cam)
        self.filters.setBlurSharpen(0.8)

        # TODO: Consider units to be in meters, and scale the objects accordingly
        self.obj_loader = sl.ShapeNetLoader(self.loader, DATA_SPLIT)
        #self.obj_loader = yl.YCBLoader(self.loader, DATA_SPLIT)
        #self.obj_loader = ml.MultiLoader(self.loader, DATA_SPLIT)

        self.render.setShaderAuto()
        self.ground = setupGround(self.render)
        self.setupLights()
        self.randomizeLights()

        self.cam.get_node(0).get_lens().setFov(84, 63)
        self.cam.get_node(0).get_lens().setNear(0.2)
        self.disable_mouse()

        props = WindowProperties()
        props.setSize(IMAGE_W, IMAGE_H)
        self.win.requestProperties(props)

        if DATA_SPLIT == "tinytest":
            self.num_scenes = NUM_TINY_TEST_SCENES
        elif DATA_SPLIT == "train":
            self.num_scenes = NUM_TRAIN_SCENES
        elif DATA_SPLIT == "test":
            self.num_scenes = NUM_TEST_SCENES
        else:
            raise ValueError(f"Unknown data split: {DATA_SPLIT}")

    def do_render(self, cycles=2):
        for i in range(cycles):
            self.taskMgr.step()

    def run(self):
        rollouts = load_multiple_env_data_from_dir(rollout_dir, single_proc=True)
        rollouts = split_into_segs(rollouts)
        scenes_to_generate = list(range(self.num_scenes))
        max_scene = self.get_current_scene_number()
        scenes_still_to_generate = [s for s in scenes_to_generate if s > max_scene]
        print(f"Skipping scenes until {max_scene}")
        # Loop over scenes
        for scene_idx in scenes_still_to_generate:
            self.newScene()
            # Save configuration json
            self.save_json(f"{scene_idx}--config", self.current_config)
            # Loop over trajectories
            picked_rollouts = random.sample(rollouts, NUM_TRAJECTORIES_PER_SCENE)
            for traj_idx, rollout in enumerate(picked_rollouts):

                # Loop over poses within trajectory:
                for t, sample in enumerate(rollout):
                    # The trajectories tend to have around 12-15 identical frames in the beginning. Skip those
                    if t < 12 or t > 30 or t % SKIP_EVERY_N != 0:
                        continue

                    # Set the pose to drone camera pose
                    cam_pos = sample["state"].state[9:12]
                    cam_rot_quat = sample["state"].state[12:16]
                    cam_rot_rpy = euler.quat2euler(cam_rot_quat)
                    cam_pos_panda = [cam_pos[1], cam_pos[0], -cam_pos[2]]
                    cam_rot_panda = [-cam_rot_rpy[2] * 180 / 3.14159,
                                     cam_rot_rpy[1] * 180 / 3.14159,
                                     cam_rot_rpy[0] * 180 / 3.14159]
                    self.cam.setPos(*cam_pos_panda)
                    self.cam.setHpr(*cam_rot_panda)

                    # Save object bounding box coordinates:
                    num_objects_visible = self.compute_and_save_frame_geometry(f"{scene_idx}_{traj_idx}_{t}--geometry")
                    if num_objects_visible < 2:
                        print("Skipping {scene_idx}_{traj_idx}_{t} due to too few objects")
                        continue

                    # Set the background image
                    background_image = sample["state"].image
                    self.randomizeLights()
                    self.do_render(1)

                    # Render the frame and save a screenshot
                    self.setMode("scene")
                    self.do_render(5)
                    self.save_screenshot(f"{scene_idx}_{traj_idx}_{t}--scene")

                    # Switch to no-shadows mode
                    self.setMode("noshadows")
                    self.do_render(2)
                    self.save_screenshot(f"{scene_idx}_{traj_idx}_{t}--noshadows")

                    # if not MATCHING_DATA_ONLY:
                    # Render the frame and save a screenshot
                    self.setMode("objects")
                    self.do_render()
                    self.save_screenshot(f"{scene_idx}_{traj_idx}_{t}--objects")

                    # Switch to ground plane only mode
                    self.setMode("ground")
                    self.do_render()
                    self.save_screenshot(f"{scene_idx}_{traj_idx}_{t}--ground")

                    if not MATCHING_DATA_ONLY:
                        # Render segmentation masks for each object present
                        for i, obj_id in enumerate(set(self.landmark_ids)):
                            self.focus_object = obj_id
                            self.setMode("mask")
                            self.do_render()
                            self.save_screenshot(f"{scene_idx}_{traj_idx}_{t}--object_{obj_id}")

                    self.save_background(f"{scene_idx}_{traj_idx}_{t}--background", background_image)

                    # Save metadata linking this rendering to the original rollout
                    metadata = {
                        "scene_idx": scene_idx,
                        "traj_idx": traj_idx,
                        "timestep": t,
                        "env_id": sample["env_id"],
                        "set_idx": sample["set_idx"],
                        "seg_idx": sample["seg_idx"],
                        "cam_pos": list(cam_pos),
                        "cam_pos_panda": list(cam_pos_panda),
                        "cam_rot": list(cam_rot_quat),
                        "cam_rot_panda": list(cam_rot_panda)
                    }
                    self.save_json(f"{scene_idx}_{traj_idx}_{t}--metadata", metadata)

    """
    def setBackgroundFromNumpy(self, nparr):
        imageTexture = Texture("image")
        nparr = np.ascontiguousarray(np.flip(nparr, 0))
        imageTexture.setup2dTexture(nparr.shape[1], nparr.shape[0], Texture.TUnsignedByte, Texture.FRgb)
        p = PTAUchar.emptyArray(0)
        try:
            p.setData(nparr)
        except AssertionError:
            pass
        imageTexture.setRamImage(CPTAUchar(p))
        self.background.setImage(imageTexture)
    """

    def newScene(self):
        print("TASK: New Scene")
        all_valid_ids = self.obj_loader.get_all_valid_object_ids()
        self.current_config = grc.gen_config(all_valid_ids)
        self.current_config["enabled"] = []

        # Clear the previous config
        for landmarkName, landmark in zip(self.landmark_ids, self.landmarks):
            landmark.removeNode()
        self.landmarks = []
        self.landmark_ids = []
        self.landmark_colors = []

        # Note the swapping of X and Y positions! This is because Panda uses a different coordinate convention,
        # but I'd like to keep all configs and files consistent.
        for i, (y_pos, x_pos, radius, rpy, landmarkName) in enumerate(zip(
                self.current_config["xPos"],
                self.current_config["zPos"],
                self.current_config["radius"],
                self.current_config["rpy"],
                self.current_config["landmarkName"])):
            landmark, obj_id = self.obj_loader.get_object_by_id(landmarkName)
            # Disable the landmarks that can't be retrieved or don't meet criteria
            if landmark is None:
                self.current_config["enabled"].append(False)
                continue
            else:
                self.current_config["enabled"].append(True)

            if landmarkName in self.object_radii:
                radius = self.object_radii[landmarkName]
                self.current_config["radius"][i] = radius
            else:
                self.object_radii[landmarkName] = radius

            landmark.reparentTo(self.render)
            x_pos = x_pos * 4.7 / 1000
            y_pos = y_pos * 4.7 / 1000
            radius = radius * 4.7 / 1000
            print("Landmark Pos:", x_pos, y_pos)
            landmark.setHpr(self.render, *rpy)
            landmark.setScale(landmark.getScale() * radius)
            landmark.setDepthOffset(10)
            landmark.setTwoSided(True)
            cb = self.obj_loader.getCenterBottom(landmark, self.render)
            offset = [-c for c in cb]
            landmark.setPos(self.render, x_pos + offset[0], y_pos + offset[1], offset[2])
            self.landmarks.append(landmark)
            self.landmark_ids.append(landmarkName)
            self.landmark_colors.append(landmark.getColor())

        self.switchOnLights()
        self.randomizeLights()
        return Task.cont

    def setMode(self, mode):
        self.switchOnLights()
        self._enable_background(True)
        self._background_infront(False)
        self.ground.setColor(*ground_color)

        # Reset things to normal:
        if mode == "scene":
            self._enable_shadows(True)

        elif mode == "objects":
            self._enable_shadows(True)
            self.ground.setColor(1, 1, 1, 0)

        elif mode == "noshadows":
            self._enable_shadows(False)

        elif mode == "ground":
            self._enable_shadows(False)
            self._enable_background(False)
            self._blackout_objects()
            self._whiteout_ground()

        elif mode == "mask":
            self._enable_shadows(False)
            self._enable_background(False)
            self._blackout_objects()
            self._blackout_ground()
            self._whiteout_landmark(self.focus_object)

        elif mode == "background":
            self._enable_background(True)
            self._background_infront(True)

    def _enable_background(self, enable):
        if enable:
            # TODO: support loading from numpy
            self.background.setPos(self.background_pos)
            self.cam2dp.node().getDisplayRegion(0).setSort(-1)
        else:
            self.background.setPos(-100, -100, -100)

    def _background_infront(self, enable):
        if enable:
            self.cam2dp.node().getDisplayRegion(0).setSort(10)
        else:
            self.cam2dp.node().getDisplayRegion(0).setSort(-1)

    def _whiteout_ground(self):
        self.ground.clearLight()
        self.ground.setColor(*ground_color)
        self.ground.setLight(self.superbright_light_path)

    def _blackout_ground(self):
        self.ground.setColor(0, 0, 0, 1)
        self.ground.clearLight()
        self.ground.setLightOff()

    def _enable_shadows(self, on):
        for slnp, slight in self.spotlights:
            slight.setShadowCaster(on)
        self.render.setShaderAuto()

    def _blackout_objects(self):
        for color, landmark in zip(self.landmark_colors, self.landmarks):
            self._disable_lights(landmark)

    def _whiteout_landmark(self, obj_id):
        for lm_id, landmark, color in zip(self.landmark_ids, self.landmarks, self.landmark_colors):
            if lm_id == obj_id:
                landmark.setLight(self.superbright_light_path)

    def _disable_lights(self, object):
        object.clearLight()
        object.setLightOff(self.ambient_light_path)
        for snlp, slight in self.spotlights:
            object.setLightOff(snlp)

    def save_screenshot(self, name):
        filename = f"{data_out_dir}/{name}.png"
        os.makedirs(data_out_dir, exist_ok=True)
        self.screenshot(filename, False, self.win)

    def save_background(self, name, np_image):
        filename = f"{data_out_dir}/{name}.png"
        os.makedirs(data_out_dir, exist_ok=True)
        imageio.imsave(filename, np_image)

    def _get_8_corner_points(self, tight_bounds):
        ltb = tight_bounds[0]
        utb = tight_bounds[1]
        lll = Point3F(ltb[0], ltb[1], ltb[2])
        llu = Point3F(ltb[0], ltb[1], utb[2])
        lul = Point3F(ltb[0], utb[1], ltb[2])
        luu = Point3F(ltb[0], utb[1], utb[2])
        ull = Point3F(utb[0], ltb[1], ltb[2])
        ulu = Point3F(utb[0], ltb[1], utb[2])
        uul = Point3F(utb[0], utb[1], ltb[2])
        uuu = Point3F(utb[0], utb[1], utb[2])
        return [lll, llu, lul, luu, ull, ulu, uul, uuu]

    def compute_and_save_frame_geometry(self, name):
        bbox_data = {
            "landmarkNames": [],
            "landmarkBboxes": [],
            "landmarkCentersOnGround": [],
            "landmarkCenters": [],
            "landmarkCoordsInCam": []
        }
        num_boxes = 0
        for obj_id, object in zip(self.landmark_ids, self.landmarks):
            bbox_data["landmarkNames"].append(obj_id)
            lens = self.cam.get_node(0).get_lens()
            # ------------------------------------------------------
            # 1. Compute and store bounding boxes
            tight_bounds = object.getTightBounds(self.render)
            corner_points = self._get_8_corner_points(tight_bounds)
            center_bottom = self.obj_loader.getCenterBottom(object, self.render)
            center = self.obj_loader.getCenter(object, self.render)
            c_and_cb = [Point3F(*center), Point3F(*center_bottom)]
            all_points = corner_points + c_and_cb
            relative_center = self.obj_loader.getCenter(object, self.camera)

            # Convert the point to the 3-d space of the camera
            points_camera = [self.cam.getRelativePoint(self.render, pt) for pt in all_points]
            # Convert to the film coordinates
            points_film = [Point2F() for _ in points_camera]
            points_in = [lens.project(ptc, ptf) for ptc, ptf in zip(points_camera, points_film)]
            num_points_in = sum(points_in)
            if num_points_in >= 4:
                # Convert to pixel2D coordinates
                points_film_3d = [Point3F(pt[0], 0, pt[1]) for pt in points_film]
                # And then convert it to pixel2d coordinates
                points_px_3d = [self.pixel2d.getRelativePoint(self.render2d, pt) for pt in points_film_3d]
                # Convert to x-right, y-down
                points_px_2d = [[pt[0], -pt[2]] for pt in points_px_3d]
                # Extract the 2D bounding box:
                min_x = int(min([pt[0] for pt in points_px_2d]))
                max_x = int(max([pt[0] for pt in points_px_2d]))
                min_y = int(min([pt[1] for pt in points_px_2d]))
                max_y = int(max([pt[1] for pt in points_px_2d]))
                # Clip to image bounds
                min_x = min(max(0, min_x), IMAGE_W)
                max_x = min(max(0, max_x), IMAGE_W)
                min_y = min(max(0, min_y), IMAGE_H)
                max_y = min(max(0, max_y), IMAGE_H)

                bbox_2d = [[min_x, min_y], [max_x, max_y]]
                center_2d = [int(x) for x in points_px_2d[8]]
                center_bottom_2d = [int(x) for x in points_px_2d[9]]
                # Log results
                bbox_data["landmarkBboxes"].append(bbox_2d)
                bbox_data["landmarkCenters"].append(center_2d)
                bbox_data["landmarkCentersOnGround"].append(center_bottom_2d)
                bbox_data["landmarkCoordsInCam"].append(list(points_camera[8]))
                num_boxes += 1
            # Entire object is not within view - store None instead of the bounding box
            else:
                bbox_data["landmarkBboxes"].append(None)
                bbox_data["landmarkCenters"].append(None)
                bbox_data["landmarkCentersOnGround"].append(None)
                bbox_data["landmarkCoordsInCam"].append(None)

        filename = f"{data_out_dir}/{name}.json"
        os.makedirs(data_out_dir, exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(bbox_data, fp)
        return num_boxes

    def save_json(self, name, data):
        filename = f"{data_out_dir}/{name}.json"
        os.makedirs(data_out_dir, exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(data, fp)

    def get_current_scene_number(self):
        try:
            scene_files = [f for f in os.listdir(data_out_dir) if "--scene" in f]
            scene_numbers = [int(s.split("_")[0]) for s in scene_files]
            return max(scene_numbers)
        except FileNotFoundError:
            return 0

    def setupLights(self):
        self.ambient_light = AmbientLight('ambient')
        self.ambient_light_path = self.render.attachNewNode(self.ambient_light)
        self.render.setLight(self.ambient_light_path)

        self.superbright_light = AmbientLight('superbright')
        self.superbright_light_path = self.render.attachNewNode(self.superbright_light)
        self.superbright_light.setColor(VBase4(*superbright_color))

        self.blank_light = AmbientLight('blank')
        self.blank_light_path = self.render.attachNewNode(self.blank_light)
        self.blank_light.setColor(VBase4(0, 0, 0, 1))
        self.render.setLight(self.blank_light_path)

        for i in range(NUM_LIGHTS):
            slight = Spotlight(f'slight{i}')
            slight.setShadowCaster(True, 1024, 1024)
            lens = PerspectiveLens(160, 160)
            slight.setLens(lens)
            slnp = self.render.attachNewNode(slight)
            slnp.setPos(10, 10, 10)
            slnp.lookAt(*center)
            slnp.reparentTo(self.render)
            self.render.setLight(slnp)
            self.spotlights.append((slnp, slight))

        self.render.setShaderAuto()

    def switchOnLights(self):
        # First reset all light overrides (e.g. setLightOff) from all objects in the environment
        for landmark in self.landmarks:
            landmark.clearLight()
            #landmark.setLightOff(self.superbright_light_path)
        self.ground.clearLight()
        self.render.clearLight()
        # Then enable the lights on the scene
        self.render.setLight(self.blank_light_path)
        self.render.setLight(self.ambient_light_path)
        for snlp, slight in self.spotlights:
            self.render.setLight(snlp)

    def randomizeLights(self):
        # Draw ambient light intensity from a uniform, and color from a gaussian around white.
        ambient_brightness = np.random.uniform(0.4, 1.0)
        ambient_color = list(np.random.normal([ambient_brightness, ambient_brightness, ambient_brightness, 1.0],
                                              ambient_brightness * 0.05, 4).clip(0, 1.0))

        self.ambient_light.setColor(VBase4(*ambient_color))
        total_spot_intensity = np.random.uniform(0.5, 2.0)
        #total_spot_intensity = 2.0
        intensity = total_spot_intensity / len(self.spotlights)

        for slnp, slight in self.spotlights:
            #intensity = np.random.uniform(0.15, 0.4)
            spot_color = list(np.random.normal([intensity, intensity, intensity, 1.0], intensity * 0.05, 4).clip(0, 1.0))
            slight.setColor(VBase4(*spot_color))
            angle = np.random.uniform(-180, 180, 1)
            distance = np.random.normal(10.0, 3.0, 1)
            height = float(np.random.uniform(1.0, 20.0, 1))
            x = float(np.cos(np.radians(angle)) * distance)
            y = float(np.sin(np.radians(angle)) * distance)
            slnp.setPos(x, y, height)
            slnp.lookAt(*center)


if __name__ == "__main__":
    P.initialize_empty_experiment()
    app = MyApp()
    app.run()
import os
import csv
import json

sem_path = "/home/clic/BigStore/shapenet_sem"
metadata_path = os.path.join(sem_path, "metadata.csv")
models_dir = os.path.join(sem_path, "models")
object_ids_path = os.path.join(sem_path, "object_ids.json")
object_colors_path = os.path.join(sem_path, "object_colors.json")

TRAIN_SPLIT_COUNT = 7000
TEST_SPLIT_COUNT = 200


class ShapeNetLoader:

    def __init__(self, loader, data_split):
        self.current = 0
        self.loader = loader
        with open(metadata_path) as csvfile:
            reader = csv.DictReader(csvfile)
            self.metadata = [row for row in reader]
        self.metadata_by_id = {row["fullId"].split(".")[1]: row for row in self.metadata}
        with open(object_ids_path, "r") as fp:
            self.valid_object_ids = json.load(fp)

        # Choose a different split of objects for test/train
        if data_split == "train":
            self.min_object = 0
            self.num_objects = TRAIN_SPLIT_COUNT
        elif data_split == "test":
            self.min_object = TRAIN_SPLIT_COUNT
            self.num_objects = TEST_SPLIT_COUNT
        if self.min_object > 0:
            self.valid_object_ids = list(sorted(self.valid_object_ids))[self.min_object:]
        if self.num_objects > 0:
            self.valid_object_ids = list(sorted(self.valid_object_ids))[:self.num_objects]

        with open(object_colors_path, "r") as fp:
            self.object_colors = json.load(fp)

        self.all_object_ids = [row["fullId"] for row in self.metadata]
        self.object_sizes = {}

    def get_all_valid_object_ids(self):
        return self.valid_object_ids

    def get_all_object_ids(self):
        return self.all_object_ids

    def get_object(self, metadata_row):
        """
        :param metadata_row:
        :return: A tuple (path, scale), where scale is a scaling factor to be applied to scale the model to 1 unit.
        """
        model_id = metadata_row["fullId"].split(".")[1]
        model_path = os.path.join(models_dir, f"{model_id}.obj")

        # Get the units and figure out scaling
        unit_str = metadata_row["unit"]
        if not unit_str:
            return None, None, None
        unit = float(metadata_row["unit"])

        dims_str = metadata_row["aligned.dims"]
        dims = [float(f.split("\\")[0]) for f in dims_str.split(",")]
        max_dim = max(dims)
        if max_dim < 0.001: # was 0.001
            return None, None, None

        landmark = self.loader.loadModel(model_path)
        wlh = self.getWLH(landmark)
        scale = 1.0 / max(wlh)
        landmark.setScale(scale, scale, scale)
        wlh = self.getWLH(landmark)

        if model_id in self.object_colors:
            color = self.object_colors[model_id]
            color = [c / 255.0 for c in color]
            #print("Coloring: ", color)
            landmark.setColor(*color)

        #curr_color = landmark.getColor()
        #color = [c / 3 for c in curr_color]
        #landmark.setColor(*color)

        # Figure out center, so that we can re-center to zero
        #center = self.getCenterBottom(landmark)
        #offset = [-c for c in center]

        # Skip objects that are very flat or pointy
        if min(wlh) < 0.2: # was 0.2
            return None, None, None

        return landmark, model_id

    def get_object_by_id(self, object_id):
        metadata_row = self.metadata_by_id[object_id]
        return self.get_object(metadata_row)

    def getCenterBottom(self, object, parent_node):
        lb, rb = object.getTightBounds(parent_node)
        cx = (rb[0] + lb[0]) / 2
        cy = (rb[1] + lb[1]) / 2
        cz = min(rb[2], lb[2])
        return [cx, cy, cz]

    def getCenter(self, object, parent_node):
        lb, rb = object.getTightBounds(parent_node)
        cx = (rb[0] + lb[0]) / 2
        cy = (rb[1] + lb[1]) / 2
        cz = (rb[2] + lb[2]) / 2
        return [cx, cy, cz]

    def getWLH(self, object):
        lb, rb = object.getTightBounds()
        wlh = [rb[0] - lb[0], rb[1] - lb[1], rb[2] - lb[2]]
        return wlh

    def next(self, loop=True):
        object = None
        while not object:
            if self.current >= len(self.metadata):
                if loop:
                    self.current = 0
                else:
                    raise StopIteration

            object = self.get_object(self.metadata[self.current])
            self.current += 1
        return object

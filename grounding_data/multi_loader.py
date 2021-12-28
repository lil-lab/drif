import os

from grounding_data.ycb_loader import YCBLoader
from grounding_data.shapenet_loader import ShapeNetLoader


class MultiLoader:

    def __init__(self, loader, data_split):
        self.current = 0
        self.ycb_loader = YCBLoader(loader, data_split)
        self.ycb_ids = self.ycb_loader.get_all_valid_object_ids()
        self.shapenet_loader = ShapeNetLoader(loader, data_split)
        self.shapenet_ids = self.shapenet_loader.get_all_valid_object_ids()

        self.model_id_map = {mid: dataset
                             for mid, dataset in zip(self.ycb_ids + self.shapenet_ids,
                                                     ["ycb"]*len(self.ycb_ids) + ["shapenet"]*len(self.shapenet_ids))}
        self.all_ids = list(sorted(self.model_id_map.keys()))

    def get_all_valid_object_ids(self):
        return list(self.model_id_map.keys())

    def get_all_object_ids(self):
        return list(self.model_id_map.keys())

    def get_object_by_id(self, model_id):
        if self.model_id_map[model_id] == "ycb":
            return self.ycb_loader.get_object_by_id(model_id)
        elif self.model_id_map[model_id] == "shapenet":
            return self.shapenet_loader.get_object_by_id(model_id)
        else:
            raise ValueError("Whaat?")

    def next(self, loop=True):
        object = None
        while not object:
            if self.current >= len(self.model_id_map):
                if loop:
                    self.current = 0
                else:
                    raise StopIteration
            object = self.get_object_by_id(self.all_ids[self.current])
            self.current += 1
        return object

    def getCenterBottom(self, *args, **kwargs):
        return self.shapenet_loader.getCenterBottom(*args, **kwargs)

    def getCenter(self, *args, **kwargs):
        return self.shapenet_loader.getCenter(*args, **kwargs)


import os

ycb_dir = "/media/clic/BigStore/ycb"


class YCBLoader:

    def __init__(self, loader, data_split):
        self.current = 0
        self.loader = loader
        self.all_object_ids = list(sorted(os.listdir(ycb_dir)))

    def get_all_valid_object_ids(self):
        return self.all_object_ids

    def get_all_object_ids(self):
        return self.all_object_ids

    def get_object(self, model_id):
        """
        :param metadata_row:
        :return: A tuple (path, scale), where scale is a scaling factor to be applied to scale the model to 1 unit.
        """
        model_path = os.path.join(ycb_dir, model_id, "tsdf", "textured.obj")

        landmark = self.loader.loadModel(model_path)
        wlh = self.getWLH(landmark)
        scale = 1.0 / max(wlh)
        landmark.setScale(scale, scale, scale)
        return landmark, model_id

    def get_object_by_id(self, model_id):
        return self.get_object(model_id)

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
            if self.current >= len(self.all_object_ids):
                if loop:
                    self.current = 0
                else:
                    raise StopIteration

            object = self.get_object(self.all_object_ids[self.current])
            self.current += 1
        return object

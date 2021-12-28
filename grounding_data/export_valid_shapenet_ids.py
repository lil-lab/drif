import os
from math import pi, sin, cos
import numpy as np
import json

from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from direct.task import Task
from panda3d.core import *
import grounding_data.shapenet_loader as sl


out_file = "object_ids.json"


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        #self.obj_loader = sl.ShapeNetLoader(self.loader)
        props = WindowProperties()
        props.setSize(256, 192)
        self.win.requestProperties(props)

        self.obj_id_list = list()

        self.taskMgr.add(self.getValidIdsTask, "ValidIdsTask")

    def getValidIdsTask(self, task):
        # The loader next only returns valid object IDs
        try:
            obj, offset, obj_id = self.obj_loader.next(loop=False)
            print(f"Exporting {len(self.obj_id_list)}/{len(self.obj_loader.all_object_ids)} object: {obj_id}")
            self.obj_id_list.append(obj_id)
            return Task.cont
        except StopIteration:
            with open(out_file, "w") as fp:
                json.dump(self.obj_id_list, fp, indent=4)
            return None


app = MyApp()
app.run()
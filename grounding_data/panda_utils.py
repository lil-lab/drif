from panda3d.core import *


def setupGround(render):
    format = GeomVertexFormat.getV3n3c4()
    format = GeomVertexFormat.registerFormat(format)
    vdata = GeomVertexData('name', format, Geom.UHStatic)
    vdata.setNumRows(4)
    vertex = GeomVertexWriter(vdata, 'vertex')
    vertex.addData3f(0, 0, 0)
    vertex.addData3f(4.7, 0, 0)
    vertex.addData3f(4.7, 4.7, 0)
    vertex.addData3f(0, 4.7, 0)
    color = GeomVertexWriter(vdata, 'color')
    gcolor = (1, 1, 1, 0)
    color.addData4f(*gcolor)
    color.addData4f(*gcolor)
    color.addData4f(*gcolor)
    color.addData4f(*gcolor)
    normal = GeomVertexWriter(vdata, 'normal')
    normal.addData3f(0, 0, 1)
    normal.addData3f(0, 0, 1)
    normal.addData3f(0, 0, 1)
    normal.addData3f(0, 0, 1)

    prima = GeomTriangles(Geom.UHStatic)
    prima.addVertices(0, 1, 3)
    primb = GeomTriangles(Geom.UHStatic)
    primb.addVertices(1, 2, 3)
    geom = Geom(vdata)
    geom.addPrimitive(prima)
    geom.addPrimitive(primb)
    node = GeomNode('ground')
    node.addGeom(geom)

    ground = render.attachNewNode(node)
    ground.setColor(1.0, 1.0, 1.0, 1.0)
    ground.setTransparency(TransparencyAttrib.MAlpha)
    return ground

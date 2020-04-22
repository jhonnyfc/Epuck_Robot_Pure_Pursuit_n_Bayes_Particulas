import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../pypot')
sys.path.append('../../enum34')
sys.path.append('../../pyserial-3.4')

from pypot.vrep.remoteApiBindings import vrep
from pypot.vrep.io import VrepIO, VrepIOErrors
from routine import Routine
from logger import Logger
import pandas as pd

import math
from math import sqrt, cos, sin
from numpy import average, mean, array, argmax, argmin, ones, zeros_like
from numpy.random import rand, randint
from random import sample, choice
from time import sleep
from copy import copy
from threading import Event, Condition
# from multiprocessing import Process as ParralelClass
from threading import Thread as ParralelClass
from scipy.spatial.transform import Rotation as R

from vrep.observer import Observer
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage
dimPo = 150

class MapBuilder():
    def __init__(self, pypot_io, res):
        self.io = pypot_io
        self._registered_objects = {}
        # Get objects from the scene
        self.handles, _, _, self.names = self.io.call_remote_api("simxGetObjectGroupData", vrep.sim_object_shape_type, 0, streaming=True)
        
        self.worldminx, self.worldmax, self.worldminy, self.worldmaxy = self.get_word_size()
        
        self.reso = res
        self.xw = int(round((self.worldmax - self.worldminx) / self.reso))
        self.yw = int(round((self.worldmaxy - self.worldminy) / self.reso))

        self.occupacy_map = [[0.0 for _ in range(self.yw)] for _ in range(self.xw)]

        self.occupy_grid()
        
    def occupy_grid(self):
        for h, n in zip(self.handles, self.names):
            if 'Cylinder' in n or 'Cuboid' in n or 'Wall' in n or 'Tree' in n :
                #print(n)
                objx, objy, objtheta, minx, maxx, miny, maxy = self.get_bbox(h)
                #print(objx, objy, objtheta, minx, maxx, miny, maxy)
                for ix in range(self.xw):
                    for iy in range(self.yw):
                        x, y = self.frommat2Coord(ix, iy)
                        xp, yp = self.xy2objcood(x, y, objx, objy, objtheta)
                        if self.pointinbbox(xp,yp, minx, maxx, miny, maxy):
                            self.occupacy_map[ix][iy] = 1.0
        
    def get_bbox(self, handle):
        obj_position = self.io.call_remote_api("simxGetObjectPosition", handle, self.handles[0], streaming=True)
        # get object orientation
        quaternion = self.io.call_remote_api("simxGetObjectQuaternion", handle, -1, streaming=True)
        r = R.from_quat(quaternion)
        aux,obj_yaw, _ = r.as_euler('zyx')
        if aux < 0:
            obj_yaw = math.pi/2.0 - obj_yaw
        else:
            obj_yaw = obj_yaw - math.pi/2.0
            
        _, _,  obj_yaw = self.io.call_remote_api("simxGetObjectOrientation", handle, self.handles[0], streaming=True)
        objminx = self.io.call_remote_api("simxGetObjectFloatParameter", handle, 15, streaming=True)
        objmax = self.io.call_remote_api("simxGetObjectFloatParameter", handle, 18, streaming=True)
        objminy = self.io.call_remote_api("simxGetObjectFloatParameter", handle, 16, streaming=True)
        objmaxy = self.io.call_remote_api("simxGetObjectFloatParameter", handle, 19, streaming=True)
        return obj_position[0], obj_position[1], obj_yaw, objminx, objmax, objminy, objmaxy
            
    def get_word_size(self):
        h = 1
        worldminx = self.io.call_remote_api("simxGetObjectFloatParameter", self.handles[h], 15, streaming=True)
        worldmax = self.io.call_remote_api("simxGetObjectFloatParameter", self.handles[h], 18, streaming=True)
        worldminy = self.io.call_remote_api("simxGetObjectFloatParameter", self.handles[h], 16, streaming=True)
        worldmaxy = self.io.call_remote_api("simxGetObjectFloatParameter", self.handles[h], 19, streaming=True)
        worldminx = round(worldminx,2)
        worldmax = round(worldmax,2)
        worldminy = round(worldminy,2)
        worldmaxy = round(worldmaxy,2)
        return worldminx, worldmax, worldminy, worldmaxy

    def fromCoord2mat(self, x, y):
        i = int((x - self.worldminx)/self.reso)
        j = int((y - self.worldminy)/self.reso)
        return i,j

    def frommat2Coord(self, i, j):
        x = float(i)*self.reso + self.worldminx
        y = float(j)*self.reso + self.worldminy
        return x,y
    
    def xy2objcood(self, x, y, objx, objy, theta):
        x_in_obj = (x-objx)*cos(theta) + (y-objy)*sin(theta)
        y_in_obj = -(x-objx)*sin(theta) + (y-objy)*cos(theta)
        return x_in_obj, y_in_obj
    
    def pointinbbox(self, x,y, xmin, xmax, ymin, ymax):
        if xmin < x < xmax and ymin < y < ymax:
            return True
        else:
            return False
    
    def pointinbBoxJ(self, x,y):
        if self.worldminx < x <  self.worldmax and self.worldminy <  y < self.worldmaxy:
            return True
        else:
            return False
    
    def pointinbMatJ(self, i, j):
        x,y = self.frommat2Coord(i, j)
        if self.worldminx < x <  self.worldmax and self.worldminy <  y < self.worldmaxy:
            return True
        else:
            return False
        
def printMapa(map):
    metaRosa = [-0.5,0.5]
    metaAmarillo = [1.5,2]
    metaRoja = [0.7,-0.65]
    metaAzul = [-0.5,-1.75]
    
    mx, my = calc_grid_index(map)
    fig = plt.figure(figsize=(7,7));
    fig.canvas.draw();

    ax = fig.add_subplot(111);
    maxp = max([max(igmap) for igmap in map.occupacy_map]);
    ax.pcolor(mx, my, map.occupacy_map, vmax=maxp, cmap=plt.cm.get_cmap("Blues"));
    plt.scatter(metaRosa[0],metaRosa[1],c = 'pink', s = dimPo)
    plt.scatter(metaRoja[0],metaRoja[1],c = 'red', s = dimPo)
    plt.scatter(metaAmarillo[0],metaAmarillo[1],c = 'yellow', s= dimPo)
    plt.scatter(metaAzul[0],metaAzul[1],c = 'blue', s= dimPo)
    ax.axis("equal");

def calc_grid_index(gmap):
    mx, my = np.mgrid[slice(gmap.worldminx - gmap.reso / 2.0, gmap.worldmax + gmap.reso / 2.0, gmap.reso),
                  slice(gmap.worldminy - gmap.reso / 2.0, gmap.worldmaxy + gmap.reso / 2.0, gmap.reso)]

    return mx, my

def dilateMap(map,nIter = 4):
    print('Para que el robot no se choque con ningun muro se ha decidido dilatar el map',nIter,'iteraciones')
    Immap = np.array(map.occupacy_map)
    DilateImmap = scipy.ndimage.binary_dilation(Immap, iterations=nIter).astype(Immap.dtype)
    map.occupacy_map = DilateImmap.tolist()
    printMapa(map)
    return map
    
def getMapa(resolucion = 0.02):
    try:
        vrep_host = '127.0.0.1'
        vrep_port = 19997
        scene = None
        start = False
        print('Obteniendo Mapa...')
        io = VrepIO(vrep_host, vrep_port, scene, start)
        io.restart_simulation()


        gmapa = MapBuilder(io, resolucion)
        io.stop_simulation()
        io.close()
        print('El mapa se ha construido con exito')
        return gmapa
    except:
        print("Error el la conexion con el simulador, ejecute este fragmento de nuevo")
        print('Error al obtener mapa')
        return None
    
def showMapAndRoute(gmap,caminoMasCorto):
    metaRosa = [-0.5,0.5]
    metaAmarillo = [1.5,2]
    metaRoja = [0.7,-0.65]
    metaAzul = [-0.5,-1.75]
    
    fig = plt.figure(figsize=(8,8))
    fig.canvas.draw()
    mx, my = calc_grid_index(gmap)
    maxp = max([max(igmap) for igmap in gmap.occupacy_map])
    plt.pcolor(mx, my, gmap.occupacy_map, vmax=maxp, cmap=plt.cm.get_cmap("Blues"))
    plt.scatter(metaRosa[0],metaRosa[1],c = 'pink', s = dimPo)
    plt.scatter(metaRoja[0],metaRoja[1],c = 'red', s = dimPo)
    plt.scatter(metaAmarillo[0],metaAmarillo[1],c = 'yellow', s= dimPo)
    plt.scatter(metaAzul[0],metaAzul[1],c = 'blue', s = dimPo)
    plt.plot(caminoMasCorto[:,0],caminoMasCorto[:,1],'-r')
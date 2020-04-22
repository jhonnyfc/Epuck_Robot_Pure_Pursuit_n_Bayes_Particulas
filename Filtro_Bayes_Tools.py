import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

class GridMap():
    def __init__(self):
        self.data = None
        self.xy_reso = None # Resolución. Tamaño de la celda al discretizar
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None
        self.xw = None # Tamaño matriz eje x
        self.yw = None # Tamaño matriz eje y
        self.dx = 0.0  # variables para almacenar la distancia 
        self.dy = 0.0  # que se ha movido el robot

def normalize_probability(gmap):
    #sump = sum([sum(igmap) for igmap in gmap.data])
    sump = np.sum(gmap.data)
    
    #for ix in range(gmap.xw):
     #   for iy in range(gmap.yw):
      #      gmap.data[ix,iy] /= sump

    gmap.data = gmap.data / sump
    return gmap

def init_gmap(gmap):
    # grid map param
    minx = gmap.worldminx 
    maxx = gmap.worldmax
    miny = gmap.worldminy
    maxy = gmap.worldmaxy
    
    grid_map = GridMap()

    grid_map.xy_reso = gmap.reso
    grid_map.minx = minx
    grid_map.miny = miny
    grid_map.maxx = maxx
    grid_map.maxy = maxy
    grid_map.xw = int(round((grid_map.maxx - grid_map.minx) / grid_map.xy_reso))
    grid_map.yw = int(round((grid_map.maxy - grid_map.miny) / grid_map.xy_reso))

    #grid_map.data = [[1.0 for _ in range(grid_map.yw)] for _ in range(grid_map.xw)]
    grid_map.data = np.zeros((grid_map.yw,grid_map.xw))
#     grid_map.data[:,:] = 1
    grid_map.data[26,225] = 1
    grid_map = normalize_probability(grid_map)

    return grid_map

def fromCoord2mat(x, y, xy_reso, minx, miny):
    i = int((x - minx)/xy_reso)
    j = int((y - miny)/xy_reso)
    return i,j


def delta_motion(rads_l, rads_r, yaw):
    r = (42.5/2.0)      *10**(-3) ## metros
    b = (54/2.0)        *10**(-3) ## metros
    
    sl = rads_l*r
    sr = rads_r*r
    s = (sr+sl)/2.0
    tita = (sr-sl)/(2.0*b)
    
    delta_x = s*np.cos(yaw + tita/2.0)
    delta_y = s*np.sin(yaw + tita/2.0)

    return delta_x, delta_y

def map_shift(grid_map, x_shift, y_shift):
    # realizamos una copia para poder modificar sobre la original
    tgmap = copy.deepcopy(grid_map.data)

    for ix in range(grid_map.xw):
        for iy in range(grid_map.yw):
            nix = ix + x_shift
            niy = iy + y_shift

            if 0 <= nix < grid_map.xw and 0 <= niy < grid_map.yw:
                grid_map.data[ix + x_shift][iy + y_shift] = tgmap[ix][iy]

    return grid_map # devuelve el objeto gridmap con grid_map.data actulizada

def motion_update(grid_map, u, yaw, MOTION_STD):
    # calcula los desplazamientos d y dy
    dx , dy = delta_motion(u[0],u[1],yaw)
    
    # Suma los desplazamientos en el sistema de coordenadas inercial
    # con los desplazamientos que no se han tenido en cuenta en la actualización anterior
    grid_map.dx += dx
    grid_map.dy += dy
    
    # Transforma los desplazamientos a celdas
    x_shift = int(grid_map.dx // grid_map.xy_reso)
    y_shift = int(grid_map.dy // grid_map.xy_reso)
    
    # Traslada el mapa x_shift e y_shifty almacena la cantidad de desplazamiento que
    # que no se ha trasladado el mapa en grid_map.dx y grid_map.dy: 
    if x_shift != 0:
        grid_map.dx = (grid_map.dx % grid_map.xy_reso)
    if y_shift != 0:
        grid_map.dy = (grid_map.dy % grid_map.xy_reso)
        
    grid_map = map_shift(grid_map, x_shift, y_shift)
    
    # Aplica el filtro gaussiano
    grid_map.data = gaussian_filter(grid_map.data, sigma=MOTION_STD)

    return grid_map

def pos_estimada(bel):
    sumX = np.sum(bel.data,1).reshape(1,-1)
    sumY = np.sum(bel.data,0).reshape(1,-1)
    
    xVals = np.arange(bel.minx, bel.maxx, bel.xy_reso).reshape(1,-1)
    yVals = np.arange(bel.miny, bel.maxy, bel.xy_reso).reshape(1,-1)
    
    est_x = np.sum(sumX*xVals)
    est_y = np.sum(sumY*yVals)
#     i ,j = np.unravel_index(np.argmax(bel.data, axis=None), bel.data.shape)
    
#     est_x = xVals[0,i]
#     est_y = yVals[0,j]
    return est_x, est_y

def get_z(robot):
    Trees = ["Tree0", "Tree1", "Tree2", "Tree3"]
    z = np.array([[-1, -0.5, 1.0],
                  [-1, -1, -1],
                  [-1, 1, -1.5],
                  [-1, 1.5, 0.0]])
    dist, obj = robot.proximeters(tracked_objects=["Tree"],mode=Trees)
    
    dim = len(obj)
    for i in range(dim):
        if obj[i] == "Tree0":
            if obj[i] != 'None':
                z[0,0] = dist[i]
            else:
                z[0,0] = -1
        elif obj[i] == "Tree1": 
            if obj[i] != 'None':
                z[1,0] = dist[i]
            else:
                z[1,0] = -1
        elif  obj[i] == "Tree2":
            if obj[i] != 'None':
                z[2,0] = dist[i]
            else:
                z[2,0] = -1
        elif  obj[i] == "Tree3":
            if obj[i] != 'None':
                z[3,0] = dist[i]
            else:
                z[3,0] = -1
    return z

def calc_gaussian_observation_pdf(gmap, z, iz, ix, iy, std,sensor_range):
    x = ix * gmap.xy_reso + gmap.minx
    y = iy * gmap.xy_reso + gmap.miny
    d = math.hypot(x - z[iz, 1], y - z[iz, 2])

    if d > sensor_range or z[iz,0] == -1:
        p = 0.01
    else:
        p = 1 - norm.cdf(abs(d - z[iz, 0]*10**-3), 0.0, std)

    return p

def observation_update(gmap, z, std,sensor_range):
    # Calculamos la probabilidad de la medicion nuevos
    # de cada casilla
    dim = z.shape[0]

    for ix in range(gmap.xw):
        for iy in range(gmap.yw):
            val = 1
            for iz in range(dim):
                 val = val * calc_gaussian_observation_pdf(gmap,z,iz,ix,iy,std,sensor_range)
            gmap.data[ix,iy] *= val

    gmap = normalize_probability(gmap)
    return gmap

def updateNgetPosFB(bel,u,yaw,epuck,RANGE_STD,sensor_range,MOTION_STD):
    # Actualizacion del movimento
    bel = motion_update(bel,u,yaw,MOTION_STD)
    bel = normalize_probability(bel)
    
    # Actualizacion de la medicion
    z = get_z(epuck)
    bel = observation_update(bel,z,RANGE_STD,sensor_range)
    
    x_est, y_est = pos_estimada(bel)
    return x_est,y_est,bel
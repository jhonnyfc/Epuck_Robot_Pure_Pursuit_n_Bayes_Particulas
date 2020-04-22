import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

class particle:
    # constructor
    def __init__(self):
        # definición de características físicas del robot
        self.L = 5.4*10**-2
        self.diametro = 4.25 * 10 ** -2
        self.Range = 2.0

        # Posición del robot en el sistema de coordenadas referencial
        self.x = 0
        self.y = 0
        self.theta = 0
        
        # Medición de la partícula
        self.z = -1

        # Parámetros de ruido 
        self.motor_noise = 0.2
        self.ps_noise = 0.2

        # definición del entorno del robot
        self.mundo_MinX = -5
        self.mundo_MinY = -5
        self.mundo_MaxX = 5
        self.mundo_MaxY = 5

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.theta))

    # Método para asignar la posición del robot(en coordenadas del sistema referencial)    
    def set_estado(self, x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    # Método que devuelve la posición del robot (en coordenadas del sistema referencial)
    def get_estado(self):
        return (self.x, self.y, self.theta)
    
    # Método para establecer el rango del sensor
    def set_sensor_range(self, sensor_range):
        self.Range = sensor_range
    
    # Método para asignar paramétros del entorno
    def mundo_params(self, minx, maxx, miny, maxy):
        self.mundo_MinX = minx
        self.mundo_MinY = miny
        self.mundo_MaxX = maxx
        self.mundo_MaxY = maxy
        
    # Método para asignar a la partícula la posición de los árboles
    def set_Trees(self, TreeNames, TreePoses):
        self.z = np.zeros((len(TreeNames),3))
        for i in range(len(TreeNames)):
            self.z[i,0] = -1
            self.z[i,1:] = TreePoses[i,:]
    
    # Método que simula el movimiento de la partícula si las ruedas han variado el 
    # ángulo en rads_l, rads_r
    def motion_odometry_model(self, rads_l, rads_r):  
        x_ant = self.x
        y_ant = self.y
        theta_ant = self.theta

        Desp_r_l = rads_l * self.diametro / 2.0
        Desp_r_r = rads_r * self.diametro / 2.0

        # Añadir una cantidad de ruido a Desp_r_l y Desp_r_r
        Desp_r_l += np.random.uniform(-Desp_r_l,Desp_r_l)*0.01
        Desp_r_r += np.random.uniform(-Desp_r_r,Desp_r_r)*0.01

        Desp = (Desp_r_r + Desp_r_l) / 2.0
        Delta_theta = (Desp_r_r - Desp_r_l) / (self.L)

        self.x +=  Desp * math.cos(self.theta + Delta_theta / 2.0)
        self.y +=  Desp * math.sin(self.theta + Delta_theta / 2.0)
        self.theta += Delta_theta
        if not(self.es_posible()):
            self.x = x_ant
            self.y = y_ant
            self.theta = theta_ant

    # Método que simula la medición de la particula
    def get_z(self):
        for i in range(self.z.shape[0]):
            d = math.hypot(self.x - self.z[i, 1], self.y - self.z[i, 2])  
            if d > self.Range:
                self.z[i,0] = -1
            else:
                self.z[i,0] = d
        return self.z
    
    # Método que comprueba si el estado es posible
    def es_posible(self):
        if self.mundo_MinX + self.L/2.0 <= self.x < self.mundo_MaxX - self.L/2.0 and \
        self.mundo_MinY + self.L/2.0 <= self.y < self.mundo_MaxY - self.L/2.0:
            return True
        else:
            return False

def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        # hacer la orientacion media es dificil ya que es ciclica
        # normalizamos a partir de la primera particula para resolver el problema que 2pi = 0
        orientation += (((p[i].theta - p[0].theta + math.pi) % (2.0 * math.pi)) 
                        + p[0].theta - math.pi)
    
    return [x / len(p), y / len(p), orientation / len(p)]

def peso_particula(z_particula, z_robot):
    dim = z_particula.shape[0]
    
    p = 1;
    for i in range(dim):
        if (z_particula[i,0] == -1) or (z_robot[i,0] == -1):
            p *= 0.01
        else:
            d = np.sum((z_particula[i,0] - z_robot[i,0])**2)
            p *= np.exp(-d*10)

    return p

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
                z[0,0] = dist[i]*10**-3
            else:
                z[0,0] = -1
        elif obj[i] == "Tree1": 
            if obj[i] != 'None':
                z[1,0] = dist[i]*10**-3
            else:
                z[1,0] = -1
        elif  obj[i] == "Tree2":
            if obj[i] != 'None':
                z[2,0] = dist[i]*10**-3
            else:
                z[2,0] = -1
        elif  obj[i] == "Tree3":
            if obj[i] != 'None':
                z[3,0] = dist[i]*10**-3
            else:
                z[3,0] = -1

    return z

def generate_new_particles(old_particles, weights):
    N = len(old_particles)
    new_particles = []

    pos = np.random.choice(np.arange(N), N, replace=True,p = weights)
    for i in range(N):
        new_particles.append(copy.deepcopy(old_particles[pos[i]]))

    return new_particles

def generate_new_particles(old_particles, weights):
    N = len(old_particles)
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(weights)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        new_particles.append(copy.deepcopy(old_particles[index]))

    return new_particles

def createParticulas(sensor_range,EXTEND_AREA,gmap,posBot,yawBot,N):
    Trees = ["Tree0", "Tree1", "Tree2", "Tree3"]
    PosTrees = np.array([[-0.5, 1.0],
                          [-1, -1],
                          [1, -1.5], 
                         [1.5, 0.0]])

    # grid map param
    MINX = gmap.worldminx 
    MAXX = gmap.worldmax
    MINY = gmap.worldminy
    MAXY = gmap.worldmaxy
    
    particulas = []
    i = 0
    x,y = posBot
    while i < N:
        p = particle()
        p.mundo_params(MINX, MAXX, MINY, MAXY)
        p.set_Trees(Trees, PosTrees)
        #theta = random.uniform(-math.pi, math.pi)
        theta = yawBot
#         x = random.uniform(MINX, MAXX)
#         y = random.uniform(MINY, MAXY)

        p.set_estado(x,y,theta)
    #     p.set_estado(pos[0],pos[1],theta)
        if p.es_posible() == True:
            particulas.append(p)
            i += 1

    return particulas

def actualizacionParticulas(particulas, N, deltal, deltar,epuck):
    for j in range(N):
        particulas[j].motion_odometry_model(deltal, deltar)

    # Actualización de la medida:
    # Medición del robot epuck
    z = get_z(epuck)

    # Calcula el peso w de cada particula:
    wPar = []
    for j in range(N):
        wPar.append(peso_particula(particulas[j].get_z(),z))
    
    # Calcula la probabilidad de supervivencia de cada partícula:
    norm = np.sum(wPar)
    wPar = wPar / norm

    # Calcula el nuevo conjunto de particulas utilizando la rueda de remuestreo:
    particulas = generate_new_particles(particulas,wPar)
    
    # Posicion Actual
    est = get_position(particulas)
    x,y,yaw = est
    return x,y,yaw, particulas
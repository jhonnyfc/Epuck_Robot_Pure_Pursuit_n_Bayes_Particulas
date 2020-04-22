import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../pypot')
sys.path.append('../../enum34')
sys.path.append('../../pyserial-3.4')
import numpy as np
from simulator_interface import open_session, close_session
import math
from math import sqrt, cos, sin

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, newx, newy, newyaw, newv):    
        self.x = newx
        self.y = newy
        self.yaw = newyaw
        self.v = newv

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)
    
class Trajectory:
    def __init__(self, cx, cy, init_index):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = init_index

    def search_target_index(self, state,Lfc,k):
        ind = self.old_nearest_point_index
        distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
        
        # Bucle para saltar puntos de la trayectoria que hemos dejado atrás
        while True:
            ind = ind + 1 if (ind + 1) < len(self.cx) else ind
            distance_next_index = state.calc_distance(self.cx[ind], self.cy[ind])
            if distance_this_index < distance_next_index:
                break
            distance_this_index = distance_next_index
        self.old_nearest_point_index = ind

        L = 0.0
        
        # La distancia al siguiente punto objetivos tiene un término fijo 
        # más un termino que varia con la velocidad del robot
        Lf = k * state.v + Lfc

        # Debes completar un bucle para encontrar el indice de la trayectoria que está 
        # a una distancia Lf. Ten en cuenta el indice no puede ser mayor que
        # el número de puntos de la trayectoria.
        # Calculo del siguinte punto objetivo
        while Lf > L and (ind + 1) < len(self.cx):
            L = state.calc_distance(self.cx[ind], self.cy[ind])
            ind += 1


        return ind

def PIDControl(v_ref, v_actual,Kp):
    delta_v = Kp * (v_ref - v_actual)
    
    return delta_v

def YawControl(state, trajectory, ind,Lfc,k):
    r = 42.5/2      *10**(-3) ## metros
    b = 54/2        *10**(-3) ## metros
    L = b*2
    # Obtencion de las cooredeandas de la trajectori
    # compobando que el indice no pase el limite
    # de los nodos que hay en la lista
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    delta_t = 1

    alpha = np.arctan2(ty - state.y, tx - state.x) - state.yaw

    # Lfc: distancia minima al puto objetivo
    # k: factor proporcion para la velocidad
    Lf = k * state.v + Lfc

    delta_yaw = np.arctan2(2.0 * L * math.sin(alpha) / Lf, 1)

    return delta_yaw/delta_t

def pure_pursuit_control(state, trajectory, pind, target_velocity,Lfc,Kp,k):

    ind = trajectory.search_target_index(state,Lfc,k)
    
    # pind: es el antrior target_ind
    if pind >= ind:
        ind = pind
    
    delta_v = PIDControl(target_velocity, state.v,Kp)
    
    delta_yaw = YawControl(state, trajectory, ind,Lfc,k)

    return delta_v, delta_yaw, ind

def inversa(vx, vy, vyaw):
    r = 42.5/2*10**(-3)
    b = 54/2*10**(-3)

    kinemInver = np.array([[1/r,0,b/r],
                        [1/r,0,-b/r]])
    matVlin = np.array([[vx],
                        [vy],
                        [vyaw]])

    v_ang_dcha,v_ang_izda = kinemInver.dot(matVlin).ravel()
    
    return v_ang_izda, v_ang_dcha

def getNewVeloPPT(state, trajectory, target_ind, target_speed,Lfc,Kp,k,dtGiro):
    # Llamar a pure_pursuit_control para obtener los cambios en las velocidades 
    # y el siguiente punto objetivo
    # El control PID devuelve una cantidad que debe sumarse a la velocidad 
    # de desplazamiento actual del robot
    # El control de Yaw devuelve la velocidad de giro del robot.
    delta_v, delta_yaw, target_ind = pure_pursuit_control(state, trajectory, target_ind, target_speed,Lfc,Kp,k)
    
    # Calcular las velocidades de las ruedas con la cinematica inversa
    veloci = state.v + delta_v
    v_ang_izda, v_ang_dcha = inversa(veloci, veloci, delta_yaw/dtGiro)
    
    return v_ang_izda, v_ang_dcha, target_ind
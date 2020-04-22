from simulator_interface import open_session, close_session
import MapBuilder_Tools
from MapBuilder_Tools import MapBuilder
import sys
import Calculador_Camino_Tools
import pandas as pd
import Pure_Pursuit_Tools as ppt
import Filtro_Particulas_Tools as fpt
import matplotlib.pyplot as plt
import numpy as np
import Filtro_Bayes_Tools as fbt

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../pypot')
sys.path.append('../../enum34')
sys.path.append('../../pyserial-3.4')

def getPosRobot():
    simulator = None
    try:
        print('\nObteniendo la posicion del robot')
        simulator, epuck = open_session()

        # posicion inical del robot
        xINI,yINI,_ =  epuck.position()
        robotPosIni = [xINI,yINI]
        
        close_session(simulator)
        print('La posicion inicial del robot es:',robotPosIni[0],robotPosIni[1])
        return robotPosIni
    except:
        if (simulator != None):
            close_session(simulator)
        print("Error el la conexion con el simulador, ejecute este fragmento de nuevo")
        print('Error al obtener la posicion del robot')
        print('')
        return None

def getMapas(iterDila = 4,resolucion = 0.02):
    gmap = MapBuilder_Tools.getMapa(resolucion)
    if (gmap == None):
        return None
    else:
        MapBuilder_Tools.printMapa(gmap)
        gmap = MapBuilder_Tools.dilateMap(gmap,nIter=iterDila)
        return gmap

def verificacionColor(dato,anterior = None):
    colores = ['Rosa', 'Rojo', 'Amarillo', 'Azul']
    if (anterior == None):
        if (dato in colores):
            return dato
        else:
            print('Color erroneo, introduzca uno de lista: ')
            return verificacionColor(input('\nIntroduzca el primer color: '))
    elif (dato == anterior):
        print('El color ya a sido introducido: ')
        return verificacionColor(input('\nIntroduzca el segundo color: '),anterior)
    elif (dato in colores):
        return dato
    else:
        print('Color erroneo, introduzca uno de lista: ')
        return verificacionColor(input('\nIntroduzca el segundo color: '),anterior)

def getColors():
    print('\n***********************************************************')
    print('* En el mapa hay 4 puntos,  [Rosa, Rojo, Amarillo, Azul]  *')
    print('* El robot parte de la casilla Verde y pasara por los dos *')
    print('* puntos seleccionados calculando el camino mas corto.    *')
    print('***********************************************************\n')
    color1 = verificacionColor(input('Introduzca el primer color:'))
    color2 = verificacionColor(input('Introduzca el segundo color: '),color1)
    return color1,color2

def planificaCamino(gmap,vecinos,resolucion = 0.02):
    if (gmap == None):
        print('mapa no recivido')
        return None
    else:
        color1,color2 = getColors()
        posRobot = getPosRobot()
        if (posRobot == None):
            return None
        else:
            camino = Calculador_Camino_Tools.calculaTrajectoriaMasCorta(color1,color2,gmap,vecinos,posRobot,resolucion)
            MapBuilder_Tools.showMapAndRoute(gmap,camino)
            print('')
            return camino

def caminoSeguido(robot = [], path = [],estima = []):
    if len(robot) > 0 or len(path) > 0 or len(estima) > 0:
        fig = plt.figure(figsize=(8,8))
#         fig.show()
        fig.canvas.draw()
        plt.cla()
        if len(path) > 0:
            plt.plot(path[:,0], path[:,1], "*y", label="Trayectoria")
        if len(robot) > 0:
            plt.plot(robot[:,0], robot[:,1], ".r", label="posRobot")
        if len(estima) > 0:
            plt.plot(estima[:,0], estima[:,1], "-b", label="Estimada")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)
    
# Lfc: distancia minima al puto objetivo (m)
# Kp: constante de proporcion para alzanzar la velocidad objetivo [0,1]
# k: constante de timepo para estimar la distancia que se recorre (s)
# dt: intervalo de tiempo para el control del robot
# target_speed: velocidad maxima que alcanzara el robot (m,s)
# patk: array con el camino que sigue el robot
def simulacionTrajectoriaFP(Lfc = 0.1, Kp = 0.1, k = 0.05, dt = 0.5, dtGiro = 1, target_speed = 0.05, path = None, sensor_range = 2, EXTEND_AREA = 5.0, N = 100, gmap = None, loc = False):
    simulator = None
    robotPos = []
    estima = []
    try:
        simulator, epuck = open_session()

        # Obtencion de la posicion, oritentacion y velocidad inicial de robot
        angl, angr = epuck.odometry()
        x, y, _ = epuck.position()
        yaw  = epuck.yaw()
        v = epuck.fwd_spd

        # Creamos objeto state Con el estado inicial del ROBOT
        state = ppt.State(x,y,yaw,v)

        # Último índice de la trayectoria
        lastIndex = path.shape[0] - 1

        # Indice inicial de la trayectoria
        init_index = 0

        # Creamos el objeto trayectoria
        trajectory = ppt.Trajectory(path[:,0], path[:,1], init_index)

        # Siguient indice objetivo
        target_ind = init_index

        # Creacion de las Particulas FP
        if loc == True:
            print('Filtro de Particulas Activado')
            posBot = [x,y]
            particulas = fpt.createParticulas(sensor_range,EXTEND_AREA,gmap,posBot,yaw,N)
            estima = np.array([[x,y]])

        robotPos = np.array([[x,y]])
        print('Ejecutando Simulacion...')
        while lastIndex > target_ind:
            # Obtencion de las nuevas velocidades con PurePursuit
            v_ang_izda, v_ang_dcha, target_ind = ppt.getNewVeloPPT(state, trajectory, target_ind, target_speed,Lfc,Kp,k,dtGiro)

            # Asignar las velocidades a las ruedas:
            epuck.right_spd = v_ang_dcha
            epuck.left_spd = v_ang_izda

            epuck.wait(dt)

            if loc == True:
                # Seccion de Localizacion Mediante el Filtro de Parituclas
                # Obtenemos los ángulos que han variado las ruedas
                newangl, newangr = epuck.odometry()
                deltal = newangl - angl
                deltar = newangr - angr
                angl = newangl
                angr = newangr

                # Obtencion de la posicion mediante el filtro de partiuclas
                x,y,yaw, particulas = fpt.actualizacionParticulas(particulas, N, deltal, deltar,epuck)
                estima = np.append(estima, [[x,y]], axis = 0)
            else:
                x, y, _ = epuck.position()
                yaw  = epuck.yaw()

            v = epuck.fwd_spd
            state.update(x,y,yaw,v)


            # se guarda la ruata verdadera del robot
            if loc == True:
                xr,yr, _ = epuck.position()
                robotPos = np.append(robotPos, [[xr,yr]], axis = 0)
            else:
                robotPos = np.append(robotPos, [[x,y]], axis = 0)

        caminoSeguido(robotPos,path,estima)
        print('Simulacion terminada')
        close_session(simulator)
    except:
        if simulator != None:
            close_session(simulator)
        if len(robotPos) > 0:
            caminoSeguido(robotPos,path,estima)
        print("Error el la conexion con el simulador, ejecute este fragmento de nuevo")

# Simulacion mediante el Filtro de Bayes
def simulacionTrajectoriaFB(Lfc = 0.1, Kp = 0.1, k = 0.05, dt = 0.5, dtGiro = 1, target_speed = 0.05, path = None, sensor_range = 2, EXTEND_AREA = 5.0, gmap = None, loc = False, MOTION_STD = 0.2,RANGE_STD = 0.2):

    simulator = None
    robotPos = []
    estima = []
    try:
        simulator, epuck = open_session()

        # Obtencion de la posicion, oritentacion y velocidad inicial de robot
        angl, angr = epuck.odometry()
        x, y, _ = epuck.position()
        yaw  = epuck.yaw()
        v = epuck.fwd_spd

        # Creamos objeto state Con el estado inicial del ROBOT
        state = ppt.State(x,y,yaw,v)

        # Último índice de la trayectoria
        lastIndex = path.shape[0] - 1

        # Indice inicial de la trayectoria
        init_index = 0

        # Creamos el objeto trayectoria
        trajectory = ppt.Trajectory(path[:,0], path[:,1], init_index)

        # Siguient indice objetivo
        target_ind = init_index

        # Creacion de las Particulas FP
        if loc == True:
            print('Filtro de Bayes Activado')
            bel = fbt.init_gmap(gmap)
            estima = np.array([fbt.pos_estimada(bel)])

        robotPos = np.array([[x,y]])
        print('Ejecutando Simulacion...')
        while lastIndex > target_ind:
            # Obtencion de las nuevas velocidades con PurePursuit
            v_ang_izda, v_ang_dcha, target_ind = ppt.getNewVeloPPT(state, trajectory, target_ind, target_speed,Lfc,Kp,k,dtGiro)

            # Asignar las velocidades a las ruedas:
            epuck.right_spd = v_ang_dcha
            epuck.left_spd = v_ang_izda

            epuck.wait(dt)

            if loc == True:
                # Seccion de Localizacion Mediante el Filtro de Parituclas
                # Obtenemos los ángulos que han variado las ruedas
                newangl, newangr = epuck.odometry()
                deltal = newangl - angl
                deltar = newangr - angr
                angl = newangl
                angr = newangr

                # Obtencion de la posicion mediante el filtro de partiuclas
                x,y,bel = fbt.updateNgetPosFB(bel,[deltal,deltar],yaw,epuck,RANGE_STD,sensor_range,MOTION_STD)
                estima = np.append(estima, [[x,y]], axis = 0)
            else:
                x, y, _ = epuck.position()

            # con el filtro de bayes en 2d no se puede estimar la oritentacion del robot
            yaw  = epuck.yaw()
            v = epuck.fwd_spd
            state.update(x,y,yaw,v)


            # se guarda la ruata verdadera del robot
            if loc == True:
                xr,yr, _ = epuck.position()
                robotPos = np.append(robotPos, [[xr,yr]], axis = 0)
            else:
                robotPos = np.append(robotPos, [[x,y]], axis = 0)

        caminoSeguido(robotPos,path,estima)
        print('Simulacion terminada')
        close_session(simulator)
    except:
        if simulator != None:
            close_session(simulator)
        if len(robotPos) > 0:
            caminoSeguido(robotPos,path,estima)
        print("Error el la conexion con el simulador, ejecute este fragmento de nuevo")
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../pypot')
sys.path.append('../../enum34')
sys.path.append('../../pyserial-3.4')

from MapBuilder_Tools import MapBuilder
import MapBuilder_Tools
from simulator_interface import open_session, close_session
import numpy as np
import pandas as pd

class Node:
    def __init__(self, x, y, cost, nodoant,heuro):
        self.x = x  # indice del mapa
        self.y = y  # indice del mapa
        self.cost = cost
        self.heu = heuro
        self.nodoant = nodoant # index of previous Node

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.nodoant)
    
    def getCostmHeuro(self):
        return self.cost+self.heu

def getPath(nodoMeta,gmap):
    if nodoMeta.nodoant == None:
        return []
    else:
        lis = getPath(nodoMeta.nodoant,gmap)
        x,y = gmap.frommat2Coord(nodoMeta.x,nodoMeta.y)
        lis.append([x,y])
        return lis

#Se se helegido como funcion heuristica la distancia euclidea
def funHeuristica(newI,newJ,meta):
    return np.sqrt((newI-meta[0])**2+(newJ-meta[1])**2)

def notIsInAbiertos(nuevoNodo,abiertos):
    noEsta = True
    estaMenor = [False,None] # Booleano y pos del nodo en la lista
    for i in range(len(abiertos)):
        if nuevoNodo.x == abiertos[i].x and nuevoNodo.y == abiertos[i].y:
            noEsta = False
            if nuevoNodo.cost < abiertos[i].cost:
                estaMenor[0] = True
                estaMenor[1] = i
                
    if estaMenor[0] == True:
        del abiertos[estaMenor[1]]
        return True
    else: return noEsta

def notIsInCerrados(nuevoNodo, cerrados):
    noEsta = True
    estaMenor = [False,None] # Booleano y pos del nodo en la lista
    for i in range(len(cerrados)):
        if nuevoNodo.x == cerrados[i].x and nuevoNodo.y == cerrados[i].y:
            noEsta = False
            if nuevoNodo.cost < cerrados[i].cost:
                estaMenor[0] = True
                estaMenor[1] = i
                
    if estaMenor[0] == True:
        del cerrados[estaMenor[1]]
        return True
    else: return noEsta

def nuevosSucesores(actual,abiertos,cerrados,vecinos,gmap,matOccu,meta):
    newSuce = []
    for i in range(vecinos.shape[0]):
        i,j,peso = vecinos[i,:]
        newI, newJ = int(i + actual.x), int(j + actual.y)
        # Comprobar que el nodo este dentro del rango y
        # ver que la ueva casilla no sea una pared
        if gmap.pointinbMatJ(newI, newJ) == True and matOccu[newI,newJ] == 0:
            # Comprobar que el nodo no este en abiertos y si esta 
            # ver si es de menor peso y eliminar el enterior existente
            nuevoNodo = Node(newI, newJ, peso+actual.cost, actual,funHeuristica(newI,newJ,meta))
            if notIsInAbiertos(nuevoNodo,abiertos) == True and notIsInCerrados(nuevoNodo, cerrados) == True:
                newSuce.append(nuevoNodo)
    return newSuce

# meta: x,y de la meta, hay que pasarlo a i,j
def busquedaAstar(gmap, robotPosIni, meta, vecinos, resolu):
    # Vectores de las matrices, 0 donde no hay parede y 1 donde hay pared
    matOccu= np.array(gmap.occupacy_map)

    # Coordenadas de la posicion inicial en index de la matriz
    iRo,jRo = gmap.fromCoord2mat(robotPosIni[0],robotPosIni[1])

    # Coordenadas de la meta en index de la matriz
    iMeta,jMeta = gmap.fromCoord2mat(meta[0],meta[1])

    # Creamos el nodo inicial como la posicion acutal del robot, con coste 0 y
    # como nodo ant Nulo (None)
    nodoInicial = Node(iRo,jRo,0,None,funHeuristica(iRo,jRo,[iMeta,jMeta]))

    abiertos = []
    abiertos.append(nodoInicial)
    cerrados = []
    actual = []
    sucesores = []

    print('Calculando camino...')
    while (not abiertos) == False:
        actual = abiertos[0]
        del abiertos[0]
        cerrados.append(actual)

        if actual.x == iMeta and actual.y == jMeta:
            return actual
        else:
            sucesores = nuevosSucesores(actual,abiertos,cerrados,vecinos,gmap,matOccu,[iMeta,jMeta])
            abiertos = abiertos + sucesores # Concatenacion de listas
            abiertos = sorted(abiertos, key=lambda a: a.cost+a.heu)
    return None

def seccionMasCorta(robotPosIni,c1,c2,gmap,vecinos,pathMitad,resolucion = 0.02):
    nodoMeta1 = busquedaAstar(gmap,robotPosIni,c1,vecinos,resolucion)
    camino1 = np.array(getPath(nodoMeta1,gmap))
    
    nodoMeta2 = busquedaAstar(gmap,robotPosIni,c2,vecinos,resolucion)
    camino2 = np.array(getPath(nodoMeta2,gmap))
    print('Camino encontrado.')
    
    if nodoMeta1.cost > nodoMeta2.cost:
        x,y = gmap.frommat2Coord(nodoMeta2.x, nodoMeta2.y)
        if funHeuristica(x, y, pathMitad[-1,:].flatten()) <= resolucion:
            return np.vstack((camino2,pathMitad[::-1]))
        else:
            return np.vstack((camino2,pathMitad))
    else:
        x,y = gmap.frommat2Coord(nodoMeta1.x, nodoMeta1.y)
        if funHeuristica(x, y, pathMitad[-1,:].flatten()) <= resolucion:
            return np.vstack((camino1,pathMitad[::-1])) 
        else:
            return np.vstack((camino1,pathMitad))
        
def calculaTrajectoriaMasCorta(color1,color2,gmap,vecinos,posRobot,resolucion = 0.02):
    #las Coordenadas del robot se introducen en las coordenadas del mapa
    metaRosa = [-0.5,0.5]
    metaAmarillo = [1.5,2]
    metaRoja = [0.7,-0.65]
    metaAzul = [-0.5,-1.75]
    ruta = 'caminos_inter/pre_cac/'
    
    print('')
    print('Calculando el camino mas corto para los colores:',color1,color2+"...")
    if (color1 == 'Rosa' and color2 == 'Amarillo' or color1 == 'Amarillo' and color2 == 'Rosa'):
        pathMitad = np.array(pd.read_csv(ruta+'Rosa_Amarillo.csv'))
        return seccionMasCorta(posRobot,metaRosa,metaAmarillo,gmap,vecinos,pathMitad,resolucion)
        
    elif (color1 == 'Rosa' and color2 == 'Azul' or color1 == 'Azul' and color2 == 'Rosa'):
        pathMitad = np.array(pd.read_csv(ruta+'Rosa_Azul.csv'))
        return seccionMasCorta(posRobot,metaRosa,metaAzul,gmap,vecinos,pathMitad,resolucion)

    elif (color1 == 'Rosa' and color2 == 'Rojo' or color1 == 'Rojo' and color2 == 'Rosa'):
        pathMitad = np.array(pd.read_csv(ruta+'Rosa_Rojo.csv'))
        return seccionMasCorta(posRobot,metaRosa,metaRoja,gmap,vecinos,pathMitad,resolucion)

    elif (color1 == 'Rojo' and color2 == 'Amarillo' or color1 == 'Amarillo' and color2 == 'Rojo'):
        pathMitad = np.array(pd.read_csv(ruta+'Rojo_Amarillo.csv'))
        return seccionMasCorta(posRobot,metaRoja,metaAmarillo,gmap,vecinos,pathMitad,resolucion)

    elif (color1 == 'Rojo' and color2 == 'Azul' or color1 == 'Azul' and color2 == 'Rojo'):
        pathMitad = np.array(pd.read_csv(ruta+'Rojo_Azul.csv'))
        return seccionMasCorta(posRobot,metaRoja,metaAzul,gmap,vecinos,pathMitad,resolucion)

    elif (color1 == 'Amarillo' and color2 == 'Azul' or color1 == 'Azul' and color2 == 'Amarillo'):
        pathMitad = np.array(pd.read_csv(ruta+'Amarillo_Azul.csv'))
        return seccionMasCorta(posRobot,metaAmarillo,metaAzul,gmap,vecinos,pathMitad,resolucion)

def calculaCamino(gmap,robotPosIni,meta,vecinos,resolucion = 0.02):
    print('')
    print('Coordenadas en map')
    print('La poscion inical es:',robotPosIni[0],robotPosIni[1],)
    print('La posicion de la meta es:',meta[0],meta[1])
    print('')
    print('Cooredenadas en mat')
    print('La poscion inical es:',gmap.fromCoord2mat(robotPosIni[0],robotPosIni[1]))
    print('La posicion de la meta es:',gmap.fromCoord2mat(meta[0],meta[1]))
    print('')

    nodoMeta = busquedaAstar(gmap,robotPosIni,meta,vecinos,resolucion)
    if nodoMeta != None:
        print('Se ha llegado al nodo meta\n')
        return np.array(getPath(nodoMeta,gmap))
    else:
        print('No se a encontrado camino\n')
        return nodoMeta

def guardaRuta(nombre,camino):
    camino = pd.DataFrame(camino,columns=['x','y'])
    camino.to_csv('caminos_inter/'+nombre+'.csv',index = False)
    print('CSV guardado correctamente.')
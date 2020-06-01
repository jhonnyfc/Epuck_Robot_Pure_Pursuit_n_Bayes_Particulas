# Epuck_Robot_Pure_Pursuit_n_Bayes_Particulas

## Busqueda & Seguimiento
## Epuck

### Jhonny Fabricio

### UPNA, 22 abr 2020


## Indice

Objetivo.................. 3
Explicación código...... 3
Conclusiones............. 4

Respositorio GuitHub:
https://github.com/jhonnyfc/Epuck_Robot_Pure_Pursuit_n_Bayes_Particulas


## Objetivo

Mediante el filtro de Bayes, el Filtro de Partículas y Pure Pursuit se ha construido un
sistema que permite guiar al robot por un camino calculado mediante el algoritmo
A*.

Para que el camino que busca el algoritmo A* sea ademas de corto seguro se ha
dilatado el mapa con 6 iteraciones.

## Código, Secciones

El código que ejecuta el sistema esta en el notebook llamado Practica_final.ipynb,
este llama a las librerías necesarias.

Controlador.py :
En este archivo se ha hecho la reunión de los algoritmos necesarios para la
construcción del camino y para la navegación.

Calclador_Camino_Tools.py:
En este archivo se encuentra el algoritmo A* junto a las funciones necesarias
para su ejecución.

Filtro_Bayes_Tools.py, MapBuilder_Tools.py, Pure_Pursuit_Tools.py,
Filtro_Particulas_Tools.py:
En estos archivos se ha repartido el resto de código correspondiente a lo que su
titulo hace referencia. Esto se ha hecho para que haya mas facilidad a la hora de
modificar e interpretar el código.


## Conclusiones

Si se utiliza el pure pursuit con la posición real del robot y con una velocidad baja se
consiguen completar todos los camino. Si sen intenta subir la velocidad el robot se
choca en la sección del punto rojo, debido que hay un giro muy cerrado. En este caso
la configuración optima para realizar todos los caminos se destaca en el notebook.

Si se utiliza el filtro de Partículas para que se haga una buena aproximación hay que
reducir drásticamente la velocidad dado que hay que hacer mas cálculos dependiendo
del número de partículas. Se han probado distintas configuraciones pero no se ha
encontrado la ideal.

En la ejecución del filtro de Bayes para la localización no se ha conseguido localizar
bien el robot, dado que parte de una esquina y el robot solo alcanza a medir la
distancia a un árbol. Se han probado distintas configuraciones pero no se ha
encontrado la ideal.



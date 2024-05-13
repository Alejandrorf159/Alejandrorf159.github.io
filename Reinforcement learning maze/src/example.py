
"""example.py: Use a maze and solve it with classic wall-following
   algorithm.

   The RL versions should be faster
"""

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import time
import math
import random
import dispatcher
import env
import numpy as np


#funcion de recompensa

def reward(s,sdesp,a,olda):
    if s < sdesp:
        #recompensa = 0.7
        recompensa = 0.7
    if s > sdesp:
        #recompensa = 0.4
        recompensa = 0.7
    if olda == 0 and a == 2:
        recompensa = 0
    if olda == 1 and a == 3:
        recompensa = 0
    if olda == 2 and a == 0:
        recompensa = 0
    if olda == 3 and a == 1:
        recompensa = 0
    if s == sdesp:
        recompensa = -1

    return (recompensa)


def rewardsen(sensor,a):
    recompensa = [0.7,0.7,0.7,0.7]
    if a == 0:
        if sensor[0] != -1:
            recompensa[3] = -1
        if sensor[1] != -1:
            recompensa[0] = -1
        if sensor[2] != -1:
            recompensa[1] = -1
        if sensor[3] != -1:
            recompensa[2] = -1
    if a == 1:
        if sensor[0] != -1:
            recompensa[0] = -1
        if sensor[1] != -1:
            recompensa[1] = -1
        if sensor[2] != -1:
            recompensa[2] = -1
        if sensor[3] != -1:
            recompensa[3] = -1
    if a == 2:
        if sensor[0] != -1:
            recompensa[1] = -1
        if sensor[1] != -1:
            recompensa[2] = -1
        if sensor[2] != -1:
            recompensa[3] = -1
        if sensor[3] != -1:
            recompensa[0] = -1
    if a == 3:
        if sensor[0] != -1:
            recompensa[2] = -1
        if sensor[1] != -1:
            recompensa[3] = -1
        if sensor[2] != -1:
            recompensa[0] = -1
        if sensor[3] != -1:
            recompensa[1] = -1

    return (recompensa)


class ExampleStepper:
    """Simple stepper performing a random walk"""

    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.reset()
        self.init()
        self.lastRenderingTime = time.perf_counter()
        
    def reset(self):
        """Account for posible serialization of new envirnoments"""
        print("ExampleStepper.reset() called")

        # Do this to get a valid reference to the environment in use
        self.env = self.dispatch.env()

        # You can for instance get the number of cells in the maze with:
        self.xCells = self.env.maze.nx
        self.yCells = self.env.maze.ny

        # Or also get the size of a cell
        self.cellSizeX = self.env.maze.cellSizeX
        self.cellSizeY = self.env.maze.cellSizeY

        # Here an arbitrary step size for the "advance" action
        step = self.env.maze.cellSizeX*1.2

        # Sample 50/50 a rotation or a translation
        self.actions = [self.env.agent.right,
                        self.env.agent.left,


                        lambda: self.env.agent.advance(step),
                        lambda: self.env.agent.advance(step)]
        self.acciones = [self.env.agent.advance, self.env.agent.advance,
                         self.env.agent.left, self.env.agent.right]

        #variables globales
        self.cont=0
        self.cont2=0
        self.cont3=0
        self.i=0
        self.olddir = 0
        self.direc = [0,1,2,3]
        self.p = 0
        self.numram = [0,1,2]
        self.e = 1
        self.olddireward = 0
        self.gamma = 0.9
        self.alpha = 0.5
        
    def init(self):
        """Reset and setup the stepper"""

        print("ExampleStepper.init() called")

        # Tell the environment to place the agent at the beginning
        self.env.reset()

        # Ensure the agent's position is properly displayed
        self.dispatch.render()
        self.lastRenderingTime = time.perf_counter()
#creacion de matrices

        self.numCells = self.env.maze.nx * self.env.maze.ny
        self.numStates = self.env.maze.nx * self.env.maze.ny * 4
        #print(self.numStates)

        #Tabla de Q y buffer de tabla de Q
        
        self.PA = np.zeros((1,self.numCells,4))/self.numStates #Prob to go forward
        self.Qtable = np.zeros((1,self.numCells,4))/self.numStates #Q table
        self.Q = np.zeros((1,self.numCells,4))/self.numStates #Q table
        self.Proba = np.zeros((1,self.numCells,self.numCells))/self.numStates #Prob
        self.recompe = np.zeros((1,self.numCells,4))
        #print(self.PA)
        self.TQ = np.zeros((1,self.numCells,4))
        self.CPA = np.zeros((1,self.numCells,4))
        self.iterp = np.zeros((1,self.numCells,self.numCells))
        self.Valor = np.zeros((1,self.numCells,4))

        self.pasx = np.zeros(1)
        self.pasy = np.zeros(1)
        self.pasdir = np.zeros(1)

        
#fin matrices

    cardinalPoints = ['E', 'S', 'W', 'N', 'E']

    def step(self, iteration):
        """Perform one simulation step"""
    
##        Etapa de exploracion
        itmax=900000000
        regy=math.floor(self.pasy[self.cont])
        regx=math.floor(self.pasx[self.cont])
        regdir=math.floor(self.pasdir[self.cont])
        
        if self.cont < itmax+1:        
            
            direccionact = random.choice(self.direc)

            posicionant = self.xCells*regy+regx
            
            prob = random.choice(self.numram)
            #probabilidad mas alta para que el robot avance y evite mas los giros
            if direccionact != self.olddir:
                if 1 == prob:
                    direccionact = self.olddir

            #Criterio para usar valores aletorios y lo aprendido
            if self.e <= (random.randrange(10)/10):
                print("Usar funcion de valor")
                direccionact = np.argmax(self.Valor[0][self.xCells*regy+regx])


            self.p = self.olddir
            #Seleccion de direccion a la que quiere ir dependiendo de la direccion anterior
            if self.p == 0:
                if direccionact == 1:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)


                    
                if direccionact == 2:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                    
                if direccionact == 3:
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)


            if self.p == 1:
                if direccionact == 0:
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                
                if direccionact == 2:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                    
                if direccionact == 3:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)


            if self.p == 2:
                if direccionact == 0:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                  
                if direccionact == 1:
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                   
                if direccionact == 3:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                   


            if self.p == 3:
                if direccionact == 0:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                   
                if direccionact == 1:
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                   
                if direccionact == 2:
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)                   


            #Despues de generar los giros hace un avance para llegar a ese nuevo estado
            self.env.tryAction(self.env.agent, self.env.agent.advance)
            #brinco al centro
            self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
            if self.env.agent.state.posY==0:
                self.env.agent.state.posY=5
            if self.env.agent.state.posX==0:
                self.env.agent.state.posX=5
            self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)



            self.dispatch.render()

        
        # How to extract the agent's position and orientation
        apx = self.env.agent.state.posX
        apy = self.env.agent.state.posY
        aa = self.env.agent.state.angle

        # How to determine which cell the agent is in
        cx = math.floor(apx/self.env.maze.cellSizeX)
        cy = math.floor(apy/self.env.maze.cellSizeY)

        
        # Which direction the agent is facing?
        dir = math.floor((aa+45)/90)
        # Only at most 15fps:
        # - Print information about current position 
        # - Draw the agent's position and (if activated) the sensors
        if (time.perf_counter() - self.lastRenderingTime) > 1/15:
            print("ExampleStepper.step(", iteration, ")")
            self.dispatch.render()
            self.lastRenderingTime = time.perf_counter()
#inicio agregar valores a las matrices


        if dir==4:
            dir=0


        print(rewardsen(self.env.agent.observations,dir))
        
        self.Proba[0][self.xCells*regy+regx][self.xCells*cy+cx]=self.Proba[0][self.xCells*regy+regx][self.xCells*cy+cx]+1


        self.iterp[0] = self.Proba[0]
        total=np.sum(self.Proba[0], axis=1)
        total=total+0.0000000000000000000001
        total=total.reshape(self.numCells,1)
        self.iterp[0] = self.iterp[0]/total


        recompensa = reward(posicionant,self.xCells*cy+cx,dir,self.olddireward)
        posimax = self.Qtable [0][self.xCells*regy+regx] [np.argmax(self.Qtable[0][self.xCells*cy+cx])]


        # Arreglo en las recompensas para cuando llega a la meta, llega al cuadro de inicio y cuando pasa por
        #una posicion que ya habia estado antes.
        if self.env.finished():
            recompensa = 999

        if self.xCells*cy+cx == 0:
            recompensa = -1

        if self.CPA[0][self.xCells*cy+cx][direccionact] > 1:
            recompensa = -0.1

        #en caso final de pasar 2 veces por la meta!!!! trabajar
        #if self.cont2 == 2:
            

        
        target = recompensa + self.gamma * posimax - self.Qtable[0][self.xCells*regy+regx][direccionact]
        
        self.Qtable[0][self.xCells*regy+regx][direccionact] += self.alpha * target


        #ecuacion de Bellman
        valmax = self.Valor [0][self.xCells*regy+regx] [np.argmax(self.Valor[0][self.xCells*cy+cx])]  
        promax = self.iterp [0][self.xCells*regy+regx] [np.argmax(self.iterp[0][self.xCells*cy+cx])]

        self.recompe[0][self.xCells*regy+regx][direccionact] += recompensa
        recom = self.recompe[0][self.xCells*regy+regx][direccionact]
        
        self.Valor[0][self.xCells*regy+regx][direccionact] = recom + valmax * promax * self.gamma
        

        self.CPA[0][self.xCells*regy+regx][direccionact]=self.CPA[0][self.xCells*regy+regx][direccionact]+0.5
        

        #ubicar la direccion anterior para la recompensa
        if reward(posicionant,self.xCells*cy+cx,dir,self.olddir) != -1:
            self.olddireward = dir

        #Saber la direccion anterior
        if dir != self.olddir:
            self.olddir = dir   


        #termina de hacer movimiento(etapa de analisis y resultados)


        if self.env.finished():
            print("Hacer reset TOTAL")
            self.Q[0] = self.Qtable[0]
            self.env.reset()
            self.e = self.e - 0.1
            
            print("Numero de pasos para completar el laberinto: ",self.cont3)
            self.cont3 = 0
            self.cont2 += 1
            self.CPA[0] = 0
            self.Proba[0] = 1
            print(self.Valor[0])
            
        print("Se a completado ",self.cont2," de 20 veces")

        self.cont3 += 1
        
        if (self.cont % 50) == 0:
            print("Actualizacion tabla")
            self.Q[0] = self.Qtable[0]


        #avance final, sujeto a cambios dependiendo del laberinto que se va a utilizar




        #Arreglar contadores y posiciones para el siguiente step            
        self.cont=self.cont+1
        self.pasx=np.append(self.pasx,cx)
        self.pasy=np.append(self.pasy,cy)
        self.pasdir=np.append(self.pasdir,dir)
        if self.cont==itmax+1:
            self.env.agent.setPos(5*round(4/5),5*round(4/5))



#fin agrego valores a matrices
        
        # Check which is the finishing cell
        finX, finY = self.env.maze.endX, self.env.maze.endY

        # Check if the agent as reached the finish cell
        if self.env.finished():
            print("GOAL REACHED!")
#            self.dispatch.pause()
#            self.dispatch.restart()  # We could restart instead!
        


# #######################
# # Main control center #
# #######################

# This object centralizes everything
theDispatcher = dispatcher.Dispatcher()

# Provide a new environment (maze + agent)
theDispatcher.setEnvironment(env.Environment(12, 8, 10))

# Provide also the simulation stepper, which needs access to the
# agent and maze in the dispatcher.
theDispatcher.setStepper(ExampleStepper(theDispatcher))

# Start the GUI and run it until quit is selected
# (Remember Ctrl+\ forces python to quit, in case it is necessary)
theDispatcher.run()


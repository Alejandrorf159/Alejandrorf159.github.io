
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


from keras.layers import Dense,Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam

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

def build_dqn(lr, n_actions, input_dims, fully_dims_1, fully_dims_2):
    
    # Creo la red
    model = Sequential([
            # capa fully connected. la coma me permite pasar o un batch o una 
            # memoria singular (para aprender o elegir una accion) 
            Dense(fully_dims_1, input_shape=(input_dims, )), 
            
            # Para la capa de activación voy a usar un relu, luego pienso si 
            # cambiarlo
            Activation('relu'),
            
            # Por el momento voy a tener dos capas escondidas
            Dense(fully_dims_2), 
            Activation('relu'),
            
            Dense(n_actions)
        ])
    
    # Ahora compilo el modelo. Por el momento voy a usar Adam como optimizer
    # y con funcion de perdida mean square error
    model.compile(optimizer=Adam(lr=lr), loss = 'mse')
    
    return model
        
    

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
        self.eval_dpq = build_dqn(self.alpha, self.direc, 8, 256, 256)
        
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
            #print(regx)
            #print(regy)
            #print(self.env.agent.state.posX)
            #print(self.env.agent.state.posY)
            #print("ExampleStepper.step(", iteration, ")")



            #self.env.inCell(hor,ver,self.env.agent)
            #self.env.tryAction(self.env.agent, aleatorio)
            #brinco al centro
            #self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
            #if self.env.agent.state.posY==0:
            #    self.env.agent.state.posY=5
            #if self.env.agent.state.posX==0:
            #    self.env.agent.state.posX=5
            #self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
            #print(self.env.agent.state.posX)
            #print(self.env.agent.state.posY)

            
            
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
                #!print (self.Qtable[0])
                #direccionact = np.argmax(self.Q[0][self.xCells*regy+regx])
                direccionact = np.argmax(self.Valor[0][self.xCells*regy+regx])
                #direccionact = np.argmax(self.TQ[0][self.xCells*regy+regx])


            #variar valor de e y alpha
            #self.conte = self.e - 0.0001
            #self.alpha = self.alpha * 0.99

            self.p = self.olddir
            #Seleccion de direccion a la que quiere ir dependiendo de la direccion anterior
            if self.p == 0:
                if direccionact == 1:
                    #print ("este a sur")
                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)


                    
                if direccionact == 2:
                    #print ("este a west")
                    #self.env.inCell(hor,ver,self.env.agent)
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
                    #print ("este a norte")
                    #self.env.inCell(hor,ver,self.env.agent)
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
                    #print ("sur a este")
                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                
                if direccionact == 2:
                    #print ("sur a west")
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
                    #print ("sur a norte")
                    #self.env.inCell(hor,ver,self.env.agent)
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


            if self.p == 2:
                if direccionact == 0:
                    #print ("west a este")
                    #self.env.inCell(hor,ver,self.env.agent)
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
                  
                if direccionact == 1:
                    #print ("west a sur")
                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)

                   
                if direccionact == 3:
                    #print ("west a norte")
                    #self.env.inCell(hor,ver,self.env.agent)
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
                    #print ("norte a este")
                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.right)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)
                   
                if direccionact == 1:
                    #print ("norte a sur")
                    #self.env.inCell(hor,ver,self.env.agent)
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
                   
                if direccionact == 2:
                    #print ("norte a west")
                    #self.env.inCell(hor,ver,self.env.agent)
                    self.env.tryAction(self.env.agent, self.env.agent.left)
                    #brinco al centro
                    self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
                    if self.env.agent.state.posY==0:
                        self.env.agent.state.posY=5
                    if self.env.agent.state.posX==0:
                        self.env.agent.state.posX=5
                    self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)                   


            #Despues de generar los giros hace un avance para llegar a ese nuevo estado
            #self.env.inCell(hor,ver,self.env.agent)
            self.env.tryAction(self.env.agent, self.env.agent.advance)
            #brinco al centro
            self.env.agent.setPos(5*round(self.env.agent.state.posX/5),5*round(self.env.agent.state.posY/5))
            if self.env.agent.state.posY==0:
                self.env.agent.state.posY=5
            if self.env.agent.state.posX==0:
                self.env.agent.state.posX=5
            self.env.agent.state.angle = 90*round(self.env.agent.state.angle/90)



            
            #print("               Agent after last step: ", self.env.agent)
    ##        print(math.floor(apx/self.env.maze.cellSizeX))
            #print(self.env.agent.AgentState.posX)
           
    ##      random.choice(self.actions)
            self.dispatch.render()


#que hace        self.env.tryAction(self.env.agent, random.choice(self.actions))

        
        # How to extract the agent's position and orientation
        apx = self.env.agent.state.posX
        apy = self.env.agent.state.posY
        aa = self.env.agent.state.angle

        # How to determine which cell the agent is in
        cx = math.floor(apx/self.env.maze.cellSizeX)
        cy = math.floor(apy/self.env.maze.cellSizeY)

#imprimir posicion X y Y

        #print(cx)
        #print(cy)

        
#fin impresion
        
        # Which direction the agent is facing?
        dir = math.floor((aa+45)/90)
        # Only at most 15fps:
        # - Print information about current position 
        # - Draw the agent's position and (if activated) the sensors
        if (time.perf_counter() - self.lastRenderingTime) > 1/15:
            print("ExampleStepper.step(", iteration, ")")
            #print("  Agent continous state: ", self.env.agent)
            #print("  Agent discrete state: ({},{})@{}".\
                  #format(cx,
                         #cy,
                         #self.cardinalPoints[dir]))
            self.dispatch.render()
            self.lastRenderingTime = time.perf_counter()
#inicio agregar valores a las matrices
        #print (dir)


        if dir==4:
            dir=0


        self.Proba[0][self.xCells*regy+regx][self.xCells*cy+cx]=self.Proba[0][self.xCells*regy+regx][self.xCells*cy+cx]+1


        self.iterp[0] = self.Proba[0]
        total=np.sum(self.Proba[0], axis=1)
        total=total+0.0000000000000000000001
        total=total.reshape(self.numCells,1)
        self.iterp[0] = self.iterp[0]/total

        
        #print ("posicion antes: ",posicionant)
        #print ("posicion nueva: ",self.xCells*cy+cx)
        #print ("direccion que señala: ",dir)

        #if reward(posicionant,self.xCells*cy+cx,dir,self.olddir) != -1:
        #    self.olddireward = dir


        #print ("antigua posicion: ", self.olddireward)
        #print ("La recompensa es de: ",reward(posicionant,self.xCells*cy+cx,dir,self.olddireward))


        #ecuacion de bellman con un extra en recompensa en la meta

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
        #print(recom)

        #print("valor de recompensa: ",recom)
        #print("valor de valmax: ",valmax)
        #print("valor de promax: ",promax)
        
        self.Valor[0][self.xCells*regy+regx][direccionact] = recom + valmax * promax * self.gamma
        
        #print(self.Valor[0])
        #print("Valor de argumento maximo: ",np.argmax(self.CPA[0][self.xCells*regy+regx]))
        #print (self.Qtable[0])


        #tra = recom + self.gamma * self.TQ[0][self.xCells*regy+regx][np.argmax(self.TQ[0][self.xCells*cy+cx])]_______________________
        

        self.CPA[0][self.xCells*regy+regx][direccionact]=self.CPA[0][self.xCells*regy+regx][direccionact]+0.5
        

        #ubicar la direccion anterior para la recompensa
        if reward(posicionant,self.xCells*cy+cx,dir,self.olddir) != -1:
            self.olddireward = dir

        #Saber la direccion anterior
        if dir != self.olddir:
            self.olddir = dir   


        #termina de hacer movimiento(etapa de analisis y resultados)

        #if self.cont == 10000:
        #    print("Hacer reset TOTAL y usar tabla")
        #    self.env.reset()
        #    self.e = -1

        #if (self.cont % 80) == 0 and self.cont > 5000:
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
            #self.CPA[0] = self.PA[0]
            #print(self.Proba[0])
            
            #self.iterp[0] = self.Proba[0]
            #total=np.sum(self.Proba[0], axis=1)
            #total=total+0.0000000000000000000001
            #total=total.reshape(self.numCells,1)
            #self.iterp[0] = self.iterp[0]/total
            
            #print(self.iterp[0])
            print(self.Valor[0])
            
        print("Se a completado ",self.cont2," de 20 veces")

        self.cont3 += 1
        
        if (self.cont % 50) == 0:
            print("Actualizacion tabla")
            self.Q[0] = self.Qtable[0]


        #avance final, sujeto a cambios dependiendo del laberinto que se va a utilizar
        #if self.cont > 10000:
        #    self.cont3 += 1
        #    print("Numero total de pasos despues de entrenamiento: ", self.cont3)




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


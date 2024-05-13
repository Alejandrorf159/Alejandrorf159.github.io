
"""env.py: Environment configuration for the maze simulation"""

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import math
import random
import maze
import agent


class Environment:
    """"Environment class

        The environment holds a maze and one or more agents
    """

    def __init__(self, nx, ny, cellSizeX, cellSizeY=None):
        if cellSizeY is None:
            cellSizeY = cellSizeX
        self.maze = maze.Maze(nx, ny, cellSizeX, cellSizeY)
        self.agent = agent.Agent()
        self.reset()

    def reset(self):
        self.agent.state.reset()
        self.agent.setPos((self.maze.startX+0.5)*self.maze.cellSizeX,
                          (self.maze.startY+0.5)*self.maze.cellSizeY)
        if self.agent.sensors and self.agent.maxDistance > self.agent.radius:
            self.agent.observations = self.observe(self.agent)
        

    def tryAction(self, anAgent, action, arg=None):
        """Executes the action on the agent, but verifies that the achieved
           result is compatible with the environment.  The environment
           fixes all invalid information.

           "action" must be a method of the agent class (i.e. right, left or
           advance) which might receive an optional argument "arg".
        """

        if arg is None:
            pre = action()
        else:
            pre = action(arg)

        anAgent.setState(self.validate(anAgent, pre))
        if anAgent.sensors and anAgent.maxDistance > anAgent.radius:
            anAgent.observations = self.observe(anAgent)
            # print("[DBG] Observations: ", anAgent.observations)

    def validate(self, anAgent, newState):
        """Restrict the newState to a valid environmental compatible
           state.

           This method just returned the possible state, but does not
           affect the agent.
        """

        res = agent.AgentState(newState)

        # Is it rotating?
        if (not anAgent.state.sameAngle(newState)):
            # For now, let's asume no major obstacles for rotation,
            # but some noise
            res.angle += random.normalvariate(0, anAgent.rotationNoise)

        # Is it moving?
        if (not anAgent.state.samePos(newState)):
            length = anAgent.state.distance(newState)
            nx = random.normalvariate(0, length*anAgent.translationNoiseFactor)
            ny = random.normalvariate(0, length*anAgent.translationNoiseFactor)

            dx, dy = self.maze.tryMovement(anAgent.state.posX,
                                           anAgent.state.posY,
                                           newState.posX+nx,
                                           newState.posY+ny,
                                           anAgent.radius)

            res.posX, res.posY = dx, dy

        return res

    def observe(self, anAgent):
        """Observe the environment.  The agent sensor array must be properly
           configured.

           The return observation array correspons to one distance
           measurment for each angle (in degrees) specified in the agent's
           sensorArray.  If the measurements exceed the sensor's reach
           then -1 is returned
        """

        if anAgent.sensorArray is None or not anAgent.sensorArray:
            return []

        ret = [0]*len(anAgent.sensorArray)
        i = 0
        for sangle in anAgent.sensorArray:
            d = self.maze.observe(anAgent.state.posX,
                                  anAgent.state.posY,
                                  anAgent.state.angle + sangle,
                                  anAgent.maxDistance)
            ret[i] = d
            i += 1

        return ret

    def inCell(self, cx, cy, anAgent):
        """Check if the provided agent is touching the center of the given
           cell
        """

        x, y = anAgent.state.posX, anAgent.state.posY
        r2 = anAgent.radius*anAgent.radius

        ccx, ccy = (cx+0.5)*self.maze.cellSizeX, (cy+0.5)*self.maze.cellSizeY
        dx, dy = ccx-x, ccy-y

        d2 = dx*dx+dy*dy

        # if d2 <= r2:
        #     d = math.sqrt(d2)
        #     print("[DBG] d=", d, " < ", anAgent.radius)

        return (d2 <= r2)

    def finished(self):
        """ Check if the agent is touching the center of the finish cell """
        return self.inCell(self.maze.endX, self.maze.endY, self.agent)

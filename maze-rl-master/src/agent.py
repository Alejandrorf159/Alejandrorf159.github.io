
"""agent.py: Virtual robot agent, who interacts with the environment"""

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import math


class AgentState:
    """AgentState holds the position and orientation of an agent and
       nothing else.

       The class provides a deep copy method, to avoid copying references
       when just the content is intended to be transfered.
    """
    def __init__(self, other=None):
        if other is None:
            self.reset()
        else:
            self.copy(other)

    def reset(self):
        self.angle = 0
        self.posX = 0
        self.posY = 0

    def copy(self, other):
        self.angle = other.angle
        self.posX = other.posX
        self.posY = other.posY

    def samePos(self, other):
        return (self.posX == other.posX) and (self.posY == other.posY)

    def distance(self, other):
        dx = self.posX - other.posX
        dy = self.posY - other.posY

        return math.sqrt(dx*dx + dy*dy)

    def sameAngle(self, other):
        return self.angle == other.angle

    def __str__(self):
        return '('+str(self.posX)+','+str(self.posY)+')@'+str(self.angle)+'°'


class Agent:
    """"Virtual mobile robot as agent

        The agent holds its state of position and orientation.

        An additional set of observations can be used in more advanced setups.

        The position is used in the same coordinate system as the
        maze.  The orientation is used in mathematical angle
        conventions, i.e. a horizontal direction to the right means
        0°, and positive angles are given in degrees,
        counter-clockwise.

        It has also some properties such as radius, which is used to
        compute collisions with the walls or other agents.

        It can execute some actions.

        At a simple level it rotates fixed but noisy angles, and make
        steps towards that direction (also noisy).

        At a complex more complex level it can rotate arbitrary angles
        and advance larger distances.

        The action methods (left, right, advance) do not perform anything.
        They simply provide the desired new state and the environment will
        use the information to compute a new state compatible with the
        information it holds.  Therefore, particularly advancing will
        change depending if a wall or similar is met

    """

    def __init__(self):
        self.state = AgentState()

        self.radius = 2.5
        self.color = [128, 64, 0]
        self.sensors = True  # Use sensors or not
        
        # Angles (in degrees) at which observations will be made
        # w.r.t the agent's orientation as stated in state.angle
        self.sensorArray = [-90,0,90,180]

        # Distances at the given sensorArray angles (or -1 if further
        # than maxDistance)
        self.observations = []
        
        # Maximum measurable distance of the sensors
        # (distances larger than this value are set to -1)
        self.maxDistance = 13
        
        self.rotationNoise = 1
        self.translationNoiseFactor = 0.01

    def right(self, angle=90):
        ret = AgentState(self.state)
        ret.angle += angle
        if ret.angle >= 360:
            ret.angle -= 360

        # print("[DBG]   right(", angle, ") -> ", ret.angle)

        return ret

    def left(self, angle=90):
        ret = AgentState(self.state)
        ret.angle -= angle
        if ret.angle < 0:
            ret.angle += 360

        # print("[DBG]   left(", angle, ") -> ", ret.angle)

        return ret

    def advance(self, stepSize=10.0):
        ret = AgentState(self.state)

        c = math.cos(math.radians(self.state.angle))
        s = math.sin(math.radians(self.state.angle))

        ret.posX += stepSize*c
        ret.posY += stepSize*s

        # print("[DBG]   advance(", stepSize, ") -> (",
        #       ret.posX, ",", ret.posY, ")")

        return ret

    def setState(self, state):
        self.state.copy(state)

    def setPos(self, x, y):
        self.state.posX, self.state.posY = x, y

    def __str__(self):
        return self.state.__str__()

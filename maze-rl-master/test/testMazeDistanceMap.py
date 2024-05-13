import os
import sys

import numpy as np
from PIL import Image

srcdir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "src/")
sys.path.append(srcdir)

import maze

m = maze.Maze(64, 64, 20)
m.build()
p = m.distanceMap
p = p * (255/np.max(p))
img = Image.fromarray(p)
img.show()

"""maze.py: Maze """

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import math
import random
import numpy as np
from scipy import ndimage
import img


# Sign function not available in Python math
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1

    return 0


def clipU(x):
    return min(1, max(0, x))


class Cell:
    """A cell in the maze.

       A maze "Cell" is a point in the grid which may be surrounded by
       walls to the north, east, south or west.

       Based partially on the code at https://scipython.com/blog/making-a-maze/
       by Christian Hill, April 2017.
    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def hasAllWalls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def deleteWall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self,
                 nx, ny,
                 cellSizeX=15, cellSizeY=15,
                 ix=0, iy=0,
                 cycles=0):
        """Initialize the maze grid.

           The maze consists of nx x ny cells and will be constructed
           starting at the cell indexed at (ix, iy).

        """

        # Number of cells in x and y directions
        self.nx, self.ny = nx, ny

        # Unused
        self.ix, self.iy = ix, iy

        # Size of cells (in pixels)
        self.cellSizeX = cellSizeX
        self.cellSizeY = cellSizeY

        # Initial cell, where the agent starts
        self.startX, self.startY = 0, 0

        # Finish cell, where the agent ends
        self.endX, self.endY = self.nx-1, self.ny-1

        # 
        self.cycles = cycles

        self.mazeMap = [[Cell(x, y) for x in range(self.nx)]
                        for y in range(self.ny)]

        self.build(0)  # Ensure valid maze at all times with same seed

    def cellAt(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.mazeMap[y][x]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        mazeRows = ['-' * self.nx*2]
        for y in range(self.ny):
            mazeRow = ['|']
            for x in range(self.nx):
                if self.mazeMap[y][x].walls['E']:
                    mazeRow.append(' |')
                else:
                    mazeRow.append('  ')
            mazeRows.append(''.join(mazeRow))
            mazeRow = ['|']
            for x in range(self.nx):
                if self.mazeMap[y][x].walls['S']:
                    mazeRow.append('-+')
                else:
                    mazeRow.append(' +')
            mazeRows.append(''.join(mazeRow))
        return '\n'.join(mazeRows)

    def writeSVG(self, filename):
        """Write an SVG image of the maze to filename."""

        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = self.ny*self.cellSizeY + 1
        width = self.nx*self.cellSizeX + 1

        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = self.cellSizeY, self.cellSizeX

        def writeWall(f, x1, y1, x2, y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(x1, y1, x2, y2), file=f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width+2*padding, height+2*padding,
                          -padding, -padding,
                          width+2*padding, height+2*padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 2;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for y in range(self.ny):
                for x in range(self.nx):
                    if self.cellAt(x, y).walls['S']:
                        x1, y1, x2, y2 = x*scx, (y+1)*scy, (x+1)*scx, (y+1)*scy
                        writeWall(f, x1, y1, x2, y2)
                    if self.cellAt(x, y).walls['E']:
                        x1, y1, x2, y2 = (x+1)*scx, y*scy, (x+1)*scx, (y+1)*scy
                        writeWall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width),
                  file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height),
                  file=f)
            print('</svg>', file=f)

    def whichCellHas(self, posX, posY):
        cx = math.floor(posX / self.cellSizeX)
        cy = math.floor(posY / self.cellSizeY)
        return self.cellAt(cx, cy)

    def wallInter(self, fx, fy, tx, ty, r):
        """Given a segment parameterization on t in [0,1] from (fx,fy) to
           (tx,ty) find all values of t where the line segment
           intersects with potential walls
        """

        cx, cy = math.floor(fx/self.cellSizeX), math.floor(fy/self.cellSizeY)
        dx, dy = tx-fx, ty-fy
        sx, sy = sign(dx), sign(dy)

        if (sx == 0 and sy == 0):
            # No movement at all
            return []

        ts = []
        if (sx > 0):
            wx = (cx+1)*self.cellSizeX
            while (wx < tx):
                ts.append(((wx-fx)/dx, "E"))
                wx += self.cellSizeX

            if (wx-tx) < r:
                ts.append((clipU((wx-r-fx)/dx), "E"))
        elif (sx < 0):
            wx = cx*self.cellSizeX
            while (wx > tx):
                ts.append(((wx-fx)/dx, "W"))
                wx -= self.cellSizeX

            if (tx-wx) < r:
                ts.append((clipU((wx+r-fx)/dx), "W"))

        if (sy > 0):
            wy = (cy+1)*self.cellSizeY
            while (wy < ty):
                ts.append(((wy-fy)/dy, "S"))
                wy += self.cellSizeY

            if (wy-ty) < r:
                ts.append((clipU((wy-r-fy)/dx), "S"))

        elif (sy < 0):
            wy = cy*self.cellSizeY
            while (wy > ty):
                ts.append(((wy-fy)/dy, "N"))
                wy -= self.cellSizeY

            if (ty-wy) < r:
                ts.append((clipU((wy+r-fy)/dy), "N"))

        ts.sort()

        return ts

    def observe(self, fx, fy, angle, reach):
        """Measure the distance from the current position fx,fy to the wall
           following the given angle (in degrees).

           If the measured distance is larger than "reach", then -1 is
           returned, as a flag for "not-found".
        """

        # if self.distanceMap is None:  # Called with no maze build yet
        #     return -1

        # print("[DBG] (", fx, fy, ")@", angle, "<", reach, ">")

        rad = math.radians(angle)
        c = math.cos(rad)
        s = math.sin(rad)

        # Find out which maze border is crossed by the ray
        if abs(c) >= abs(s):
            # If cosine larger than sine we are between +/-45°
            sx = sign(c)
            sy = s/abs(c)
            tx, ty = fx, fy
            border = self.nx*self.cellSizeX if sx > 0 else -1
            # Avoid round() or otherwise we may step each 2
            while int(tx+0.5) != border:
                if self.distanceMap[int(ty+0.5), int(tx+0.5)] == 0:
                    break
                tx += sx
                ty += sy
        else:
            # y dominates
            sx = c/abs(s)
            sy = sign(s)
            tx, ty = fx, fy
            border = self.ny*self.cellSizeY if sy > 0 else -1
            # Avoid round() or otherwise we may step each 2
            while int(ty+0.5) != border:
                if self.distanceMap[int(ty+0.5), int(tx+0.5)] == 0:
                    break
                tx += sx
                ty += sy

        dx, dy = tx-fx, ty-fy
        dist = math.sqrt(dx*dx + dy*dy)
        return dist if dist < reach else -1

    def clipPos(self, fx, fy, tx, ty, r):
        """Restrict tx,ty to lie within the maze limits.  The algorithm
           expects that fx,fy lies inside the limits to work properly
        """

        lx = self.nx*self.cellSizeX - r
        ly = self.ny*self.cellSizeY - r

        if (tx < r or ty < r or tx >= lx or ty >= ly):

            dx, dy = tx-fx, ty-fy

            if dx > 0:
                ax = (lx-fx)/dx
            elif dx < 0:
                ax = (r-fx)/dx
            else:
                ax = 1

            if dy > 0:
                ay = (ly-fy)/dy
            elif dy < 0:
                ay = (r-fy)/dy
            else:
                ay = 1

            a = min(ax, ay)
            tx = fx + dx*a
            ty = fy + dy*a

        return tx, ty

    def tryMovement(self, fx, fy, tx, ty, r):
        """Tries to move a circle of radius r from (fx,fy) to (tx,ty).
           If it hits some wall on the way, return the stopping point
        """

        tx, ty = self.clipPos(fx, fy, tx, ty, r)
        dx, dy = tx-fx, ty-fy

        # No movement at all
        if (abs(dx) == 0) and (abs(dy) == 0):
            return tx, ty

        # Avoid round() or otherwise we may step each 2
        if self.distanceMap[int(fy+0.5), int(fx+0.5)] < r:
            print("[ERR] starting point at (", fx, ",", fy, ")",
                  " too close to walls")

        x, y = fx, fy
        lx, ly = x, y

        if abs(dx) >= abs(dy):
            # less than 45 degrees, use sx to run
            deltay = dy/abs(dx)
            sx = sign(dx)

            while abs(x-tx) > 1:
                # Avoid round() or otherwise we may step each 2
                if self.distanceMap[int(y+0.5), int(x+0.5)] <= r:
                    return lx, ly  # Collision detected.  Return last valid pos

                lx, ly = x, y
                x += sx
                y += deltay
        else:
            # less than 45 degrees around the y axis, use sy to run
            deltax = dx/abs(dy)
            sy = sign(dy)

            while abs(y-ty) > 1:
                # Avoid round() or otherwise we may step each 2
                if self.distanceMap[int(y+0.5), int(x+0.5)] <= r:
                    return lx, ly  # Collision detected.  Return last valid pos

                lx, ly = x, y
                y += sy
                x += deltax

        if self.distanceMap[int(ty+0.5), int(tx+0.5)] <= r:
            return lx, ly  # Collision detected.  Return last valid pos

        return tx, ty

    # Class constants used to encode neighbour directions
    delta = [('W', (-1, 0)),
             ('E', (1, 0)),
             ('S', (0, 1)),
             ('N', (0, -1))]

    def findValidNeighbors(self, cell):
        """Return a list of unvisited neighbours to cell."""

        neighbours = []
        for direction, (dx, dy) in self.delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cellAt(x2, y2)
                if neighbour.hasAllWalls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def findAWall(self, cell):
        """Return a list of pairs direction-neighbour if there is a wall
           between them.
        """

        neighbours = []
        for direction, (dx, dy) in self.delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cellAt(x2, y2)
                if cell.walls[direction]:
                    neighbours.append((direction, neighbour))
        return neighbours

    def build(self, seed=None):
        """To build a maze, first a cycle-free maze is created.  Afterwards, a
           number of random cells are picked and a random wall is
           removed.

           The number of cycles is given by self.cycles
        """

        # Full cells
        self.mazeMap = [[Cell(x, y) for x in range(self.nx)]
                        for y in range(self.ny)]

        # Total number of cells.
        n = self.nx * self.ny
        cellStack = []
        currentCell = self.cellAt(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        random.seed(seed)

        while nv < n:
            neighbours = self.findValidNeighbors(currentCell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                currentCell = cellStack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, nextCell = random.choice(neighbours)
            currentCell.deleteWall(nextCell, direction)
            cellStack.append(currentCell)
            currentCell = nextCell
            nv += 1

        for i in range(self.cycles):
            cx = random.randint(1, self.nx-2)  # exclude the border
            cy = random.randint(1, self.ny-2)  # exclude the border
            currentCell = self.cellAt(cx, cy)
            neighbours = self.findAWall(currentCell)
            if len(neighbours):
                direction, nextCell = random.choice(neighbours)
                print("[DBG] {}. Removing wall at {} in ({},{})".format(i,
                                                                        direction,
                                                                        cx,
                                                                        cy))
                currentCell.deleteWall(nextCell, direction)

        raster = self.rasterMap()
        self.distanceMap = ndimage.distance_transform_edt(raster)

    def saveDistanceMap(self, filename):
        img.toimage(self.distanceMap).save(filename)

    def hline(self, map, x1, y1, x2):
        map[y1, x1:x2+1] = 0

    def vline(self, map, x1, y1, y2):
        map[y1:y2+1, x1] = 0

    def rasterMap(self):
        """ Compute a numpy 2D binary array of the active maze, with
            1 representing floor and 0 representing a wall.
        """

        # Height and width of the maze image (excluding padding), in pixels
        height = self.ny*self.cellSizeY + 1
        width = self.nx*self.cellSizeX + 1

        rmap = np.ones((height, width))

        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = self.cellSizeY, self.cellSizeX

        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        for y in range(self.ny):
            for x in range(self.nx):
                if self.cellAt(x, y).walls['S']:
                    x1, ys, x2 = x*scx, (y+1)*scy, (x+1)*scx
                    self.hline(rmap, x1, ys, x2)
                if self.cellAt(x, y).walls['E']:
                    xe, y1, y2 = (x+1)*scx, y*scy, (y+1)*scy
                    self.vline(rmap, xe, y1, y2)

        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        self.vline(rmap, 0, 0, height-1)
        self.hline(rmap, 0, 0, width-1)

        return rmap

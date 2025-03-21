import numpy as np

class point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class cell:
    
    def __init__(self, x, y, z, cfg):
        self.x = x
        self.y = y
        self.z = z
        self.side = cfg.cell_side
        self.vertexes = self.__get_vertex()
        self.explored = False
        self.occupied = True
    
    # Vertexes follow the octant order
    def __get_vertex(self):

        vertexes = []
        vertexes.append(point(self.x + self.side/2, self.y + self.side/2, self.z + self.side/2))
        vertexes.append(point(self.x - self.side/2, self.y + self.side/2, self.z + self.side/2))
        vertexes.append(point(self.x - self.side/2, self.y - self.side/2, self.z + self.side/2))
        vertexes.append(point(self.x + self.side/2, self.y - self.side/2, self.z + self.side/2))
        vertexes.append(point(self.x + self.side/2, self.y + self.side/2, self.z - self.side/2))
        vertexes.append(point(self.x - self.side/2, self.y + self.side/2, self.z - self.side/2))
        vertexes.append(point(self.x - self.side/2, self.y - self.side/2, self.z - self.side/2))
        vertexes.append(point(self.x + self.side/2, self.y - self.side/2, self.z - self.side/2))

        return np.array(vertexes).reshape(-1)
    
    def IsIntersected(self, target):

        pass

    def setExplored(self):

        self.explored = True

    def setUnoccupied(self):

        self.occupied = False

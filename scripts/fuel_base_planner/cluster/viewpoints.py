import numpy as np
from .cells import point
from .frontier_cluster import frontier


class viewpoint:

    def __init__(self, dist, angle):

        self.dist = dist
        self.angle = angle

    def cal_coverage(self, f:frontier):
        self.coord = self.cal_coordinate(f)
        pass

    def cal_coordinate(self, f:frontier) -> point:

        return None


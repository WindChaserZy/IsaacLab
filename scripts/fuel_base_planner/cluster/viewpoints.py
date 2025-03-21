import numpy as np
from .cells import point
from .frontier_cluster import cluster


class viewpoint:

    def __init__(self, dist, angle):

        self.dist = dist
        self.angle = angle

    def cal_coverage(self, f:cluster):
        self.coord = self.cal_coordinate(f)
        pass

    def cal_coordinate(self, f:cluster) -> point:

        return None


import numpy as np

from ..cluster import cell, point, cluster


class map_manager:

    def __init__(self, cfg):

        self.set_border(cfg)
        self.uav_pos = point(cfg.uav_init_x, cfg.uav_init_y, cfg.uav_init_z)
        self.init_map(cfg)
        self.frontier_list = []

    def set_border(self, cfg):

        self.x_max = cfg.map_x_max
        self.y_max = cfg.map_y_max
        self.z_max = cfg.map_z_max

    def init_map(self, cfg):

        self.voxel_map = [[[cell(x, y, z, cfg) for z in range(self.z_max)] for y in range(self.y_max)] for x in range(self.x_max)]



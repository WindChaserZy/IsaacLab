import numpy as np
from .cells import cell, point


class frontier:

    def __init__(self, cell_list, order, cfg):

        self.cells = cell_list
        self.avg_cell_pos = self._avg_cell_pos()
        self.order = order
        self.vp_max = cfg.N_view
        self.vp_list = self._vp_filter()
        self.cost_list = []


    def _avg_cell_pos(self):

        if not self.cells:
            return None
        
        avg_x = sum(cell.x for cell in self.cells) / len(self.cells)
        avg_y = sum(cell.y for cell in self.cells) / len(self.cells)
        avg_z = sum(cell.z for cell in self.cells) / len(self.cells)

        return point(avg_x, avg_y, avg_z)
        
    def _vp_gen(self):
        pass

    def _vp_filter(self):
        pass



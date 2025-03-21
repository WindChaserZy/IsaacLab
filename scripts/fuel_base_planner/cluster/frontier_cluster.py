import numpy as np
from .cells import cell, point


class cluster:

    def __init__(self, cell_list, cfg):

        self.cells = cell_list
        self.cfg = cfg
        self.avg_cell_pos = self._avg_cell_pos()
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

    def recursively_split(self):

        N_max = self.N_max
        result_clusters = []

        if len(self.cells) > N_max:
            cluster1, cluster2 = self.pca_split()

            result_clusters.extend(cluster1.pca_split())
            result_clusters.extend(cluster2.pca_split())
        else:
            result_clusters.append(self)

        return result_clusters

    def pca_split(self):

        points = [c.get_coord() for c in self.cells]

        mean = np.mean(points, axis=0)
        centered_points = points - mean

        cov_matrix = np.cov(centered_points, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        principal_component = eigenvectors[:, np.argmax(eigenvalues)]

        projections = np.dot(centered_points, principal_component)

        median_projection = np.median(projections)
        mask = projections < median_projection

        return cluster(self.cells[mask], self.cfg), cluster(self.cells[-mask], self.cfg)
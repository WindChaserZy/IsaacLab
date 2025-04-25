from typing import List, Tuple
from ..utils import Vector3d
from .sdf_map import SDFmap
from .obj_predictor import PolynomialPrediction as PolyPred

class EDTEnv:

    def __init__(self):

        self.obj_prediction_ : List[PolyPred] = []
        self.obj_scale_ : List[Vector3d] = []
        self.resolution_inv_  = 0.0
    
    def setMap(self, map: SDFmap):
        
        self.sdf_map_ = map
        self.resolution_inv_ = 1.0 / self.sdf_map_.getResolution()

    def setObjPrediction(self, obj_prediction: List[PolyPred]):
        self.obj_prediction_ = obj_prediction

    def setObjScale(self, obj_scale: List[Vector3d]):
        self.obj_scale_ = obj_scale

    def distToBox(self, idx: int, pos: Vector3d, time: float):

        pos_box = self.obj_prediction_[idx].evaluateConstVel(time)
        pos_box_vec = Vector3d(pos_box[0], pos_box[1], pos_box[2])
        box_max = pos_box_vec + 0.5 * self.obj_scale_[idx]
        box_min = pos_box_vec - 0.5 * self.obj_scale_[idx]

        dist = Vector3d()

        for i in range(3):
            if box_min[i] <= pos[i] <= box_max[i]:
                dist[i] = 0.0
            else:
                dist[i] = min(abs(pos[i] - box_min[i]), 
                            abs(pos[i] - box_max[i]))
        
        return dist.norm()
    
    def minDistToAllBox(self, pos: Vector3d, time: float):

        dist = 10000000.0
        for i in range(len(self.obj_prediction_)):

            di = self.distToBox(i, pos, time)
            if di < dist:
                dist = di
        return dist
    
    def getSurroundDistance(self, pts: List[List[List[Vector3d]]]) -> List[List[List[float]]]:
        dists = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    dists[x][y][z] = self.sdf_map_.getDistance(pts[x][y][z])       
        return dists
    
    def interpolateTrilinear(self, values: List[List[List[float]]], diff: Vector3d) -> Tuple[float, Vector3d]:
        # trilinear interpolation
        v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0]
        v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1]
        v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0]
        v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1]
        v0 = (1 - diff[1]) * v00 + diff[1] * v10
        v1 = (1 - diff[1]) * v01 + diff[1] * v11

        value = (1 - diff[2]) * v0 + diff[2] * v1

        grad = Vector3d()
        grad[2] = (v1 - v0) * self.resolution_inv_
        grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * self.resolution_inv_
        
        grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0])
        grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0])
        grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1])
        grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1])
        grad[0] *= self.resolution_inv_

        return value, grad
    
    def evaluateEDTWithGrad(self, pos:Vector3d, time:float, dist: float, grad: Vector3d) -> Tuple[float, Vector3d]:

        dist, grad = self.sdf_map_.getDistWithGrad(pos, grad)
        return dist, grad
    
    def evaluateCoarseEDT(self, pos: Vector3d, time: float) -> float:

        d1 = self.sdf_map_.getDistance(pos)
        if time < 0.0:
            return d1
        else:
            d2 = self.minDistToAllBox(pos, time)
            return min(d1, d2)
        


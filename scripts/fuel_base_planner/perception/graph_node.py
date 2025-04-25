from typing import List
import math
from ..env import EDTEnv, SDFmap, RayCaster
from ..searching import AStar
from ..utils import Vector3d, Vector3i

class BaseNode:

    def __init__(self):

        self.g_value_ = 1000000
        self.closed_ = False
        self.id_ = 0

    def __repr__(self):
        return f"Base Node"
    
    def __str__(self):
        return self.__repr__()
    
class ViewNode(BaseNode):

    vm_ = 0.0
    am_ = 0.0
    yd_ = 0.0
    ydd_ = 0.0
    w_dir_ = 0.0
    astar_ = AStar(EDTEnv()) #####################
    caster_ = RayCaster()
    map_ = SDFmap()

    def __init__(self, p: Vector3d = Vector3d(), y: float = 0.0):

        super().__init__()
        self.pos_ = p
        self.yaw_ = y
        self.parent_ = None
        self.vel_ = Vector3d()
        self.neighbors_: List[ViewNode] = []

        self.yaw_dot_ = 0.0
    
    def __lt__(self, other: 'ViewNode'):
        return self.g_value_ < other.g_value_
    
    def __eq__(self, other: 'ViewNode'):
        return self.id_ == other.id_
    
    def costTo(self, node: 'ViewNode') -> float:

        path = []
        c = self.computeCost(self.pos_, node.pos_, self.yaw_, node.yaw_, self.vel_, self.yaw_dot_, path)
        return c
    
    def searchPath(self, p1: Vector3d, p2: Vector3d, path: List[Vector3d]):

        safe = True
        idx = Vector3i()
        self.caster_.input(p1, p2)
        while self.caster_.nextId(idx):
            if self.map_.getInflatedOccupancy(idx) == 1 or self.map_.getOccupancy(idx) == self.map_.UNKNOWN or not self.map_.isInBox(idx):
                safe = False
                break
        if safe:
            path = [p1, p2]
            return (p1 - p2).norm(), path
        res = [0.4]
        for k in range(len(res)):
            self.astar_.reset()
            self.astar_.setResolution(res[k])
            if self.astar_.search(p1, p2) == AStar.REACH_END:
                path = self.astar_.getPath()
                return self.astar_.pathLength(path), path
        path = [p1, p2]
        return 1000, path

    def computeCost(self, p1: Vector3d, p2: Vector3d, y1: float, y2: float, v1: Vector3d, yd1: float, path: List[Vector3d]) -> float:

        pos_cost, path = self.searchPath(p1, p2, path)
        pos_cost /= self.vm_
        
        if v1.norm() > 1e-3:
            dir = (p2 - p1).normalized()
            vdir = v1.normalized()
            diff = math.acos(vdir.x * dir.x + vdir.y * dir.y + vdir.z * dir.z)
            pos_cost += self.w_dir_ * diff

        diff = abs(y1 - y2)
        diff = min(diff, 2 * math.pi - diff)
        yaw_cost = diff / self.yd_

        return max(pos_cost, yaw_cost)
        
  
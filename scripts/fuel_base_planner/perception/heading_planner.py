from ..env import RayCaster, SDFmap
from ..utils import Vector3d, Vector3i
import numpy as np
import math
import heapq
from typing import List, Union

class BaseVertex:

    def __init__(self):
        self.id_ = 0
        self.g_value_ = 1000000

class YawVertex(BaseVertex):

    def __init__(self, y: float, gain: float, id: int):

        self.yaw_ = y
        self.info_gain_ = gain
        self.id_ = id
        self.visib_ = 0
        self.neighbors_ : List[YawVertex] = []
        self.parent_ : Union[YawVertex, None] = None

    def gain(self) -> float:
        return self.info_gain_
    
    def dist(self, v: 'YawVertex') -> float:
        return abs(self.yaw_ - v.yaw_)
    
class Graph:

    def __init__(self):

        self.vertice_ : List[YawVertex] = []
        self.w_ = 0
        self.max_yaw_rate_ = 0
        self.dt_ = 0
    
    def addVertex(self, vertex: YawVertex):
        self.vertice_.append(vertex)
    
    def addEdge(self, from_vertex: int, to_vertex: int):
        self.vertice_[from_vertex].neighbors_.append(self.vertice_[to_vertex])
    
    def setParams(self, w: float, max_yaw_rate: float, dt: float):
        self.w_ = w
        self.max_yaw_rate_ = max_yaw_rate
        self.dt_ = dt
    
    def penal(self, diff: float) -> float:

        yr = diff / self.dt_
        if yr <= self.max_yaw_rate_:
            return 0.0
        else:
            return math.pow(yr - self.max_yaw_rate_, 2)
    
    def dijkstraSearch(self, start: int, goal: int, path: List[YawVertex]):

        start_v = self.vertice_[start]
        goal_v = self.vertice_[goal]

        start_v.g_value_ = 0
        open_set = []
        open_set_map = {}
        close_set = {}

        heapq.heappush(open_set, start_v)
        open_set_map[start_v.id_] = 1

        while open_set:
            vc = heapq.heappop(open_set)
            open_set_map.pop(vc.id_)
            close_set[vc.id_] = 1

            if vc == goal_v:
                vit = vc.copy()
                while vit.parent_ != None:
                    path.append(vit)
                    vit = vit.parent_
                path.reverse()
                return path
            
            for vb in vc.neighbors_:
                if vb.id_ in close_set:
                    continue
                g_tmp = vc.g_value_ + self.w_ * self.penal(vc.dist(vb)) - vb.gain()
                if vb.id_ not in open_set_map:
                    open_set_map[vb.id_] = 1
                    open_set.append(vb)
                elif g_tmp > vb.g_value_:
                    continue
                vb.parent_ = vc
                vb.g_value_ = g_tmp

        return path

class CastFlags:

    NON_UNIFORM = 0
    UNIFORM = 1

    def __init__(self, size = 0):
        self.flags_ = [0] * size
        self.lb_ = Vector3i()
        self.ub_ = Vector3i()
        self.cells_ = Vector3i()
    
    def reset(self, lb: Vector3i, ub: Vector3i):
        self.lb_ = lb
        self.ub_ = ub
        self.cells_ = ub - lb
        for i, flag in enumerate(self.flags_):
            self.flags_[i] = 0

    def address(self, idx: Vector3i):
        diff = idx - self.lb_
        return diff.z + diff.y * self.cells_.z + diff.x * self.cells_.z * self.cells_.y
    
    def getFlag(self, idx: Vector3i):
        return self.flags_[self.address(idx)]
    
    def setFlag(self, idx: Vector3i, flag: int):
        self.flags_[self.address(idx)] = flag

class HeadingPlanner:

    def __init__(self):

        self.frontier_: List[Vector3d] = []
        self.ft_kdtree_ = None
        self.sdf_map_ = SDFmap()
        self.casters_ : List[RayCaster] = []

        self.tan_yz_ = 0
        self.tan_xz_ = 0
        self.near_ = 0
        self.far_ = 4.5
        
        #########################
        self.yaw_diff_ = -1.0
        self.lambda1_ = -1.0
        self.lambda2_ = -1.0
        self.half_vert_num_ = -1
        self.max_yaw_rate_ = -1.0
        self.w_ = -1.0
        self.weight_type_ = -1
        #########################

        top_ang = 0.56125
        self.n_top_ = Vector3d(0, math.sin(math.pi/2 - top_ang), math.cos(math.pi/2 - top_ang))
        self.n_bottom_ = Vector3d(0, -math.sin(math.pi/2 - top_ang), math.cos(math.pi/2 - top_ang))
        left_ang = 0.69222
        right_ang = 0.68925
        self.n_left_ = Vector3d(math.sin(math.pi/2 - left_ang), 0, math.cos(math.pi/2 - left_ang))
        self.n_right_ = Vector3d(-math.sin(math.pi/2 - right_ang), 0, math.cos(math.pi/2 - right_ang))

        ## minor change on lefttop
        self.lefttop_ = Vector3d(-self.far_ * math.sin(left_ang), -self.far_ * math.sin(top_ang), self.far_)
        self.rightbottom_ = Vector3d(self.far_ * math.sin(right_ang), self.far_ * math.sin(top_ang), self.far_)
        self.leftbottom_ = Vector3d(-self.far_ * math.sin(left_ang), self.far_ * math.sin(top_ang), self.far_)
        self.righttop_ = Vector3d(self.far_ * math.sin(right_ang), -self.far_ * math.sin(top_ang), -self.far_)
        
        self.cast_flags_ = CastFlags(1000000)
        self.T_cb_ = np.array([0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
        self.T_bc_ = np.linalg.inv(self.T_cb_)
    
        self.casters_ = [RayCaster() for _ in range(2 * self.half_vert_num_ + 1)]

    def setMap(self, map: SDFmap):
        self.sdf_map_ = map

    def searchPathOfYaw(self, pts: List[Vector3d], yaws: List[float], dt: float, ctrl_pts: np.ndarray, path: List[float]):

        yaw_graph = Graph()
        yaw_graph.setParams(self.w_, self.max_yaw_rate_, dt)
        gid = 0
        layer, last_layer = [], []
        cur_pos = pts[0]

        for i in range(len(yaws)):
            pass

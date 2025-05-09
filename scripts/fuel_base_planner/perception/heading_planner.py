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
        self.yaw_diff_ = 30 * 3.1415926 / 180
        self.lambda1_ = 2.0
        self.lambda2_ = 1.0
        self.half_vert_num_ = 5
        self.max_yaw_rate_ = 10 * 3.1415926 / 180
        self.w_ = 20000.0
        self.weight_type_ = 1
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
            start_end = (i == 0 or i == len(yaws) - 1)
            if start_end:
                vert = YawVertex(yaws[i], 0, gid)
                gid += 1
                yaw_graph.addVertex(vert)
                layer.append(vert)
            else:
                self.initCastFlag(pts[i])
                vert_num = 2 * self.half_vert_num_ + 1
                futs = []
                for j in range(vert_num):
                    ys = yaws[i] + float(j - self.half_vert_num_) * self.yaw_diff_
                    futs.append(self.calcInformationGain(pts[i], ys, ctrl_pts, j))
                
                for j in range(vert_num):
                    ys = yaws[i] + float(j - self.half_vert_num_) * self.yaw_diff_
                    gain = futs[j]
                    vert = YawVertex(ys, gain, gid)
                    gid += 1
                    yaw_graph.addVertex(vert)
                    layer.append(vert)
            for v1 in last_layer:
                for v2 in layer:
                    yaw_graph.addEdge(v1.id_, v2.id_)
            last_layer = layer

        vert_path : List[YawVertex] = []
        vert_path = yaw_graph.dijkstraSearch(0, gid - 1, vert_path)
        for vert in vert_path:
            path.append(vert.yaw_)
        return path
            
    def initCastFlag(self, pos: Vector3d):
        vec = Vector3d(self.far_, self.far_, self.rightbottom_[1])
        lbi = self.sdf_map_.posToIndex(pos - vec)
        ubi = self.sdf_map_.posToIndex(pos + vec)
        self.cast_flags_.reset(lbi, ubi)

    def calcInformationGain(self, pt: Vector3d, yaw: float, ctrl_pts: np.ndarray, task_id: int):

        R_wb = np.array([math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]).reshape(3, 3)
        
        T_wb = np.identity(4)
        T_wb[:3, :3] = R_wb
        T_wb[:3, 3] = pt

        T_wc = T_wb @ self.T_bc_
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        
        normals = [self.n_top_, self.n_bottom_, self.n_left_, self.n_right_]
        for i in range(len(normals)):
            normal_array = np.array([normals[i].x, normals[i].y, normals[i].z])
            rotated = R_wc @ normal_array
            normals[i] = Vector3d(rotated[0], rotated[1], rotated[2])

        lbi, ubi = self.calcFovAABB(R_wc, t_wc)

        resolution = self.sdf_map_.getResolution()
        origin, size = self.sdf_map_.getRegion()
        offset = Vector3d(0.5, 0.5, 0.5) - origin / resolution

        gain_pts = []
        dist12 = (0.0, 0.0)
        factor = 4
        gain = 0.0

        for i in range(lbi.x, ubi.x + 1):
            for j in range(lbi.y, ubi.y + 1):
                for k in range(lbi.z, ubi.z + 1):
                    if not (i % factor == 0 and j % factor == 0 and k % factor == 0):
                        continue

                    pt_idx = Vector3i(i, j, k)
                    if not self.sdf_map_.getOccupancy(pt_idx) == SDFmap.UNKNOWN:
                        continue
                    if not self.sdf_map_.isInBox(pt_idx):
                        continue
                    check_pt = self.sdf_map_.indexToPos(pt_idx)
                    if not self.insideFOV(check_pt, pt, normals):
                        continue
                    
                    flag = self.cast_flags_.getFlag(pt_idx)
                    if flag == 1:
                        if self.weight_type_ == CastFlags.UNIFORM:
                            gain += 1.0
                        elif self.weight_type_ == CastFlags.NON_UNIFORM:
                            dist1, dist2 = self.distToPathAndCurPos(check_pt, ctrl_pts)
                            gain += math.exp(-self.lambda1_ * dist1 - self.lambda2_ * dist2)
                    elif flag == 0:
                        result = 1
                        self.casters_[task_id].setInput(check_pt / resolution, pt / resolution)
                        ray_pt = Vector3d()
                        ray_id = Vector3i()
                        st, ray_pt = self.casters_[task_id].step(ray_pt)
                        while st:
                            ray_id = ray_pt + offset
                            if self.sdf_map_.getOccupancy(ray_id) == SDFmap.UNKNOWN:
                                result = 2
                                break
                        if result == 1:
                            if self.weight_type_ == CastFlags.UNIFORM:
                                gain += 1.0
                            elif self.weight_type_ == CastFlags.NON_UNIFORM:
                                dist1, dist2 = self.distToPathAndCurPos(check_pt, ctrl_pts)
                                gain += math.exp(-self.lambda1_ * dist1 - self.lambda2_ * dist2)
                        self.cast_flags_.setFlag(pt_idx, result)
    
    def calcFovAABB(self, R_wc: np.ndarray, t_wc: np.ndarray):
        vertices = np.zeros((5, 3))
        vertices[0] = R_wc @ self.lefttop_ + t_wc
        vertices[1] = R_wc @ self.leftbottom_ + t_wc
        vertices[2] = R_wc @ self.righttop_ + t_wc
        vertices[3] = R_wc @ self.rightbottom_ + t_wc
        vertices[4] = t_wc

        lbd = np.min(vertices, axis=0)
        ubd = np.max(vertices, axis=0)

        lbd = Vector3d(lbd[0], lbd[1], lbd[2])
        ubd = Vector3d(ubd[0], ubd[1], ubd[2])

        lbd, ubd = self.sdf_map_.boundBox(lbd, ubd)

        lbi = self.sdf_map_.posToIndex(lbd)
        ubi = self.sdf_map_.posToIndex(ubd)

        return lbi, ubi
    
    def insideFOV(self, pw: Vector3d, pc: Vector3d, normals: List[Vector3d]):
        dir = pw - pc
        if dir.norm() > self.far_:
            return False
        
        for n in normals:
            dot_product = n.x * dir.x + n.y * dir.y + n.z * dir.z
            if dot_product < 0.1:
                return False
        return True
    
    def distToPathAndCurPos(self, pt: Vector3d, ctrl_pts: np.ndarray):

        min_squ = float('inf')
        idx = -1
        for i in range(ctrl_pts.shape[0]):
            ctrl_pt = ctrl_pts[i]
            squ = (ctrl_pt[0] - pt.x)**2 + (ctrl_pt[1] - pt.y)**2 + (ctrl_pt[2] - pt.z)**2
            if squ < min_squ:
                min_squ = squ
                idx = i

        dist_to_pt = np.sqrt(min_squ)
        dist_along_path = 0.0
        
        for i in range(idx):
            dist_along_path += np.linalg.norm(ctrl_pts[i+1] - ctrl_pts[i])

        return dist_to_pt, dist_along_path
from typing import List, Tuple, Dict, Union
import time
from math import sqrt
import heapq
from ..env import EDTEnv
from ..utils import Vector3d, Vector3i
from .matrix_hash import MatrixHash

class Node:

    def __init__(self):

        self.index: Vector3i = Vector3i()
        self.position: Vector3d = Vector3d()
        self.g_score: float = 0.0
        self.f_score: float = 0.0
        self.parent: Union[Node, None] = None

    def __lt__(self, other: 'Node'):
        return self.f_score < other.f_score
    
    def __eq__(self, other: 'Node'):
        return self.index == other.index
    

class AStar:

    REACH_END = 1
    NO_PATH = 2

    def __init__(self, env: EDTEnv):
        # Config to be added
        self.lambda_heu_ = -1.0
        self.max_search_time_ = -1.0
        self.resolution_ = -1.0
        self.allocate_num_ = 1000000
        ########################
        self.use_node_num_: int = 0
        self.iter_num_: int = 0
        self.open_set_: List[Node] = []  # Priority queue for open set
        self.open_set_map_: Dict[Vector3i, Node] = {}  # Hash map for open set
        self.close_set_map_: Dict[Vector3i, Node] = {}  # Hash map for closed set
        self.path_nodes_: List[Vector3d] = []  # Path nodes

        self.early_terminate_cost_ = 0.0
        self.edt_env_: EDTEnv = env
        self.margin_ = 0.0
        self.tie_breaker_ = 1.0 + 1.0 / 1000
        self.inv_resolution_ = 1.0 / self.resolution_
        self.origin_, self.map_size_3d_ = self.edt_env_.sdf_map_.getRegion()
        self.path_node_pool_ = [Node() for _ in range(self.allocate_num_)] 

    def setResolution(self, res: float):
        self.resolution_ = res
        self.inv_resolution_ = 1.0 / self.resolution_

    def getEarlyTerminateCost(self) -> float:
        return self.early_terminate_cost_
    
    def reset(self):
        self.open_set_map_.clear()
        self.close_set_map_.clear()
        self.path_nodes_.clear()
        self.open_set_.clear()
        for i in range(self.use_node_num_):
            self.path_node_pool_[i].parent = None
        self.use_node_num_ = 0
        self.iter_num_ = 0

    @staticmethod
    def pathLength(path: List[Vector3d]) -> float:
        length = 0.0
        if len(path) < 2:
            return length
        for i in range(1, len(path)):
            length += (path[i] - path[i - 1]).norm()
        return length
        
    def backtrack(self, end_node: Node, end: Vector3d):

        self.path_nodes_.append(end)
        self.path_nodes_.append(end_node.position)
        cur_node = end_node
        while cur_node.parent is not None:
            self.path_nodes_.append(cur_node.parent.position)
            cur_node = cur_node.parent
        self.path_nodes_.reverse()

    def getPath(self) -> List[Vector3d]:
        return self.path_nodes_
        
    def getDiagHeu(self, x1: Vector3d, x2: Vector3d) -> float:
        dx = abs(x1.x - x2.x)
        dy = abs(x1.y - x2.y)
        dz = abs(x1.z - x2.z)
        diag = min(dx, dy, dz)
        dx -= diag
        dy -= diag
        dz -= diag
        h = 0
        if dx < 1e-4:
            h = 1.0 * sqrt(3) * diag + sqrt(2) * min(dy, dz) + 1.0 * abs(dy - dz)
        elif dy < 1e-4:
            h = 1.0 * sqrt(3) * diag + sqrt(2) * min(dx, dz) + 1.0 * abs(dx - dz)
        elif dz < 1e-4:
            h = 1.0 * sqrt(3) * diag + sqrt(2) * min(dx, dy) + 1.0 * abs(dx - dy)
        return self.tie_breaker_ * h
    
    def getManHeu(self, x1: Vector3d, x2: Vector3d) -> float:
        dx = abs(x1.x - x2.x)
        dy = abs(x1.y - x2.y)
        dz = abs(x1.z - x2.z)
        return self.tie_breaker_ * (dx + dy + dz)
    

    def getEucHeu(self, x1: Vector3d, x2: Vector3d) -> float:
        return self.tie_breaker_ * (x1 - x2).norm()
    
    def getVisited(self):
        visited = []
        for i in range(self.use_node_num_):
            visited.append(self.path_node_pool_[i].position)
        return visited
    
    def posToIndex(self, pos: Vector3d) -> Vector3i:
        return Vector3i(
            int((pos.x - self.origin_.x) * self.inv_resolution_),
            int((pos.y - self.origin_.y) * self.inv_resolution_),
            int((pos.z - self.origin_.z) * self.inv_resolution_)
        )

    def search(self, start_pt: Vector3d, end_pt: Vector3d):

        cur_node = self.path_node_pool_[0]
        cur_node.parent = None
        cur_node.position = start_pt
        cur_node.index = self.posToIndex(start_pt)
        cur_node.g_score = 0.0
        cur_node.f_score = self.lambda_heu_ * self.getDiagHeu(cur_node.position, end_pt)

        end_index = self.posToIndex(end_pt)
        heapq.heappush(self.open_set_, cur_node)
        self.open_set_map_[cur_node.index] = cur_node
        self.use_node_num_ += 1

        while len(self.open_set_) > 0:
            cur_node = self.open_set_[0]
            reach_end = abs(cur_node.index.x - end_index.x) <= 1 and abs(cur_node.index.y - end_index.y) <= 1 and abs(cur_node.index.z - end_index.z) <= 1
            if reach_end:
                self.backtrack(cur_node, end_pt)
                return self.REACH_END
            
            heapq.heappop(self.open_set_)
            self.open_set_map_.pop(cur_node.index)
            self.close_set_map_[cur_node.index] = cur_node
            self.iter_num_ += 1
            if self.iter_num_ > self.max_search_time_:
                return self.NO_PATH
            
            cur_pos = cur_node.position
            nbr_pos = Vector3d()
            step = Vector3d()

            for dx in [-self.resolution_, 0, self.resolution_]:
                for dy in [-self.resolution_, 0, self.resolution_]:
                    for dz in [-self.resolution_, 0, self.resolution_]:
                        step = Vector3d(dx, dy, dz)
                        if step.norm() < 1e-3:
                            continue
                        nbr_pos = cur_pos + step
                        # Check if neighbor is in bounds and safe
                        if not self.edt_env_.sdf_map_.isInBox(nbr_pos):
                            continue
                        if (self.edt_env_.sdf_map_.getInflatedOccupancy(nbr_pos) == 1 or
                            self.edt_env_.sdf_map_.getOccupancy(nbr_pos) == self.edt_env_.sdf_map_.UNKNOWN):
                            continue

                        # Check line-of-sight safety
                        safe = True
                        dir = nbr_pos - cur_pos 
                        dir_len = dir.norm()
                        dir.normalize() # Normalize
                        
                        for l in range(1, int(dir_len/0.1)):
                            ckpt = cur_pos + 0.1 * l * dir
                            if (self.edt_env_.sdf_map_.getInflatedOccupancy(ckpt) == 1 or
                                self.edt_env_.sdf_map_.getOccupancy(ckpt) == self.edt_env_.sdf_map_.UNKNOWN):
                                safe = False
                                break
                        
                        if not safe:
                            continue

                        # Check if neighbor is in closed set
                        nbr_idx = self.posToIndex(nbr_pos)
                        if nbr_idx in self.close_set_map_:
                            continue

                        # Calculate scores and update neighbor
                        neighbor = Node()
                        tmp_g_score = step.norm() + cur_node.g_score
                        
                        if nbr_idx not in self.open_set_map_:
                            neighbor = self.path_node_pool_[self.use_node_num_]
                            self.use_node_num_ += 1
                            if self.use_node_num_ >= self.allocate_num_:
                                print("Ran out of node pool")
                                return self.NO_PATH
                            neighbor.index = nbr_idx
                            neighbor.position = nbr_pos
                        elif tmp_g_score < self.open_set_map_[nbr_idx].g_score:
                            neighbor = self.open_set_map_[nbr_idx]
                        else:
                            continue
                        
                        neighbor.parent = cur_node
                        neighbor.g_score = tmp_g_score
                        neighbor.f_score = tmp_g_score + self.lambda_heu_ * self.getDiagHeu(nbr_pos, end_pt)
                        heapq.heappush(self.open_set_, neighbor)
                        self.open_set_map_[nbr_idx] = neighbor
        
        return self.NO_PATH
    
            
                    
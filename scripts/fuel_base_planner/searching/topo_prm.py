from enum import Enum
from typing import List, Optional
from ..utils import Vector3d, Vector3i
from ..env import EDTEnv, RayCaster
import random
import os
import numpy as np

class TopoIterator:

    def __init__(self, pn: List[int] = []):

        self.path_nums_ : List[int] = pn
        self.cur_index_ : List[int] = [0 for _ in range(len(pn))]
        self.combine_num_ = 1
        self.cur_num_ = 0

        for i in range(len(pn)):
            self.combine_num_ *= (self.path_nums_[i] if self.path_nums_[i] > 0 else 1)
    
    def increase(self, bit_num: int):
        self.cur_index_[bit_num] += 1
        if self.cur_index_[bit_num] >= self.path_nums_[bit_num]:
            self.cur_index_[bit_num] = 0
            self.increase(bit_num + 1)
    
    def nextIndex(self, index: List[int]):
        index = self.cur_index_
        self.cur_num_ += 1
        if self.cur_num_ == self.combine_num_:
            return False, index
        self.increase(0)
        return True, index
    
class GraphNode:
    class NodeType(Enum):
        Guard = 1
        Connector = 2

    class NodeState(Enum):
        NEW = 1
        CLOSE = 2
        OPEN = 3
    def __init__(self, pos: Vector3d, node_type: NodeType, id: int):

        self.pos_ = pos
        self.type_ = node_type
        self.state_ = self.NodeState.NEW
        self.id_ = id
        self.neighbors_: List[GraphNode] = []

class TopologyPRM:

    def __init__(self):
        self.graph_ :List[GraphNode] = []
        self.rd_ = int.from_bytes(os.urandom(8), 'big')  
        self.eng_ = random.Random(self.rd_)                 
        self.sample_inflate_ = Vector3d()
        ####### Config #######
        self.sample_inflate_[0] = 0.0
        self.sample_inflate_[1] = 0.0
        self.sample_inflate_[2] = 0.0
        self.clearance_ = -1.0
        self.short_cut_num_ = -1
        self.reserve_num_ = -1
        self.ratio_to_short_ = -1.0
        self.max_sample_num_ = -1
        self.max_sample_time_ = -1.0
        self.max_raw_path_ = 1
        self.max_raw_path2_ = -1
        self.parallel_shortcut_ = False
        ################################
        self.casters_ :List[RayCaster] = [RayCaster() for _ in range(self.max_raw_path_)]

        self.sample_r_ = Vector3d()
        self.translation_ = Vector3d()
        self.rotation_ = np.eye(3)
        self.raw_paths_ :List[List[Vector3d]] = []
        self.short_paths_ :List[List[Vector3d]] = []
        self.final_paths_ :List[List[Vector3d]] = []
        self.start_pts_ :List[Vector3d] = []
        self.end_pts_ :List[Vector3d] = [] 

    def rand_pos_(self):
        return self.eng_.uniform(-1.0, 1.0)

    def setEnviroment(self, env: EDTEnv):
        self.edt_enviroment_ = env
        self.resolution_ = self.edt_enviroment_.sdf_map_.getResolution()
        origin, size = self.edt_enviroment_.sdf_map_.getRegion()
        self.offset_ = Vector3d(0.5, 0.5, 0.5) - origin / self.resolution_
    
    def findTopoPaths(self, start: Vector3d, end: Vector3d, start_pts: List[Vector3d], end_pts: List[Vector3d], graph: List[GraphNode], raw_paths: List[List[Vector3d]], filtered_paths: List[List[Vector3d]], selected_paths: List[List[Vector3d]]):

        self.start_pts_ = start_pts
        self.end_pts_ = end_pts

        graph = self.createGraph(start, end)

        raw_paths = self.searchPaths()

        self.shortcutPaths()

        filtered_paths = self.pruneEquivalent(self.short_paths_)

        selected_paths = self.selectShortPaths(filtered_paths, 1)

        self.final_paths_ = selected_paths

        return graph, raw_paths, filtered_paths, selected_paths

    def createGraph(self, start: Vector3d, end: Vector3d):

        self.graph_.clear()
        self.graph_.append(GraphNode(start, GraphNode.NodeType.Guard, 0))
        self.graph_.append(GraphNode(end, GraphNode.NodeType.Guard, 1))

        self.sample_r_[0] = 0.5 * (end - start).norm() + self.sample_inflate_[0]
        self.sample_r_[1] = self.sample_inflate_[1]
        self.sample_r_[2] = self.sample_inflate_[2]

        self.translation_ = (end + start) / 2.0
        downward = Vector3d(0.0, 0.0, -1.0)
        xtf = (end - self.translation_).normalized()
        ytf = Vector3d(
            xtf.y * downward.z - xtf.z * downward.y,
            xtf.z * downward.x - xtf.x * downward.z,
            xtf.x * downward.y - xtf.y * downward.x
        ).normalized()
        ztf = Vector3d(
            xtf.y * ytf.z - xtf.z * ytf.y,
            xtf.z * ytf.x - xtf.x * ytf.z,
            xtf.x * ytf.y - xtf.y * ytf.x
        )

        self.rotation_[:, 0] = [xtf.x, xtf.y, xtf.z]
        self.rotation_[:, 1] = [ytf.x, ytf.y, ytf.z]
        self.rotation_[:, 2] = [ztf.x, ztf.y, ztf.z]

        node_id = 1
        sample_num = 0
        sample_time = 0.0
        pt = Vector3d()
        ## time to be used
        while sample_num < self.max_sample_num_:
            pt = self.getSample()
            sample_num += 1
            dist = self.edt_enviroment_.evaluateCoarseEDT(pt, -1.0)
            if dist <= self.clearance_:
                continue
            visib_guards = self.findVisibGuard(pt)
            if len(visib_guards) == 0:
                guard = GraphNode(pt, GraphNode.NodeType.Guard, node_id)
                node_id += 1
            elif len(visib_guards) == 2:
                need_connect = self.needConnection(visib_guards[0], visib_guards[1], pt)
                if not need_connect:
                    continue
                connector = GraphNode(pt, GraphNode.NodeType.Connector, node_id)
                node_id += 1
                self.graph_.append(connector)
                visib_guards[0].neighbors_.append(connector)
                visib_guards[1].neighbors_.append(connector)
                connector.neighbors_.append(visib_guards[0])
                connector.neighbors_.append(visib_guards[1])
        
        self.pruneGraph()

        return self.graph_

    def findVisibGuard(self, pt: Vector3d):

        visib_guards = []
        pc = Vector3d()
        visib_num = 0

        # Find visible GUARD from pt
        for node in self.graph_:
            if node.type_ == GraphNode.NodeType.Connector:
                continue
            visib, pc = self.lineVisib(pt, node.pos_, self.resolution_, pc)
            if visib:
                visib_guards.append(node)
                visib_num += 1
                if visib_num > 2:
                    break

        return visib_guards
    
    def lineVisib(self, p1: Vector3d, p2: Vector3d, thresh: float, pc: Vector3d, caster_id: int = 0):
        ray_pt = Vector3d()
        pt_id = Vector3i()
        dist = 0.0

        # Convert start and end points to grid coordinates
        start = p1 / self.resolution_
        end = p2 / self.resolution_
        self.casters_[caster_id].setInput(start, end)

        st, ray_pt = self.casters_[caster_id].step(ray_pt)
        while st:
            pt_id[0] = int(ray_pt.x) + self.offset_[0] 
            pt_id[1] = int(ray_pt.y) + self.offset_[1]
            pt_id[2] = int(ray_pt.z) + self.offset_[2]
            dist = self.edt_enviroment_.sdf_map_.getDistance(pt_id)
            if dist <= thresh:
                pc = self.edt_enviroment_.sdf_map_.indexToPos(pt_id)
                return False, pc
            st, ray_pt = self.casters_[caster_id].step(ray_pt)

        return True, pc
    
    def needConnection(self, g1: GraphNode, g2: GraphNode, pt: Vector3d):
        # Create paths through pt and through existing connections
        path1 = [g1.pos_, pt, g2.pos_]
        path2 = [g1.pos_, None, g2.pos_]

        connetc_pts = []
        has_connect = False
        for i, n1 in enumerate(g1.neighbors_):
            for j, n2 in enumerate(g2.neighbors_):
                if n1.id_ == n2.id_:
                    # Found common neighbor - check if paths have same topology
                    path2[1] = n1.pos_
                    same_topo = self.sameTopoPath(path1, path2, 0.0)
                    if same_topo:
                        # If new path through pt is shorter, update existing connection
                        if self.pathLength(path1) < self.pathLength(path2):
                            g1.neighbors_[i].pos_ = pt
                        return False
        return True
        
    def getSample(self):

        pt = Vector3d(
            self.rand_pos_() * self.sample_r_[0],
            self.rand_pos_() * self.sample_r_[1], 
            self.rand_pos_() * self.sample_r_[2]
        )

        # Rotate and translate the point
        rotated = Vector3d(
            self.rotation_[0,0] * pt.x + self.rotation_[0,1] * pt.y + self.rotation_[0,2] * pt.z,
            self.rotation_[1,0] * pt.x + self.rotation_[1,1] * pt.y + self.rotation_[1,2] * pt.z,
            self.rotation_[2,0] * pt.x + self.rotation_[2,1] * pt.y + self.rotation_[2,2] * pt.z
        )

        return rotated + self.translation_
    
    def sameTopoPath(self, path1: List[Vector3d], path2: List[Vector3d], thresh: float):

        len1 = self.pathLength(path1)
        len2 = self.pathLength(path2)

        max_len = max(len1, len2)
        pt_num = int(np.ceil(max_len / self.resolution_))

        pts1 = self.discretizePath(path1, pt_num)
        pts2 = self.discretizePath(path2, pt_num)

        pc = Vector3d()
        for i in range(pt_num):
            if not self.lineVisib(pts1[i], pts2[i], thresh, pc):
                return False

        return True

    def pathLength(self, path: List[Vector3d]):
        length = 0.0
        if len(path) < 2:
            return length
        for i in range(1, len(path)):
            length += (path[i] - path[i-1]).norm()
        return length
    
    def discretizePath(self, path: List[Vector3d], pt_num: int = -1) -> List[Vector3d]:

        if pt_num == -1:
            return self.discretizePath1(path)
        
        # Initialize length list with 0.0 for first point
        len_list = [0.0]

        # Calculate cumulative lengths along path
        for i in range(len(path)-1):
            inc_l = (path[i+1] - path[i]).norm()
            len_list.append(inc_l + len_list[i])

        # Calculate total length and segment length
        len_total = len_list[-1]
        dl = len_total / (pt_num - 1)

        # Initialize discretized path
        dis_path = []

        # Generate pt_num evenly spaced points
        for i in range(pt_num):
            cur_l = float(i) * dl

            # Find which segment cur_l falls in
            idx = -1
            for j in range(len(len_list)-1):
                if cur_l >= len_list[j] - 1e-4 and cur_l <= len_list[j+1] + 1e-4:
                    idx = j
                    break

            # Interpolate point
            lambda_val = (cur_l - len_list[idx]) / (len_list[idx+1] - len_list[idx])
            inter_pt = Vector3d(
                path[idx].x * (1 - lambda_val) + path[idx+1].x * lambda_val,
                path[idx].y * (1 - lambda_val) + path[idx+1].y * lambda_val,
                path[idx].z * (1 - lambda_val) + path[idx+1].z * lambda_val
            )
            dis_path.append(inter_pt)

        return dis_path

    def discretizePath1(self, path: List[Vector3d]):
        dis_path, segment = [], []
        if len(path) < 2:
            return dis_path
        for i in range(len(path)-1):
            segment = self.discretizeLine(path[i], path[i+1])
            if len(segment) < 1:
                continue
            dis_path.extend(segment)
            if i != len(path) - 2:
                dis_path.pop(-1)
        return dis_path
    
    def discretizeLine(self, p1: Vector3d, p2: Vector3d):
        dir = p2 - p1
        len = dir.norm()
        seg_num = int(np.ceil(len / self.resolution_))
        line_pts = []
        if seg_num <= 0:
            return line_pts
        for i in range(seg_num):
            line_pts.append(p1 + dir * (float(i) / seg_num))
        return line_pts

    
    def pruneGraph(self):
        if len(self.graph_) > 2:
            # Iterate through nodes in graph
            i = 0
            while i < len(self.graph_) and len(self.graph_) > 2:
                node = self.graph_[i]
                # Skip start and end nodes
                if node.id_ <= 1:
                    i += 1
                    continue
                # Remove nodes with 1 or fewer neighbors
                if len(node.neighbors_) <= 1:
                    # Remove this node from others' neighbor lists
                    for other_node in self.graph_:
                        for j, n in enumerate(other_node.neighbors_):
                            if n.id_ == node.id_:
                                other_node.neighbors_.pop(j)
                    self.graph_.pop(i)
                    i = 0  # Restart checking from beginning
                else:
                    i += 1

    def searchPaths(self):

        self.raw_paths_.clear()

        visited = [self.graph_[0]]
        visited = self.depthFirstSearch(visited)

        # Sort paths by node number
        min_node_num = 100000
        max_node_num = 1
        path_list = [[] for _ in range(100)]
        
        for i in range(len(self.raw_paths_)):
            path_size = len(self.raw_paths_[i])
            max_node_num = max(max_node_num, path_size)
            min_node_num = min(min_node_num, path_size)
            path_list[path_size].append(i)

        # Select paths with fewer nodes
        filter_raw_paths = []
        for i in range(min_node_num, max_node_num + 1):
            reach_max = False
            for j in range(len(path_list[i])):
                filter_raw_paths.append(self.raw_paths_[path_list[i][j]])
                if len(filter_raw_paths) >= self.max_raw_path2_:
                    reach_max = True
                    break
            if reach_max:
                break

        self.raw_paths_ = filter_raw_paths

        return self.raw_paths_
    
    def depthFirstSearch(self, visited: List[GraphNode]):

        cur = visited[-1]

        # Check if any neighbors reach goal
        for neighbor in cur.neighbors_:
            if neighbor.id_ == 1:
                # Add this path to paths set
                path = []

                for node in visited:
                    path.append(node.pos_)
                path.append(neighbor.pos_)

                self.raw_paths_.append(path)
                if len(self.raw_paths_) >= self.max_raw_path_:
                    return visited

                break

        # Recursively search through other neighbors
        for neighbor in cur.neighbors_:
            # Skip if reaching goal
            if neighbor.id_ == 1:
                continue

            # Skip if already visited
            revisit = False
            for node in visited:
                if neighbor.id_ == node.id_:
                    revisit = True
                    break
            if revisit:
                continue

            # Recursive search
            visited.append(neighbor)
            visited = self.depthFirstSearch(visited)
            if len(self.raw_paths_) >= self.max_raw_path_:
                return visited

            visited.pop()

        return visited
    
    def shortcutPaths(self):

        # Resize short_paths_ to match raw_paths_
        self.short_paths_ = [[] for _ in range(len(self.raw_paths_))]

        if self.parallel_shortcut_:
            # Not sure about the effect of parallel shortcutting
            import threading
            threads = []
            for i in range(len(self.raw_paths_)):
                thread = threading.Thread(target=self.shortcutPath, args=(self.raw_paths_[i], i, 1))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        else:
            # Sequential shortcutting
            for i in range(len(self.raw_paths_)):
                self.shortcutPath(self.raw_paths_[i], i)

    def shortcutPath(self, path: List[Vector3d], path_id: int, iter_num: int = 1):

        short_path = path.copy()
        last_path = []
        for k in range(iter_num):
            last_path = short_path.copy()
            dis_path = self.discretizePath(short_path)
            if len(dis_path) < 2:
                self.short_paths_[path_id] = dis_path
                return
            colli_pt, grad, dir, push_dir = Vector3d(), Vector3d(), Vector3d(), Vector3d()
            dist = 0.0
            short_path.clear()
            short_path.append(dis_path[0])
            for i in range(1, len(dis_path)):
                if self.lineVisib(short_path[-1], dis_path[i], self.resolution_, colli_pt, path_id):
                    continue
                dist, grad = self.edt_enviroment_.evaluateEDTWithGrad(colli_pt, -1, dist, grad)
                if grad.norm() > 1e-3:
                    grad = grad.normalized()
                    dir = (dis_path[i] - short_path[-1]).normalized()
                    push_dir = grad - dir * (grad * dir)
                    push_dir = push_dir.normalized()
                    colli_pt = colli_pt + self.resolution_ * push_dir
                short_path.append(colli_pt)
            short_path.append(dis_path[-1])

            len1 = self.pathLength(last_path)
            len2 = self.pathLength(short_path)
            if len2 > len1:
                short_path = last_path
                break
        
        self.short_paths_[path_id] = short_path

    def pruneEquivalent(self, paths: List[List[Vector3d]]):

        pruned_paths = []
        if len(paths) < 1:
            return pruned_paths
        
        exist_paths_id = [0]
        for i in range(1, len(paths)):
            new_path = True

            for j in range(len(exist_paths_id)):
                same_topo = self.sameTopoPath(paths[i], paths[exist_paths_id[j]], 0.0)

                if same_topo:
                    new_path = False
                    break

            if new_path:
                exist_paths_id.append(i)

        for i in range(len(exist_paths_id)):
            pruned_paths.append(paths[exist_paths_id[i]])

        return pruned_paths
    
    def selectShortPaths(self, paths: List[List[Vector3d]], num: int):

        short_paths = []
        short_path = []
        min_len = 0.1

        for i in range(self.reserve_num_):
            if len(paths) == 0:
                break

            path_id = self.shortestPath(paths)
            if i == 0:
                short_paths.append(paths[path_id])
                min_len = self.pathLength(paths[path_id])
                paths.pop(path_id)
            else:
                rat = self.pathLength(paths[path_id]) / min_len
                if rat < self.ratio_to_short_:
                    short_paths.append(paths[path_id])
                    paths.pop(path_id)
                else:
                    break

        for i in range(len(short_paths)):
            short_paths[i] = self.start_pts_ + short_paths[i] + self.end_pts_

        for i in range(len(short_paths)):
            self.shortcutPath(short_paths[i], i, 5)
            short_paths[i] = self.short_paths_[i]

        short_paths = self.pruneEquivalent(short_paths)

        return short_paths
    
    def shortestPath(self, paths: List[List[Vector3d]]):

        short_id = -1
        min_len = 1e8
        for i in range(len(paths)):
            length = self.pathLength(paths[i])
            if length < min_len:
                min_len = length
                short_id = i

        return short_id

    def pathToGuidePts(self, path: List[Vector3d], pt_num: int, pts: List[Vector3d]):

        pts = self.discretizePath(path, pt_num)

        return pts


from typing import List
import heapq
from ..utils import Vector3d, Vector3i
from .graph_node import ViewNode

class GraphSearch:
    def __init__(self):
        self.node_num_ = 0
        self.edge_num_ = 0
        self.nodes_ = []
    
    def addNode(self, node: ViewNode):
        self.nodes_.append(node)
        self.nodes_[-1].id_ = self.node_num_
        self.node_num_ += 1

    def addEdge(self, from_node: int, to_node: int):
        self.nodes_[from_node].neighbors_.append(to_node)
        self.edge_num_ += 1

    def DijkstraSearch(self, start: int, goal: int, path: List[ViewNode]):

        open_set = []
        start_v = self.nodes_[start]
        end_v = self.nodes_[goal]

        start_v.g_value_ = 0
        heapq.heappush(open_set, start_v)

        while open_set:
            vc = heapq.heappop(open_set)
            vc.closed_ = True

            if vc == end_v:
                vit = vc.copy()
                while vit.parent_ != None:
                    path.append(vit)
                    vit = vit.parent_
                path.reverse()
                return path
            
            for vb in vc.neighbors_:
                if vb.closed_:
                    continue
                g_tmp = vc.g_value_ + vc.costTo(vb)
                if g_tmp < vb.g_value_:
                    vb.g_value_ = g_tmp
                    vb.parent_ = vc
                    heapq.heappush(open_set, vb)
        
        return path



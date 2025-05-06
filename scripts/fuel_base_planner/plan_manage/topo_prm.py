from enum import Enum
from typing import List, Optional
from ..utils import Vector3d

class NodeType(Enum):
    Guard = 1
    Connector = 2

class NodeState(Enum):
    NEW = 1
    CLOSE = 2
    OPEN = 3

class GraphNode:
    def __init__(self, pos: Optional[Vector3d] = None, node_type: Optional[NodeType] = None, id: Optional[int] = None):

        self.pos_ = pos
        self.type_ = node_type
        self.state_ = NodeState.NEW
        self.id_ = id
        self.neighbors_: List[GraphNode] = []

from ..utils import Vector3d

class VisiblePair:
    def __init__(self):
        self.from_ : int = 0          # idx of qi, see qj
        self.to_ : int = 0            # idx of qj, can be seen by qi
        self.qb_ : Vector3d = Vector3d(0.0, 0.0, 0.0)    # cell blocking view from qi->qj

class ViewConstraint:
    def __init__(self):
        self.pt_ : Vector3d = Vector3d(0.0, 0.0, 0.0)     # unknown point along the traj
        self.pc_ : Vector3d = Vector3d(0.0, 0.0, 0.0)     # critical view point  
        self.dir_ : Vector3d = Vector3d(0.0, 0.0, 0.0)    # critical view direction with safe length
        self.pcons_ : Vector3d = Vector3d(0.0, 0.0, 0.0)  # pt to add view constraint
        self.idx_ : int = 0            # idx to add view constraint

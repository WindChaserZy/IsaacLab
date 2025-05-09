from ..bspline import NonUniformBspline
from ..poly_traj import PolynomialTraj
from ..utils import Vector3d
from ..searching.topo_prm import GraphNode
from ..perception import VisiblePair, ViewConstraint
from typing import List
import math
import numpy as np
import time

class GlobalTrajData:

    def __init__(self):

        self.global_traj_  = PolynomialTraj()
        self.local_traj_ : List[NonUniformBspline] = []
        self.global_duration_ : float = 0.0
        self.local_start_time_ : float = 0.0
        self.local_end_time_ : float = 0.0
        self.time_change_ : float = 0.0
        self.last_time_inc : float = 0.0
        self.global_start_time_ : float = 0.0
    
    def localTrajReachTarget(self) -> bool:

        return abs(self.local_end_time_ - self.global_duration_) < 1e-3
    
    def setGlobalTraj(self, traj: PolynomialTraj, global_start_time: float = 0.0):

        self.global_traj_ = traj
        self.global_duration_ = self.global_traj_.getTotalTime()
        self.global_start_time_ = global_start_time
        self.local_traj_.clear()
        self.local_start_time_ = -1
        self.local_end_time_ = -1
        self.time_change_ = 0.0
        self.last_time_inc = 0.0
    
    def setLocalTraj(self, traj: NonUniformBspline, local_ts: float, local_te: float, time_change: float):

        self.local_traj_ = []
        self.local_traj_.append(traj)
        self.local_traj_.append(self.local_traj_[0].getDerivative())
        self.local_traj_.append(self.local_traj_[1].getDerivative())

        self.local_start_time_ = local_ts
        self.local_end_time_ = local_te
        self.global_duration_ += time_change
        self.time_change_ += time_change
        self.last_time_inc = time_change

    def getState(self, t: float, k: int) -> Vector3d:

        if t >= -1e-3 and t <= self.local_start_time_:
            return self.global_traj_.evaluate(t - self.time_change_ + self.last_time_inc, k)
        elif t > self.local_start_time_ and t <= self.global_duration_ + 1e-3:
            return self.global_traj_.evaluate(t - self.time_change_, k)
        else:
            return self.local_traj_[k].evaluateDeBoor(t - self.local_start_time_)
        
    def getTrajInfoInDuration(self, start_t: float, duration: float, dt: float, point_set: List[Vector3d], start_end_derivative: List[Vector3d]):
        
        tp = 0.0
        while tp <= duration + 1e-4:
            cur_pt = self.getState(start_t + tp, 0)
            point_set.append(cur_pt)
            tp += dt
        start_end_derivative.append(self.getState(start_t, 1))
        start_end_derivative.append(self.getState(start_t + duration, 1))
        start_end_derivative.append(self.getState(start_t, 2))
        start_end_derivative.append(self.getState(start_t + duration, 2))
        return point_set, start_end_derivative
    
    def getTrajInfoInSphere(self, start_t: float, radius: float, dist_pt: float, point_set: List[Vector3d], start_end_derivative: List[Vector3d], dt: float, duration: float):

        segment_len = 0.0
        segment_time = 0.0
        first_pt = self.getState(start_t, 0)
        prev_pt = first_pt
        cur_pt = first_pt

        # Search the time at which distance to current position is larger than a threshold
        delta_t = 0.2
        while (cur_pt - first_pt).norm() < radius and start_t + segment_time < self.global_duration_ - 1e-3:
            segment_time = min(segment_time + delta_t, self.global_duration_ - start_t)
            cur_pt = self.getState(start_t + segment_time, 0)
            segment_len += (cur_pt - prev_pt).norm()
            prev_pt = cur_pt

        # Get sampled state of the segment
        seg_num = math.floor(segment_len / dist_pt)
        seg_num = max(6, seg_num)
        duration = segment_time
        dt = duration / seg_num
        point_set, start_end_derivative = self.getTrajInfoInDuration(start_t, duration, dt, point_set, start_end_derivative)

        return point_set, start_end_derivative
    
class PlanParameters:
    
    def __init__(self):
        # Physical limits
        self.max_vel_ = 0.0
        self.max_acc_ = 0.0 
        self.max_jerk_ = 0.0
        self.accept_vel_ = 0.0
        self.accept_acc_ = 0.0

        # Yaw control
        self.max_yawdot_ = 0.0

        # Trajectory parameters
        self.local_traj_len_ = 0.0  # local replanning trajectory length
        self.ctrl_pt_dist = 0.0     # distance between adjacient B-spline control points
        self.bspline_degree_ = 0
        self.min_time_ = False

        # Safety parameters
        self.clearance_ = 0.0
        self.dynamic_ = 0

        # Processing time tracking
        self.time_search_ = 0.0
        self.time_optimize_ = 0.0
        self.time_adjust_ = 0.0

class LocalTrajData:

    def __init__(self):
        self.traj_id_ = 0
        self.duration_ = 0.0
        self.start_time_ = 0.0
        self.start_pos_ = Vector3d(0.0, 0.0, 0.0)
        self.position_traj_ = NonUniformBspline()
        self.velocity_traj_ = NonUniformBspline()
        self.acceleration_traj_ = NonUniformBspline()
        self.yaw_traj_ = NonUniformBspline()
        self.yawdot_traj_ = NonUniformBspline()
        self.yawdotdot_traj_ = NonUniformBspline()

class LocalTrajState:

    def __init__(self):
        self.pos_ = Vector3d(0.0, 0.0, 0.0)
        self.vel_ = Vector3d(0.0, 0.0, 0.0)
        self.acc_ = Vector3d(0.0, 0.0, 0.0)
        self.yaw_ = 0.0
        self.yawdot_ = 0.0
        self.id_ = 0

class LocalTrajServer:

    def __init__(self):

        self.traj1_ = LocalTrajData()
        self.traj2_ = LocalTrajData()
        self.traj1_.traj_id_ = 0
        self.traj2_.traj_id_ = 0

    def addTraj(self, traj: LocalTrajData):

        if self.traj1_.traj_id_ == 0:
            self.traj1_ = traj
        else :
            self.traj2_ = traj

    def evaluate(self, time: float, traj_state: LocalTrajState):

        if self.traj1_.traj_id_ == 0:
            return False, traj_state
        
        if self.traj2_.traj_id_ != 0 and time > self.traj2_.start_time_:
            self.traj1_ = self.traj2_
            self.traj2_.traj_id_ = 0

            t_cur = time - self.traj1_.start_time_
            if t_cur < 0:
                print("[Traj server]: invalid time.")
                return False, traj_state
            elif t_cur < self.traj1_.duration_:
                # time within range of traj 1
                traj_state.pos_ = self.traj1_.position_traj_.evaluateDeBoorT(t_cur)
                traj_state.vel_ = self.traj1_.velocity_traj_.evaluateDeBoorT(t_cur)
                traj_state.acc_ = self.traj1_.acceleration_traj_.evaluateDeBoorT(t_cur)
                traj_state.yaw_ = self.traj1_.yaw_traj_.evaluateDeBoorT(t_cur)[0]
                traj_state.yawdot_ = self.traj1_.yawdot_traj_.evaluateDeBoorT(t_cur)[0]
                traj_state.id_ = self.traj1_.traj_id_
                return True, traj_state     
            else:
                traj_state.pos_ = self.traj1_.position_traj_.evaluateDeBoorT(self.traj1_.duration_)
                traj_state.vel_ = Vector3d(0.0, 0.0, 0.0)
                traj_state.acc_ = Vector3d(0.0, 0.0, 0.0)
                traj_state.yaw_ = self.traj1_.yaw_traj_.evaluateDeBoorT(self.traj1_.duration_)[0]
                traj_state.yawdot_ = 0.0
                traj_state.id_ = self.traj1_.traj_id_
                return True, traj_state
    
    ##### To be fixed
    def resetDuration(self):

        if self.traj1_.traj_id_ != 0:
            t_stop = self.traj1_.duration_
            self.traj1_.duration_ = min(t_stop, self.traj1_.duration_)
            
        if self.traj2_.traj_id_ != 0:
            t_stop = self.traj2_.duration_
            self.traj2_.duration_ = min(t_stop, self.traj2_.duration_)
        
class MidPlanData:

    def __init__(self):
        # Global waypoints
        self.global_waypoints_ : List[Vector3d] = []

        # Initial trajectory segment
        self.initial_local_segment_ :NonUniformBspline = NonUniformBspline()
        self.local_start_end_derivative_ :List[Vector3d] = []

        # Kinodynamic path
        self.kino_path_ :List[Vector3d] = []

        # Topological paths
        self.topo_graph_ :List[GraphNode] = []
        self.topo_paths_ :List[List[Vector3d]]= []
        self.topo_filtered_paths_ :List[List[Vector3d]]= []
        self.topo_select_paths_ :List[List[Vector3d]]= []

        # Multiple topological trajectories
        self.topo_traj_pos1_ :List[NonUniformBspline] = []
        self.topo_traj_pos2_ :List[NonUniformBspline] = []
        self.refines_ :List[NonUniformBspline] = []

        # Visibility constraint
        self.block_pts_ :List[Vector3d] = []
        self.ctrl_pts_ :np.ndarray = np.array([])
        self.no_visib_traj_ :NonUniformBspline = NonUniformBspline()
        self.visib_pairs_ :List[VisiblePair] = []
        self.view_cons_ :ViewConstraint = ViewConstraint()

        # Heading planning
        self.frontiers_ :List[List[Vector3d]] = []
        self.path_yaw_ :List[float] = []
        self.dt_yaw_ :float = 0.0
        self.dt_yaw_path_ :float = 0.0

    def clearTopoPaths(self):
        self.topo_traj_pos1_.clear()
        self.topo_traj_pos2_.clear()
        self.topo_graph_.clear()
        self.topo_paths_.clear()
        self.topo_filtered_paths_.clear()
        self.topo_select_paths_.clear()

    def addTopoPath(self, graph: List[GraphNode], paths: List[List[Vector3d]], filtered_paths: List[List[Vector3d]], select_paths: List[List[Vector3d]]):
        self.topo_graph_ = graph
        self.topo_paths_ = paths
        self.topo_filtered_paths_ = filtered_paths
        self.topo_select_paths_ = select_paths
        


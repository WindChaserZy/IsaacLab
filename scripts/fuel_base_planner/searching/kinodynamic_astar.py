from typing import Optional, List
import numpy as np
import heapq
import math
from ..utils import Vector3d, Vector3i
from ..env import EDTEnv, SDFmap

inf = 1 >> 30
IN_CLOSE_SET = 'a'
IN_OPEN_SET = 'b'
NOT_EXPAND = 'c'

class PathNode:
    """Node class for kinodynamic A* path planning"""
    
    def __init__(self):
        self.index = Vector3i()  # 3D index
        self.state = np.zeros(6)  # 6D state vector
        self.g_score = 0.0  # Cost from start
        self.f_score = 0.0  # Total estimated cost
        self.input = Vector3d()  # Control input
        self.duration = 0.0  # Duration of motion
        self.time = 0.0  # Time stamp
        self.time_idx = 0  # Time index
        self.parent: Optional['PathNode'] = None  # Parent node
        self.node_state = NOT_EXPAND  # State of node expansion
    
    def __lt__(self, other: 'PathNode') -> bool:
        return self.f_score < other.f_score
    
    def __le__(self, other: 'PathNode') -> bool:
        return self.f_score <= other.f_score
    
class NodeHashTable:
    """Hash table for storing path nodes"""
    
    def __init__(self):
        # Hash tables for 3D and 4D indices
        self.data_3d = {}  # Maps Vector3i -> PathNode
        self.data_4d = {}  # Maps (Vector3i, time_idx) -> PathNode
        
    def insert(self, idx: Vector3i, node: 'PathNode', time_idx: Optional[int] = None):
        """Insert a node into the hash table"""
        if time_idx is None:
            # 3D insert
            self.data_3d[idx] = node
        else:
            # 4D insert with time index
            key = (idx, time_idx)
            self.data_4d[key] = node
            
    def find(self, idx: Vector3i, time_idx: Optional[int] = None) -> Optional['PathNode']:
        """Find a node in the hash table"""
        if time_idx is None:
            # 3D lookup
            return self.data_3d.get(idx)
        else:
            # 4D lookup with time index
            key = (idx, time_idx)
            return self.data_4d.get(key)
            
    def clear(self):
        """Clear the hash table"""
        self.data_3d.clear()
        self.data_4d.clear()

class KinodynamicAstar:
    """Kinodynamic A* path planner"""
    REACH_HORIZON = 1
    REACH_END = 2
    NO_PATH = 3
    NEAR_END = 4
    
    def __init__(self):
        self.setParam()
        self.use_node_num_ = 0
        self.iter_num_ = 0
        self.expand_nodes_ = NodeHashTable()
        self.open_set_ = []
        self.path_nodes_ : List[PathNode] = []
        self.start_vel_ = Vector3d()
        self.end_vel_ = Vector3d()
        self.start_acc_ = Vector3d()
        self.phi_ = np.eye(6)
        self.is_shot_succ_ = False
        self.coef_shot_ :np.ndarray = np.array([])
        self.t_shot_ = 0.0
        self.has_path_ = False
        self.time_origin_ = 0.0

    def setParam(self):
        ########## Config ##########
        self.max_tau_ = -1.0
        self.init_max_tau_ = -1.0
        self.max_vel_ = -1.0
        self.max_acc_ = -1.0
        self.w_time_ = -1.0
        self.horizon_ = -1.0
        self.resolution_ = -1.0
        self.time_resolution_ = -1.0
        self.lambda_heu_ = -1.0
        self.allocate_num_ = -1
        self.check_num_ = -1
        self.optimistic_ = True
        self.tie_breaker_ = 1.0 + 1.0 / 10000
        vel_mergin = 0.0
        self.max_vel_ += vel_mergin
        ####################################
        self.inv_resolution_ = 1.0 / self.resolution_
        self.inv_time_resolution_ = 1.0 / self.time_resolution_
        self.path_node_pool_ = [PathNode() for _ in range(self.allocate_num_)]

    def setEnvironment(self, env: EDTEnv):
        self.edt_environment_ = env
        self.origin_, self.map_size_3d_ = self.edt_environment_.sdf_map_.getRegion()

    def reset(self):

        self.expand_nodes_.clear()
        self.path_nodes_.clear()
        self.open_set_.clear()

        for i in range(self.use_node_num_):
            self.path_node_pool_[i].parent = None
            self.path_node_pool_[i].node_state = NOT_EXPAND

        self.use_node_num_ = 0
        self.iter_num_ = 0
        self.has_path_ = False
        self.is_shot_succ_ = False

    def retrievePath(self, end_node: PathNode):

        cur_node = end_node
        self.path_nodes_.append(cur_node)

        while cur_node.parent is not None:
            cur_node = cur_node.parent
            self.path_nodes_.append(cur_node)
        
        self.path_nodes_.reverse()
    
    def posToIndex(self, pt: Vector3d) -> Vector3i:
        idx = Vector3i()
        for i in range(3):
            idx[i] = int(math.floor((pt[i] - self.origin_[i]) * self.inv_resolution_))
        return idx
    
    def timeToIndex(self, time: float) -> int:
        return int(math.floor((time - self.time_origin_) * self.inv_time_resolution_))

    def search(self, start_pt: Vector3d, start_v: Vector3d, start_a: Vector3d, end_pt: Vector3d, end_v: Vector3d, init: bool, dynamic: bool = False, time_start: float = -1.0):

        self.start_vel_ = start_v
        self.start_acc_ = start_a

        cur_node = self.path_node_pool_[0]
        cur_node.parent = None
        cur_node.state[0:3] = np.array(start_pt.listify())
        cur_node.state[3:6] = np.array(start_v.listify())
        cur_node.g_score = 0.0
        cur_node.index = self.posToIndex(start_pt)

        end_state = np.zeros(6)
        time_to_goal = 0.0
        end_state[0:3] = np.array(end_pt.listify())
        end_state[3:6] = np.array(end_v.listify())
        end_index = self.posToIndex(end_pt)
        cur_node.f_score = self.lambda_heu_ * self.estimateHeuristic(cur_node.state, end_state, time_to_goal)[0]
        cur_node.node_state = IN_OPEN_SET
        heapq.heappush(self.open_set_, cur_node)
        self.use_node_num_ += 1

        if dynamic:
            self.time_origin_ = time_start
            cur_node.time = time_start
            cur_node.time_idx = self.timeToIndex(time_start)
            self.expand_nodes_.insert(cur_node.index, cur_node, cur_node.time_idx)
        else:
            self.expand_nodes_.insert(cur_node.index, cur_node)
        
        neighbor = None
        terminate_node = None
        init_search = init
        tolerance = int(np.ceil(1 / self.resolution_))

        while len(self.open_set_) > 0:
            cur_node = self.open_set_[0]
            # Check if we've reached horizon or are near end point
            reach_horizon = (Vector3d(cur_node.state[0], cur_node.state[1], cur_node.state[2]) - start_pt).norm() >= self.horizon_
            near_end = (abs(cur_node.index.x - end_index.x) <= tolerance and
                       abs(cur_node.index.y - end_index.y) <= tolerance and 
                       abs(cur_node.index.z - end_index.z) <= tolerance)

            if reach_horizon or near_end:
                terminate_node = cur_node
                self.retrievePath(terminate_node)
                if near_end:
                    # Check if shot trajectory exists
                    _, time_to_goal = self.estimateHeuristic(cur_node.state, end_state, time_to_goal)
                    self.computeShotTraj(cur_node.state, end_state, time_to_goal)

            if reach_horizon:
                if self.is_shot_succ_:
                    return self.REACH_END
                else:
                    return self.REACH_HORIZON
            
            if near_end:
                if self.is_shot_succ_:
                    return self.REACH_END
                elif cur_node.parent is not None:
                    return self.NEAR_END
                else:
                    return self.NO_PATH

            heapq.heappop(self.open_set_)
            cur_node.node_state = IN_CLOSE_SET
            self.iter_num_ += 1

            res = 1 / 2.0
            time_res = 1 / 1.0
            time_res_init = 1 / 20.0
            cur_state = cur_node.state
            tmp_expand_nodes = []
            inputs = []
            durations = []

            if init_search:
                inputs.append(self.start_acc_)
                tau = time_res_init * self.init_max_tau_
                while tau <= self.init_max_tau_ + 1e-3:
                    durations.append(tau)
                    tau += time_res_init * self.init_max_tau_
                init_search = False
            else:
                ax = -self.max_acc_
                while ax <= self.max_acc_ + 1e-3:
                    ay = -self.max_acc_
                    while ay <= self.max_acc_ + 1e-3:
                        az = -self.max_acc_
                        while az <= self.max_acc_ + 1e-3:
                            inputs.append(Vector3d(ax, ay, az))
                            az += self.max_acc_ * res
                        ay += self.max_acc_ * res
                    ax += self.max_acc_ * res

                tau = time_res * self.max_tau_
                while tau <= self.max_tau_:
                    durations.append(tau)
                    tau += time_res * self.max_tau_
            
            for i in range(len(inputs)):
                for j in range(len(durations)):
                    um = inputs[i]
                    tau = durations[j]
                    pro_state = self.stateTransit(cur_state, um, tau)
                    pro_t = cur_node.time + tau

                    # Check inside map range
                    pro_pos = Vector3d(pro_state[0], pro_state[1], pro_state[2])
                    if not self.edt_environment_.sdf_map_.isInMap(pro_pos):
                        if init_search:
                            print("box")
                        continue

                    # Check if in close set
                    pro_id = self.posToIndex(pro_pos)
                    pro_t_id = self.timeToIndex(pro_t)
                    pro_node = self.expand_nodes_.find(pro_id, pro_t_id) if dynamic else self.expand_nodes_.find(pro_id)
                    if pro_node is not None and pro_node.node_state == IN_CLOSE_SET:
                        if init_search:
                            print("close") 
                        continue

                    # Check maximal velocity
                    pro_v = pro_state[3:6]
                    if abs(pro_v[0]) > self.max_vel_ or abs(pro_v[1]) > self.max_vel_ or abs(pro_v[2]) > self.max_vel_:
                        if init_search:
                            print("vel")
                        continue

                    # Check not in same voxel
                    diff = pro_id - cur_node.index
                    diff_time = pro_t_id - cur_node.time_idx
                    if diff.norm() == 0 and (not dynamic or diff_time == 0):
                        if init_search:
                            print("same")
                        continue

                    # Check safety
                    pos = Vector3d()
                    xt = np.zeros(6)
                    is_occ = False
                    for k in range(1, self.check_num_ + 1):
                        dt = tau * float(k) / float(self.check_num_)
                        xt = self.stateTransit(cur_state, um, dt)
                        pos = Vector3d(xt[0], xt[1], xt[2])
                        if self.edt_environment_.sdf_map_.getInflatedOccupancy(pos) == 1:
                            is_occ = True
                            break
                        if not self.optimistic_ and self.edt_environment_.sdf_map_.getOccupancy(pos) == SDFmap.UNKNOWN:
                            is_occ = True
                            break
                    
                    if is_occ:
                        if init_search:
                            print("safe")
                        continue

                    # Calculate scores
                    time_to_goal = 0.0
                    tmp_g_score = (um.norm2() + self.w_time_) * tau + cur_node.g_score
                    tmp_f_score = tmp_g_score + self.lambda_heu_ * self.estimateHeuristic(pro_state, end_state, time_to_goal)[0]

                    # Compare nodes expanded from same parent
                    prune = False
                    for expand_node in tmp_expand_nodes:
                        if (pro_id - expand_node.index).norm() == 0 and (not dynamic or pro_t_id == expand_node.time_idx):
                            prune = True
                            if tmp_f_score < expand_node.f_score:
                                expand_node.f_score = tmp_f_score
                                expand_node.g_score = tmp_g_score
                                expand_node.state = pro_state
                                expand_node.input = um
                                expand_node.duration = tau
                                if dynamic:
                                    expand_node.time = cur_node.time + tau
                            break

                    # Handle new node
                    if not prune:
                        if pro_node is None:
                            pro_node = self.path_node_pool_[self.use_node_num_]
                            pro_node.index = pro_id
                            pro_node.state = pro_state
                            pro_node.f_score = tmp_f_score
                            pro_node.g_score = tmp_g_score
                            pro_node.input = um
                            pro_node.duration = tau
                            pro_node.parent = cur_node
                            pro_node.node_state = IN_OPEN_SET
                            if dynamic:
                                pro_node.time = cur_node.time + tau
                                pro_node.time_idx = self.timeToIndex(pro_node.time)
                            
                            heapq.heappush(self.open_set_, pro_node)

                            if dynamic:
                                self.expand_nodes_.insert(pro_id, pro_node, pro_node.time_idx)
                            else:
                                self.expand_nodes_.insert(pro_id, pro_node)

                            tmp_expand_nodes.append(pro_node)

                            self.use_node_num_ += 1
                            if self.use_node_num_ == self.allocate_num_:
                                print("run out of memory.")
                                return self.NO_PATH

                        elif pro_node.node_state == IN_OPEN_SET:
                            if tmp_g_score < pro_node.g_score:
                                pro_node.state = pro_state
                                pro_node.f_score = tmp_f_score
                                pro_node.g_score = tmp_g_score
                                pro_node.input = um
                                pro_node.duration = tau
                                pro_node.parent = cur_node
                                if dynamic:
                                    pro_node.time = cur_node.time + tau
                        else:
                            print(f"error type in searching: {pro_node.node_state}")
        print(f"use node num: {self.use_node_num_}")
        print(f"iter num: {self.iter_num_}")
        return self.NO_PATH

    def estimateHeuristic(self, x1: np.ndarray, x2: np.ndarray, optimal_time: float):

        # Get position difference vector
        dp = x2[0:3] - x1[0:3]

        # Get velocity vectors
        v0 = x1[3:6]
        v1 = x2[3:6]

        # Calculate coefficients
        c1 = -36 * np.dot(dp, dp) # Using Vector3d's __mul__ for dot product
        c2 = 24 * np.dot(v0 + v1, dp)
        c3 = -4 * (np.dot(v0, v0) + np.dot(v0, v1) + np.dot(v1, v1))
        c4 = 0
        c5 = self.w_time_

        # Get quartic solutions
        ts = self.quartic(c5, c4, c3, c2, c1)

        # Calculate t_bar using max velocity
        v_max = self.max_vel_ * 0.5
        
        # Get max component of position difference
        diff = [abs(x2[i] - x1[i]) for i in range(3)]
        t_bar = max(diff) / v_max
        ts.append(t_bar)

        # Find minimum cost and corresponding time
        cost = float('inf')
        t_d = t_bar

        for t in ts:
            if t < t_bar:
                continue
            c = -c1 / (3 * t * t * t) - c2 / (2 * t * t) - c3 / t + self.w_time_ * t
            if c < cost:
                cost = c
                t_d = t

        optimal_time = t_d

        return 1.0 * (1 + self.tie_breaker_) * cost, optimal_time
    
    def computeShotTraj(self, state1: np.ndarray, state2: np.ndarray, time_to_goal: float):

        # Get position and velocity components
        p0 = state1[0:3]
        v0 = state1[3:6]
        p1 = state2[0:3]
        v1 = state2[3:6]
        
        dp = p1 - p0
        dv = v1 - v0
        t_d = time_to_goal

        # Calculate polynomial coefficients
        a = (1.0/6.0) * (-12.0/(t_d**3) * (dp - v0 * t_d) + 6.0/(t_d**2) * dv)
        b = 0.5 * (6.0/(t_d**2) * (dp - v0 * t_d) - 2.0/t_d * dv)
        c = v0
        d = p0

        # Store coefficients in matrix form
        coef = np.zeros((3, 4))
        coef[:,3] = a.flatten()
        coef[:,2] = b.flatten() 
        coef[:,1] = c.flatten()
        coef[:,0] = d.flatten()

        # Time derivative matrix
        Tm = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]])

        # Forward check trajectory
        t_delta = t_d / 10
        time = t_delta
        while time <= t_d:
            # Time vector
            t = np.array([time**j for j in range(4)])

            # Calculate position, velocity and acceleration
            coord = Vector3d()
            vel = Vector3d()
            acc = Vector3d()

            for dim in range(3):
                poly1d = coef[dim]
                coord[dim] = np.dot(poly1d, t)
                vel[dim] = np.dot(np.dot(Tm, poly1d), t)
                acc[dim] = np.dot(np.dot(Tm @ Tm, poly1d), t)

                # Check dynamic feasibility
                if abs(vel[dim]) > self.max_vel_ or abs(acc[dim]) > self.max_acc_:
                    pass

            # Check if trajectory goes out of map bounds
            if (coord.x < self.origin_[0] or coord.x >= self.map_size_3d_[0] or
                coord.y < self.origin_[1] or coord.y >= self.map_size_3d_[1] or
                coord.z < self.origin_[2] or coord.z >= self.map_size_3d_[2]):
                return False

            # Check collision
            if self.edt_environment_.sdf_map_.getInflatedOccupancy(coord) == 1:
                return False

            time += t_delta

        # Store successful trajectory
        self.coef_shot_ = coef
        self.t_shot_ = t_d
        self.is_shot_succ_ = True
        return True
    
    def quartic(self, a: float, b: float, c: float, d: float, e: float) -> List[float]:
        
        # Initialize empty list for solutions
        dts = []

        # Normalize coefficients by dividing by a
        a3 = b / a
        a2 = c / a 
        a1 = d / a
        a0 = e / a

        # Get cubic solution
        y1 = self.cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0)[0]
        
        # Calculate r value
        r = a3 * a3 / 4 - a2 + y1
        if r < 0:
            return dts

        # Calculate R value
        R = math.sqrt(r)
        
        # Calculate D and E values
        if R != 0:
            D = math.sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 
                         0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R)
            E = math.sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 
                         0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R)
        else:
            D = math.sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * math.sqrt(y1 * y1 - 4 * a0))
            E = math.sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * math.sqrt(y1 * y1 - 4 * a0))

        # Add solutions if they are real numbers
        if not math.isnan(D):
            dts.append(-a3 / 4 + R / 2 + D / 2)
            dts.append(-a3 / 4 + R / 2 - D / 2)
        if not math.isnan(E):
            dts.append(-a3 / 4 - R / 2 + E / 2)
            dts.append(-a3 / 4 - R / 2 - E / 2)

        return dts
    
    def cubic(self, a: float, b: float, c: float, d: float) -> List[float]:
        
        # Initialize empty list for solutions
        dts = []

        # Normalize coefficients
        a2 = b / a
        a1 = c / a
        a0 = d / a

        # Calculate discriminant
        Q = (3 * a1 - a2 * a2) / 9
        R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54
        D = Q * Q * Q + R * R

        if D > 0:
            # One real root
            S = self.cbrt(R + math.sqrt(D))
            T = self.cbrt(R - math.sqrt(D))
            dts.append(-a2 / 3 + (S + T))
        elif D == 0:
            # Three real roots, at least two equal
            S = self.cbrt(R)
            dts.append(-a2 / 3 + S + S)
            dts.append(-a2 / 3 - S)
        else:
            # Three distinct real roots
            theta = math.acos(R / math.sqrt(-Q * Q * Q))
            dts.append(2 * math.sqrt(-Q) * math.cos(theta / 3) - a2 / 3)
            dts.append(2 * math.sqrt(-Q) * math.cos((theta + 2 * math.pi) / 3) - a2 / 3)
            dts.append(2 * math.sqrt(-Q) * math.cos((theta + 4 * math.pi) / 3) - a2 / 3)

        return dts

    def cbrt(self, x: float) -> float:
        return math.copysign(abs(x) ** (1/3), x)
    
    def stateTransit(self, state0: np.ndarray, um: Vector3d, tau: float) -> np.ndarray:

        for i in range(3):
            self.phi_[i, i+3] = tau
        integral = np.zeros(6)
        integral[0:3] = 0.5 * math.pow(tau, 2) * um
        integral[3:6] = tau * um

        return self.phi_ @ state0 + integral
    
    def getKinoTraj(self, delta_t: float) -> List[Vector3d]:

        state_list = []
        node = self.path_nodes_[-1]
        x0, xt = np.zeros(6), np.zeros(6)

        while node.parent is not None:
            ut = node.input
            duration = node.duration
            x0 = node.parent.state
            t = duration

            while t >= -1e-5:
                xt = self.stateTransit(x0, ut, t)
                state_list.append(Vector3d(xt[0], xt[1], xt[2]))
                t -= delta_t
            
            node = node.parent
        state_list.reverse()
        if self.is_shot_succ_:
            t = delta_t
            coord = Vector3d()
            time = np.zeros(4)
            while t <= self.t_shot_ :
                for i in range(4):
                    time[i] = t**i
                for i in range(3):
                    poly1d = self.coef_shot_[i]
                    coord[i] = np.dot(poly1d, time)
                state_list.append(coord)
                t += delta_t
        return state_list
    
    def getSamples(self, ts: float, point_set: List[Vector3d], start_end_derivatives: List[Vector3d]):
        
        T_sum = 0.0
        if self.is_shot_succ_:
            T_sum += self.t_shot_
        node = self.path_nodes_[-1]
        while node.parent is not None:
            T_sum += node.duration
            node = node.parent

        # Calculate boundary vel and acc
        end_vel = Vector3d()
        end_acc = Vector3d()
        t = 0.0
        if self.is_shot_succ_:
            t = self.t_shot_
            end_vel = self.end_vel_
            for dim in range(3):
                coe = self.coef_shot_[dim]
                end_acc[dim] = 2 * coe[2] + 6 * coe[3] * self.t_shot_
        else:
            t = self.path_nodes_[-1].duration
            end_vel = Vector3d(node.state[3], node.state[4], node.state[5])
            end_acc = self.path_nodes_[-1].input

        # Get point samples
        seg_num = math.floor(T_sum / ts)
        seg_num = max(8, seg_num)
        ts = T_sum / float(seg_num)
        sample_shot_traj = self.is_shot_succ_
        node = self.path_nodes_[-1]

        ti = T_sum
        while ti > -1e-5:
            if sample_shot_traj:
                # samples on shot traj
                coord = Vector3d()
                time = np.zeros(4)

                for j in range(4):
                    time[j] = t**j

                for dim in range(3):
                    poly1d = self.coef_shot_[dim]
                    coord[dim] = np.dot(poly1d, time)

                point_set.append(coord)
                t -= ts

                # end of segment
                if t < -1e-5:
                    sample_shot_traj = False
                    if node is not None and node.parent is not None:
                        t += node.duration
            else:
                # samples on searched traj
                if node.parent is not None:
                    x0 = node.parent.state
                    ut = node.input
                    xt = self.stateTransit(x0, ut, t)
                    point_set.append(Vector3d(xt[0], xt[1], xt[2]))
                    t -= ts

                    if t < -1e-5 and node.parent.parent is not None:
                        node = node.parent
                        t += node.duration
            ti -= ts

        point_set.reverse()

        # calculate start acc
        start_acc = Vector3d()
        if self.path_nodes_[-1].parent is None:
            # no searched traj, calculate by shot traj
            for i in range(3):
                start_acc[i] = 2 * self.coef_shot_[i][2]
        else:
            # input of searched traj
            start_acc = node.input

        start_end_derivatives.append(self.start_vel_)
        start_end_derivatives.append(end_vel)
        start_end_derivatives.append(start_acc)
        start_end_derivatives.append(end_acc)

        return ts, point_set, start_end_derivatives

    



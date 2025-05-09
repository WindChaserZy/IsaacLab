from ..utils import Vector3d
from ..env import EDTEnv, SDFmap
from ..perception import ViewConstraint
import numpy as np
import nlopt
from typing import List

class BsplineOptimizer:

    SMOOTHNESS = (1 << 0)
    DISTANCE = (1 << 1)
    FEASIBILITY = (1 << 2)
    START = (1 << 3)
    END = (1 << 4)
    GUIDE = (1 << 5)
    WAYPOINTS = (1 << 6)
    VIEWCONS = (1 << 7)
    MINTIME = (1 << 8)
    GUIDE_PHASE = SMOOTHNESS | GUIDE | START | END
    NORMAL_PHASE = SMOOTHNESS | DISTANCE | FEASIBILITY | START | END

    def __init__(self):

        self.max_iteration_num_ = [0, 0, 0, 0]
        self.max_iteration_time_ = [0.0, 0.0, 0.0, 0.0]
        ####### Config ############
        self.ld_smooth_ = -1.0
        self.ld_dist_ = -1.0
        self.ld_feasi_ = -1.0
        self.ld_start_ = -1.0
        self.ld_end_ = -1.0
        self.ld_guide_ = -1.0
        self.ld_waypt_ = -1.0
        self.ld_view_ = -1.0
        self.ld_time_ = -1.0
        self.dist0_ = -1.0
        self.max_vel_ = -1.0
        self.max_acc_ = -1.0
        self.dlmin_ = -1.0
        self.wnl_ = -1.0
        self.max_iteration_num_[0] = -1
        self.max_iteration_num_[1] = -1
        self.max_iteration_num_[2] = -1
        self.max_iteration_num_[3] = -1
        self.max_iteration_time_[0] = -1.0
        self.max_iteration_time_[1] = -1.0
        self.max_iteration_time_[2] = -1.0
        self.max_iteration_time_[3] = -1.0
        self.algorithm1_ = -1
        self.algorithm2_ = -1
        self.bspline_degree_ = 3
        ########## End Config ############
        self.time_lb_ = -1
        self.order_ = 1

        self.control_points_ : np.ndarray = np.array([])
        self.knot_span_ = -1.0        
        self.dim_ = -1
        self.start_state_ : List[Vector3d] = []
        self.end_state_ : List[Vector3d] = []
        self.guide_pts_ : List[Vector3d] = []
        self.waypoints_ : List[Vector3d] = []
        self.waypt_idx_ : List[int] = []
        self.max_num_id_ = -1
        self.max_time_id_ = -1
        self.cost_function_ = -1
        self.time_lb_ = -1.0
        self.dynamic_ = False
        self.start_time_ = -1.0

        # Gradient vectors
        self.g_q_ : List[Vector3d] = []
        self.g_smoothness_ : List[Vector3d] = []
        self.g_distance_ : List[Vector3d] = []
        self.g_feasibility_ : List[Vector3d] = []
        self.g_start_ : List[Vector3d] = []
        self.g_end_ : List[Vector3d] = []
        self.g_guide_ : List[Vector3d] = []
        self.g_waypoints_ : List[Vector3d] = []
        self.g_view_ : List[Vector3d] = []
        self.g_time_ : List[Vector3d] = []

        # Optimization variables
        self.variable_num_ = 0
        self.point_num_ = 0
        self.optimize_time_ = False
        self.iter_num_ = 0
        self.best_variable_ : List[float] = []
        self.min_cost_ = 0.0
        self.view_cons_ : ViewConstraint = ViewConstraint()
        self.pt_dist_ = 0.0

        # For benchmark evaluation
        self.vec_cost_ : List[float] = []
        self.vec_time_ : List[float] = []
        self.comb_time = 0.0

    def getCostCurve(self):
        return self.vec_cost_, self.vec_time_
    
    def setEnvironment(self, env: EDTEnv):
        self.edt_environment_ = env
        self.dynamic_ = False
    
    def setCostFunction(self, cost_code: int):
        self.cost_function_ = cost_code

    def setGuidePath(self, guide_pt: List[Vector3d]):
        self.guide_pts_ = guide_pt

    def setWaypoints(self, waypts: List[Vector3d], waypt_idx: List[int]):
        self.waypoints_ = waypts
        self.waypt_idx_ = waypt_idx

    def setViewConstraint(self, view_cons: ViewConstraint):
        self.view_cons_ = view_cons

    def enableDynamic(self, time_start: float):
        self.dynamic_ = True
        self.start_time_ = time_start

    def setBoundaryState(self, start_state: List[Vector3d], end_state: List[Vector3d]):
        self.start_state_ = start_state
        self.end_state_ = end_state

    def setTimeLowerBound(self, time_lb: float):
        self.time_lb_ = time_lb
    
    def optimize(self, points: np.ndarray, dt: float, cost_function: int, max_num_id: int, max_time_id: int):

        if self.start_state_ == []:
            print("Start state is not set")
            return
        self.control_points_ = points
        self.knot_span_ = dt
        self.max_num_id_ = max_num_id
        self.max_time_id_ = max_time_id
        self.setCostFunction(cost_function)
        self.dim_ = self.control_points_.shape[1]
        if self.dim_ == 1:
            self.order_ = 3
        else:
            self.order_ = self.bspline_degree_
        self.point_num_ = self.control_points_.shape[0]
        self.optimize_time_ = self.cost_function_ & self.MINTIME
        self.variable_num_ = self.dim_ * self.point_num_ + 1 if self.optimize_time_ else self.dim_ * self.point_num_
        if self.variable_num_ <= 0:
            print("empty optimization variable")
            return
        # Calculate average distance between control points
        self.pt_dist_ = 0.0
        for i in range(self.point_num_ - 1):
            self.pt_dist_ += np.linalg.norm(self.control_points_[i+1] - self.control_points_[i])
        self.pt_dist_ /= float(self.point_num_)

        # Initialize optimization variables
        self.iter_num_ = 0
        self.min_cost_ = float('inf')
        self.g_q_ = [Vector3d()] * self.point_num_
        self.g_smoothness_ = [Vector3d()] * self.point_num_
        self.g_distance_ = [Vector3d()] * self.point_num_
        self.g_feasibility_ = [Vector3d()] * self.point_num_
        self.g_start_ = [Vector3d()] * self.point_num_
        self.g_end_ = [Vector3d()] * self.point_num_
        self.g_guide_ = [Vector3d()] * self.point_num_
        self.g_waypoints_ = [Vector3d()] * self.point_num_
        self.g_view_ = [Vector3d()] * self.point_num_
        self.g_time_ = [Vector3d()] * self.point_num_

        self.comb_time = 0.0

        self.nloptOptimize()

        points = self.control_points_
        dt = self.knot_span_
        self.start_state_.clear()
        self.time_lb_ = -1

        return points, dt

    def nloptOptimize(self):
        algorithm = self.algorithm1_ if self.isQuadratic() else self.algorithm2_
        opt = nlopt.opt(algorithm, self.variable_num_)
        opt.set_min_objective(self.costFunction)
        opt.set_maxeval(self.max_iteration_num_[self.max_num_id_])
        opt.set_maxtime(self.max_iteration_time_[self.max_time_id_])
        opt.set_xtol_rel(1e-5)

        bmin, bmax = self.edt_environment_.sdf_map_.getBox()
        for k in range(3):
            bmin[k] += 0.1
            bmax[k] -= 0.1
        
        q = [0.0] * self.variable_num_
        # Variables for control points
        for i in range(self.point_num_):
            for j in range(self.dim_):
                cij = self.control_points_[i, j]
                if self.dim_ != 1:
                    cij = max(min(cij, bmax[j % 3]), bmin[j % 3])
                q[self.dim_ * i + j] = cij
        # Variables for knot span
        if self.optimize_time_:
            q[self.variable_num_ - 1] = self.knot_span_

        if self.dim_ != 1:
            lb = [0.0] * self.variable_num_
            ub = [0.0] * self.variable_num_
            bound = 10.0
            for i in range(3 * self.point_num_):
                lb[i] = q[i] - bound
                ub[i] = q[i] + bound
                lb[i] = max(lb[i], bmin[i % 3])
                ub[i] = min(ub[i], bmax[i % 3])
            if self.optimize_time_:
                lb[self.variable_num_ - 1] = 0.0
                ub[self.variable_num_ - 1] = 5.0
            opt.set_lower_bounds(lb)
            opt.set_upper_bounds(ub)
        
        try:
            final_cost = 0.0
            q = opt.optimize(np.array(q))
            final_cost = opt.last_optimum_value()
        except Exception as e:
            print(str(e))

        for i in range(self.point_num_):
            for j in range(self.dim_):
                self.control_points_[i, j] = self.best_variable_[self.dim_ * i + j]
        
        if self.optimize_time_:
            self.knot_span_ = self.best_variable_[self.variable_num_ - 1]
    
    def calcSmoothnessCost(self, q: List[Vector3d], dt: float, cost: float, gradient_q: List[Vector3d], gt: float):
        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero
        jerk = Vector3d(0.0, 0.0, 0.0)
        temp_j = Vector3d(0.0, 0.0, 0.0)
        for i in range(len(q) - 3):
            # Test jerk cost
            ji = (q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / self.pt_dist_
            cost += ji.norm2()
            temp_j = 2 * ji / self.pt_dist_

            # Update gradients
            gradient_q[i + 0] -= temp_j
            gradient_q[i + 1] += 3.0 * temp_j  
            gradient_q[i + 2] -= 3.0 * temp_j
            gradient_q[i + 3] += temp_j
        
        return cost, gradient_q
    
    def calcDistanceCost(self, q: List[Vector3d], cost: float, gradient_q: List[Vector3d]):
        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero
        dist = 0.0
        dist_grad = Vector3d(0.0, 0.0, 0.0)
        g_zero = Vector3d(0.0, 0.0, 0.0)
        for i in range(len(q)):
            if not self.dynamic_:
                dist, dist_grad = self.edt_environment_.evaluateEDTWithGrad(q[i], -1.0, dist, dist_grad)
                if dist_grad.norm() > 1e-4:
                    dist_grad.normalize()
            else:
                time = float(i + 2 - self.order_) * self.knot_span_ + self.start_time_
                dist, dist_grad = self.edt_environment_.evaluateEDTWithGrad(q[i], time, dist, dist_grad)

            if dist < self.dist0_:
                cost += pow(dist - self.dist0_, 2)
                gradient_q[i] += 2.0 * (dist - self.dist0_) * dist_grad

    def calcFeasibilityCost(self, q: List[Vector3d], dt: float, cost: float, gradient_q: List[Vector3d], gt: float):
        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero
        gt = 0.0
        dt_inv = 1.0 / dt
        dt_inv2 = dt_inv * dt_inv
        # Velocity feasibility cost
        for i in range(len(q) - 1):
            # Control point of velocity
            diff = q[i + 1] - q[i]
            vi = Vector3d(diff.x * dt_inv, diff.y * dt_inv, diff.z * dt_inv)
            for k in range(3):
                # Calculate cost for each axis
                vd = abs(vi[k]) - self.max_vel_
                if vd > 0.0:
                    cost += vd * vd
                    sign = 1.0 if vi[k] > 0 else -1.0
                    tmp = 2 * vd * sign * dt_inv
                    gradient_q[i][k] += -tmp
                    gradient_q[i + 1][k] += tmp
                    if self.optimize_time_:
                        gt += tmp * (-vi[k])

        # Acceleration feasibility cost
        for i in range(len(q) - 2):
            diff = q[i + 2] - 2 * q[i + 1] + q[i]
            ai = Vector3d(diff.x * dt_inv2, diff.y * dt_inv2, diff.z * dt_inv2)
            for k in range(3):
                ad = abs(ai[k]) - self.max_acc_
                if ad > 0.0:
                    cost += ad * ad
                    sign = 1.0 if ai[k] > 0 else -1.0
                    tmp = 2 * ad * sign * dt_inv2
                    gradient_q[i][k] += tmp
                    gradient_q[i + 1][k] += -2 * tmp
                    gradient_q[i + 2][k] += tmp
                    if self.optimize_time_:
                        gt += tmp * ai[k] * (-2) * dt
        
        return cost, gradient_q
    
    def calcStartCost(self, q: List[Vector3d], dt: float, cost: float, gradient_q: List[Vector3d], gt: float):

        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero
        gt = 0.0
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        w_pos = 10.0
        # Start position
        dq = Vector3d(
            (q1.x + 4*q2.x + q3.x)/6.0 - self.start_state_[0].x,
            (q1.y + 4*q2.y + q3.y)/6.0 - self.start_state_[0].y,
            (q1.z + 4*q2.z + q3.z)/6.0 - self.start_state_[0].z
        )
        cost += w_pos * dq.norm2()
        gradient_q[0] += w_pos * 2 * dq * (1/6.0)
        gradient_q[1] += w_pos * 2 * dq * (4/6.0) 
        gradient_q[2] += w_pos * 2 * dq * (1/6.0)

        # Start velocity
        dq = Vector3d(
            (q3.x - q1.x)/(2*dt) - self.start_state_[1].x,
            (q3.y - q1.y)/(2*dt) - self.start_state_[1].y,
            (q3.z - q1.z)/(2*dt) - self.start_state_[1].z
        )
        cost += dq.norm2()
        gradient_q[0] += 2 * dq * (-1.0)/(2*dt)
        gradient_q[2] += 2 * dq * 1.0/(2*dt)
        if self.optimize_time_:
            gt += (dq.x * (q3.x - q1.x) + dq.y * (q3.y - q1.y) + dq.z * (q3.z - q1.z)) / (-dt * dt)

        # Start acceleration 
        dq = Vector3d(
            (q1.x - 2*q2.x + q3.x)/(dt*dt) - self.start_state_[2].x,
            (q1.y - 2*q2.y + q3.y)/(dt*dt) - self.start_state_[2].y,
            (q1.z - 2*q2.z + q3.z)/(dt*dt) - self.start_state_[2].z
        )
        cost += dq.norm2()
        gradient_q[0] += 2 * dq * 1.0/(dt*dt)
        gradient_q[1] += 2 * dq * (-2.0)/(dt*dt)
        gradient_q[2] += 2 * dq * 1.0/(dt*dt)
        if self.optimize_time_:
            gt += (dq.x * (q1.x - 2*q2.x + q3.x) + dq.y * (q1.y - 2*q2.y + q3.y) + dq.z * (q1.z - 2*q2.z + q3.z)) / (-dt * dt * dt)
        
        return cost, gradient_q

    def calcEndCost(self, q: List[Vector3d], dt: float, cost: float, gradient_q: List[Vector3d], gt: float):
        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero
        gt = 0.0
        q1 = q[-1]
        q2 = q[-2] 
        q3 = q[-3]
        w_pos = 10.0

        # End position
        dq = Vector3d(
            (q1.x + 4*q2.x + q3.x)/6.0 - self.end_state_[0].x,
            (q1.y + 4*q2.y + q3.y)/6.0 - self.end_state_[0].y,
            (q1.z + 4*q2.z + q3.z)/6.0 - self.end_state_[0].z
        )
        cost += w_pos * dq.norm2()
        gradient_q[-1] += w_pos * 2 * dq * (1/6.0)
        gradient_q[-2] += w_pos * 2 * dq * (4/6.0)
        gradient_q[-3] += w_pos * 2 * dq * (1/6.0)

        if len(self.end_state_) >= 2:
            # End velocity
            dq = Vector3d(
                (q1.x - q3.x)/(2*dt) - self.end_state_[1].x,
                (q1.y - q3.y)/(2*dt) - self.end_state_[1].y,
                (q1.z - q3.z)/(2*dt) - self.end_state_[1].z
            )
            cost += dq.norm2()
            gradient_q[-1] += 2 * dq * 1.0/(2*dt)
            gradient_q[-3] += 2 * dq * (-1.0)/(2*dt)
            if self.optimize_time_:
                gt += (dq.x * (q1.x - q3.x) + dq.y * (q1.y - q3.y) + dq.z * (q1.z - q3.z)) / (-dt * dt)

        if len(self.end_state_) == 3:
            # End acceleration
            dq = Vector3d(
                (q1.x - 2*q2.x + q3.x)/(dt*dt) - self.end_state_[2].x,
                (q1.y - 2*q2.y + q3.y)/(dt*dt) - self.end_state_[2].y,
                (q1.z - 2*q2.z + q3.z)/(dt*dt) - self.end_state_[2].z
            )
            cost += dq.norm2()
            gradient_q[-1] += 2 * dq * 1.0/(dt*dt)
            gradient_q[-2] += 2 * dq * (-2.0)/(dt*dt)
            gradient_q[-3] += 2 * dq * 1.0/(dt*dt)
            if self.optimize_time_:
                gt += (dq.x * (q1.x - 2*q2.x + q3.x) + dq.y * (q1.y - 2*q2.y + q3.y) + dq.z * (q1.z - 2*q2.z + q3.z)) / (-dt * dt * dt)

        return cost, gradient_q
    
    def calcWaypointsCost(self, q: List[Vector3d], cost: float, gradient_q: List[Vector3d]):

        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero

        for i in range(len(self.waypoints_)):
            waypt = self.waypoints_[i]
            idx = self.waypt_idx_[i]

            q1 = q[idx]
            q2 = q[idx + 1] 
            q3 = q[idx + 2]

            dq = Vector3d(
                (q1.x + 4*q2.x + q3.x)/6.0 - waypt.x,
                (q1.y + 4*q2.y + q3.y)/6.0 - waypt.y,
                (q1.z + 4*q2.z + q3.z)/6.0 - waypt.z
            )
            cost += dq.norm2()

            gradient_q[idx] += 2 * dq * (1/6.0)
            gradient_q[idx + 1] += 2 * dq * (4/6.0)
            gradient_q[idx + 2] += 2 * dq * (1/6.0)

        return cost, gradient_q

    def calcGuideCost(self, q: List[Vector3d], cost: float, gradient_q: List[Vector3d]):

        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero

        end_idx = len(q) - self.order_

        for i in range(self.order_, end_idx):
            gpt = self.guide_pts_[i - self.order_]
            dq = q[i] - gpt
            cost += dq.norm2()
            gradient_q[i] += 2 * dq

        return cost, gradient_q

    def calcViewCost(self, q: List[Vector3d], cost: float, gradient_q: List[Vector3d]):

        cost = 0.0
        zero = Vector3d(0.0, 0.0, 0.0)
        for i, gradient in enumerate(gradient_q):
            gradient_q[i] = zero

        p = self.view_cons_.pt_
        v = self.view_cons_.dir_.normalized()
        # Calculate outer product matrix manually
        vvT = np.array([
            [v.x * v.x, v.x * v.y, v.x * v.z],
            [v.y * v.x, v.y * v.y, v.y * v.z],
            [v.z * v.x, v.z * v.y, v.z * v.z]
        ])
        I_vvT = np.eye(3) - vvT

        # perpendicular cost, increase visibility of points before blocked point
        i = self.view_cons_.idx_
        q_p = q[i] - p
        dot_prod = q_p.x * v.x + q_p.y * v.y + q_p.z * v.z
        dn = q_p - Vector3d(dot_prod * v.x, dot_prod * v.y, dot_prod * v.z)
        cost += dn.norm2()
        # Manual matrix multiplication
        I_vvT_dn = Vector3d(
            I_vvT[0,0] * dn.x + I_vvT[0,1] * dn.y + I_vvT[0,2] * dn.z,
            I_vvT[1,0] * dn.x + I_vvT[1,1] * dn.y + I_vvT[1,2] * dn.z,
            I_vvT[2,0] * dn.x + I_vvT[2,1] * dn.y + I_vvT[2,2] * dn.z
        )
        gradient_q[i] += 2 * I_vvT_dn
        norm_dn = dn.norm()

        # parallel cost, increase projection along view direction
        dl = Vector3d(dot_prod * v.x, dot_prod * v.y, dot_prod * v.z)
        norm_dl = dl.norm()
        safe_dist = self.view_cons_.dir_.norm()
        if norm_dl < safe_dist:
            cost += self.wnl_ * pow(norm_dl - safe_dist, 2)
            # Manual matrix multiplication
            vvT_dl = Vector3d(
                vvT[0,0] * dl.x + vvT[0,1] * dl.y + vvT[0,2] * dl.z,
                vvT[1,0] * dl.x + vvT[1,1] * dl.y + vvT[1,2] * dl.z,
                vvT[2,0] * dl.x + vvT[2,1] * dl.y + vvT[2,2] * dl.z
            )
            gradient_q[i] += self.wnl_ * 2 * (norm_dl - safe_dist) * vvT_dl / norm_dl

        return cost, gradient_q

    def calcTimeCost(self, dt: float, cost: float, gt: float):

        duration = (self.point_num_ - self.order_) * dt
        cost = duration
        gt = float(self.point_num_ - self.order_)

        # Time lower bound
        if self.time_lb_ > 0 and duration < self.time_lb_:
            w_lb = 10.0
            cost += w_lb * pow(duration - self.time_lb_, 2)
            gt += w_lb * 2 * (duration - self.time_lb_) * (self.point_num_ - self.order_)

        return cost, gt

    def combineCost(self, x: List[float], grad: List[float]):

        # Initialize control points from optimization variables
        for i in range(self.point_num_):
            for j in range(self.dim_):
                self.g_q_[i][j] = x[self.dim_ * i + j]
            for j in range(self.dim_, 3):
                self.g_q_[i][j] = 0.0

        # Get time step
        dt = x[self.variable_num_ - 1] if self.optimize_time_ else self.knot_span_

        # Initialize cost and gradient
        f_combine = 0.0
        if len(grad) > self.variable_num_:
            grad[:] = grad[:self.variable_num_]
        grad[:] = [0.0 for _ in range(self.variable_num_)]

        # Smoothness cost
        if self.cost_function_ & self.SMOOTHNESS:
            f_smoothness = 0.0
            gt_smoothness = 0.0
            f_smoothness, self.g_smoothness_ = self.calcSmoothnessCost(self.g_q_, dt, f_smoothness, self.g_smoothness_, gt_smoothness)
            f_combine += self.ld_smooth_ * f_smoothness
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_smooth_ * self.g_smoothness_[i][j]
            if self.optimize_time_:
                grad[self.variable_num_ - 1] += self.ld_smooth_ * gt_smoothness

        # Distance cost
        if self.cost_function_ & self.DISTANCE:
            f_distance = 0.0
            f_distance, self.g_distance_ = self.calcDistanceCost(self.g_q_, f_distance, self.g_distance_)
            f_combine += self.ld_dist_ * f_distance
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_dist_ * self.g_distance_[i][j]

        # Feasibility cost
        if self.cost_function_ & self.FEASIBILITY:
            f_feasibility = 0.0
            gt_feasibility = 0.0
            f_feasibility, self.g_feasibility_ = self.calcFeasibilityCost(self.g_q_, dt, f_feasibility, self.g_feasibility_, gt_feasibility)
            f_combine += self.ld_feasi_ * f_feasibility
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_feasi_ * self.g_feasibility_[i][j]
            if self.optimize_time_:
                grad[self.variable_num_ - 1] += self.ld_feasi_ * gt_feasibility

        # Start cost
        if self.cost_function_ & self.START:
            f_start = 0.0
            gt_start = 0.0
            f_start, self.g_start_ = self.calcStartCost(self.g_q_, dt, f_start, self.g_start_, gt_start)
            f_combine += self.ld_start_ * f_start
            for i in range(3):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_start_ * self.g_start_[i][j]
            if self.optimize_time_:
                grad[self.variable_num_ - 1] += self.ld_start_ * gt_start

        # End cost
        if self.cost_function_ & self.END:
            f_end = 0.0
            gt_end = 0.0
            f_end, self.g_end_ = self.calcEndCost(self.g_q_, dt, f_end, self.g_end_, gt_end)
            f_combine += self.ld_end_ * f_end
            for i in range(self.point_num_ - 3, self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_end_ * self.g_end_[i][j]
            if self.optimize_time_:
                grad[self.variable_num_ - 1] += self.ld_end_ * gt_end

        # Guide cost
        if self.cost_function_ & self.GUIDE:
            f_guide = 0.0
            f_guide, self.g_guide_ = self.calcGuideCost(self.g_q_, f_guide, self.g_guide_)
            f_combine += self.ld_guide_ * f_guide
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_guide_ * self.g_guide_[i][j]

        # Waypoints cost
        if self.cost_function_ & self.WAYPOINTS:
            f_waypoints = 0.0
            f_waypoints, self.g_waypoints_ = self.calcWaypointsCost(self.g_q_, f_waypoints, self.g_waypoints_)
            f_combine += self.ld_waypt_ * f_waypoints
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_waypt_ * self.g_waypoints_[i][j]

        # View constraint cost
        if self.cost_function_ & self.VIEWCONS:
            f_view = 0.0
            f_view, self.g_view_ = self.calcViewCost(self.g_q_, f_view, self.g_view_)
            f_combine += self.ld_view_ * f_view
            for i in range(self.point_num_):
                for j in range(self.dim_):
                    grad[self.dim_ * i + j] += self.ld_view_ * self.g_view_[i][j]

        # Minimum time cost
        if self.cost_function_ & self.MINTIME:
            f_time = 0.0
            gt_time = 0.0
            f_time, gt_time = self.calcTimeCost(dt, f_time, gt_time)
            f_combine += self.ld_time_ * f_time
            grad[self.variable_num_ - 1] += self.ld_time_ * gt_time
                
        return f_combine

    def costFunction(self, x: List[float], grad: List[float]):
        cost = self.combineCost(x, grad)
        self.iter_num_ += 1
        if cost < self.min_cost_:
            self.min_cost_ = cost
            self.best_variable_ = x
        return cost
    
    def matrixToVectors(self, ctrl_pts: np.ndarray):
        vectors = []
        for i in range(ctrl_pts.shape[0]):
            vectors.append(Vector3d(ctrl_pts[i, 0], ctrl_pts[i, 1], ctrl_pts[i, 2]))
        return vectors

    def getControlPoints(self):
        return self.control_points_

    def isQuadratic(self):

        if self.cost_function_ == self.GUIDE_PHASE:
            return True
        elif self.cost_function_ == self.SMOOTHNESS:
            return True
        elif self.cost_function_ == (self.SMOOTHNESS | self.WAYPOINTS):
            return True
        return False


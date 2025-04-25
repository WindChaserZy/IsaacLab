import numpy as np
from typing import List
from ..utils import Vector3d

class NonUniformBspline:
    def __init__(self, points: np.ndarray = np.array([]), order: int = 0, interval: float = 0.0):
        self.control_points_ = points
        self.p_ = order  # degree
        self.knot_span_ = interval
        self.u_ = []  # knot vector
        self.n_ = 0  # number of control points - 1 
        self.m_ = 0  # n + p + 1
        self.limit_vel_ = 0.0
        self.limit_acc_ = 0.0 
        self.limit_ratio_ = 0.0
        
        if points is not np.array([]) and order is not 0 and interval is not 0.0:
            self.setUniformBspline(points, order, interval)

    def setUniformBspline(self, points: np.ndarray, order: int, interval: float):
        self.control_points_ = points
        self.p_ = order
        self.knot_span_ = interval
        self.n_ = points.shape[0] - 1
        self.m_ = self.n_ + self.p_ + 1
        
        # Initialize uniform knot vector
        self.u_ = [0.0] * (self.m_ + 1)
        for i in range(self.m_ + 1):
            if i <= self.p_:
                self.u_[i] = (-self.p_ + i) * self.knot_span_
            else:
                self.u_[i] = self.u_[i-1] + self.knot_span_

    def setKnot(self, knot):
        self.u_ = knot

    def getKnot(self):
        return self.u_

    def getControlPoint(self):
        return self.control_points_

    def getKnotSpan(self):
        return self.knot_span_

    def getTimeSpan(self):
        return self.u_[self.p_], self.u_[self.m_ - self.p_]

    def evaluateDeBoor(self, u: float):
        # Find knot span containing u
        # Clamp u to valid range
        ub = min(max(self.u_[self.p_], u), self.u_[self.m_ - self.p_])

        # Find knot span k where u lies
        k = self.p_
        while self.u_[k + 1] < ub:
            k += 1

        # Initialize d array with control points
        d = []
        for i in range(self.p_ + 1):
            d.append(self.control_points_[k - self.p_ + i])

        # de Boor's algorithm
        for r in range(1, self.p_ + 1):
            for i in range(self.p_, r-1, -1):
                alpha = (ub - self.u_[i + k - self.p_]) / (self.u_[i + 1 + k - r] - self.u_[i + k - self.p_])
                d[i] = (1 - alpha) * d[i-1] + alpha * d[i]

        return d[self.p_]

    def evaluateDeBoorT(self, t: float):
        return self.evaluateDeBoor(t + self.u_[self.p_])

    def getTimeSum(self):
        return self.u_[self.m_ - self.p_] - self.u_[self.p_]
    
    def getDerivativeControlPoints(self):
        
        ctp = np.zeros((self.control_points_.shape[0] - 1, self.control_points_.shape[1]))
        for i in range(ctp.shape[0]):
            ctp[i] = self.p_ * (self.control_points_[i + 1] - self.control_points_[i]) / (self.u_[i + self.p_+1] - self.u_[i + 1])
        return ctp

    def computeDerivative(self, k: int, ders: List['NonUniformBspline']):
        ders.clear()
        ders.append(self.getDerivative())
        for i in range(2, k):
            ders.append(ders[-1].getDerivative())
        
        return ders

    def getDerivative(self):
        ctp = self.getDerivativeControlPoints()
        der = NonUniformBspline(ctp, self.p_, self.knot_span_)

        # Set knot vector for derivative curve by removing first and last knots
        knot = self.u_[1:-1]  
        der.setKnot(knot)
        return der
    
    def getBoundaryStates(self, ks: int, ke: int, start: List[Vector3d], end: List[Vector3d], ders: List['NonUniformBspline']):

        ders = self.computeDerivative(max(ks, ke), ders)
        dur = self.getTimeSum()

        start.clear()
        end.clear()
        start.append(self.evaluateDeBoorT(0))
        for i in range(0, ks):
            start.append(ders[i].evaluateDeBoorT(0))
        end.append(self.evaluateDeBoorT(dur))
        for i in range(0, ke):
            end.append(ders[i].evaluateDeBoorT(dur))
        return start, end

    def setPhysicalLimits(self, vel: float, acc: float):
        self.limit_vel_ = vel
        self.limit_acc_ = acc
        self.limit_ratio_ = 1.1
    
    def checkRatio(self):

        P = self.control_points_
        dimension = P.shape[1]

        max_vel = -1.0
        # Find max velocity
        for i in range(P.shape[0] - 1):
            vel = self.p_ * (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])
            for j in range(dimension):
                max_vel = max(max_vel, abs(vel[j]))

        # Find max acceleration
        max_acc = -1.0
        for i in range(P.shape[0] - 2):
            acc = self.p_ * (self.p_ - 1) * (
                (P[i + 2] - P[i + 1]) / (self.u_[i + self.p_ + 2] - self.u_[i + 2]) -
                (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])
            ) / (self.u_[i + self.p_ + 1] - self.u_[i + 2])
            
            for j in range(dimension):
                max_acc = max(max_acc, abs(acc[j]))

        ratio = max(max_vel / self.limit_vel_, (abs(max_acc) / self.limit_acc_)**0.5)
        return ratio
    
    def lengthenTime(self, ratio: float):

        num1 = 2 * self.p_ - 1
        num2 = len(self.u_) - 2 * self.p_ + 1
        if num1 >= num2:
            return

        delta_t = (ratio - 1.0) * (self.u_[num2] - self.u_[num1])
        t_inc = delta_t / float(num2 - num1)
        
        for i in range(num1 + 1, num2 + 1):
            self.u_[i] += float(i - num1) * t_inc
            
        for i in range(num2 + 1, len(self.u_)):
            self.u_[i] += delta_t

    def parameterizeToBspline(self, ts: float, point_set: List[Vector3d], start_end_derivative: List[Vector3d], degree: int, ctrl_pts: np.ndarray):

        if ts <= 0:
            print("[B-spline]:time step error.")
            return
            
        if len(point_set) < 2:
            print(f"[B-spline]:point set have only {len(point_set)} points.")
            return
            
        if len(start_end_derivative) != 4:
            print("[B-spline]:derivatives error.")

        K = len(point_set)
        A = np.zeros((K + 4, K + degree - 1))
        bx = [0.0] * (K + 4)
        by = [0.0] * (K + 4)
        bz = [0.0] * (K + 4)

        ctrl_pts = np.zeros((K + degree - 1, 3))
        if degree == 3:
            # Matrix mapping control points to waypoints and boundary derivatives
            pt_to_pos = np.array([1.0, 4.0, 1.0]) / 6.0
            pt_to_vel = np.array([-1.0, 0.0, 1.0]) / (2.0 * ts) 
            pt_to_acc = np.array([1.0, -2.0, 1.0]) / (ts * ts)

            # Fill matrix A with position constraints
            for i in range(K):
                A[i, i:i+3] = pt_to_pos

            # Fill matrix A with velocity constraints
            A[K, 0:3] = pt_to_vel
            A[K+1, K-1:K+2] = pt_to_vel

            # Fill matrix A with acceleration constraints  
            A[K+2, 0:3] = pt_to_acc
            A[K+3, K-1:K+2] = pt_to_acc
        
        elif degree == 4:
            # Matrix mapping control points to waypoints and boundary derivatives
            pt_to_pos = np.array([1.0, 11.0, 11.0, 1.0]) / 24.0
            pt_to_vel = np.array([-1.0, -3.0, 3.0, 1.0]) / (6.0 * ts)
            pt_to_acc = np.array([1.0, -1.0, -1.0, 1.0]) / (2.0 * ts * ts)

            # Fill matrix A with position constraints
            for i in range(K):
                A[i, i:i+4] = pt_to_pos

            # Fill matrix A with velocity constraints
            A[K, 0:4] = pt_to_vel
            A[K+1, K-1:K+3] = pt_to_vel

            # Fill matrix A with acceleration constraints
            A[K+2, 0:4] = pt_to_acc 
            A[K+3, K-1:K+3] = pt_to_acc
        
        elif degree == 5:
            # Matrix mapping control points to waypoints and boundary derivatives
            pt_to_pos = np.array([1.0, 26.0, 66.0, 26.0, 1.0]) / 120.0
            pt_to_vel = np.array([-1.0, -10.0, 0.0, 10.0, 1.0]) / (24.0 * ts)
            pt_to_acc = np.array([1.0, 2.0, -6.0, 2.0, 1.0]) / (6.0 * ts * ts)

            # Fill matrix A with position constraints
            for i in range(K):
                A[i, i:i+5] = pt_to_pos

            # Fill matrix A with velocity constraints
            A[K, 0:5] = pt_to_vel
            A[K+1, K-1:K+4] = pt_to_vel

            # Fill matrix A with acceleration constraints
            A[K+2, 0:5] = pt_to_acc
            A[K+3, K-1:K+4] = pt_to_acc

        for i in range(K):
            bx[i] = point_set[i].x
            by[i] = point_set[i].y
            bz[i] = point_set[i].z

        for i in range(4):
            bx[K + i] = start_end_derivative[i].x
            by[K + i] = start_end_derivative[i].y
            bz[K + i] = start_end_derivative[i].z

        ctrl_pts = np.linalg.solve(A, np.vstack((bx, by, bz)))

        return ctrl_pts

    def getLength(self, res: float):

        length = 0.0
        dur = self.getTimeSum()
        p_l = self.evaluateDeBoorT(0.0)
        t = res
        while t <= dur + 1e-4:
            p_n = self.evaluateDeBoorT(t) 
            length += np.linalg.norm(p_n - p_l)
            p_l = p_n
            t += res
        return length
    
    def getJerk(self):

        jerk_traj = self.getDerivative().getDerivative().getDerivative()
        times = jerk_traj.getKnot()
        ctrl_pts = jerk_traj.getControlPoint()
        dimension = ctrl_pts.shape[1]

        jerk = 0.0
        for i in range(ctrl_pts.shape[0]):
            for j in range(dimension):
                jerk += (times[i+1] - times[i]) * (ctrl_pts[i][j] ** 2)
        return jerk

    def getMeanAndMaxVel(self, mean_v: float, max_v: float):
        vel = self.getDerivative()
        tm, tmp = vel.getTimeSpan()
        max_vel, mean_vel = -1.0, 0.0
        num = 0
        t = tm
        while t <= tmp:
            vxd = vel.evaluateDeBoor(t)
            vn = float(np.linalg.norm(vxd))
            max_vel = max(max_vel, vn)
            mean_vel += vn
            num += 1
            t += 0.01
        mean_vel /= num
        mean_v = mean_vel
        max_v = max_vel
        return mean_v, max_v
    
    def getMeanAndMaxAcc(self, mean_a: float, max_a: float):

        acc = self.getDerivative().getDerivative()
        tm, tmp = acc.getTimeSpan()
        max_acc, mean_acc = -1.0, 0.0
        num = 0
        t = tm
        while t <= tmp:
            axd = acc.evaluateDeBoor(t)
            an = float(np.linalg.norm(axd))
            max_acc = max(max_acc, an)
            mean_acc += an
            num += 1
            t += 0.01
        mean_acc /= num
        mean_a = mean_acc
        max_a = max_acc
        return mean_a, max_a
    
    def reallocateTime(self, show: bool):

        fea = True
        P = self.control_points_
        dimension = P.shape[1]
        max_vel, max_acc = -1.0, -1.0

        for i in range(P.shape[0] - 1):
            vel = self.p_ * (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])

            if (abs(vel[0]) > self.limit_vel_ + 1e-4 or 
                abs(vel[1]) > self.limit_vel_ + 1e-4 or
                abs(vel[2]) > self.limit_vel_ + 1e-4):
                
                fea = False
                if show:
                    print(f"[Realloc]: Infeasible vel {i}: {vel}")

                max_vel = -1.0
                for j in range(dimension):
                    max_vel = max(max_vel, abs(vel[j]))

                ratio = max_vel / self.limit_vel_ + 1e-4
                if ratio > self.limit_ratio_:
                    ratio = self.limit_ratio_

                time_ori = self.u_[i + self.p_ + 1] - self.u_[i + 1]
                time_new = ratio * time_ori
                delta_t = time_new - time_ori
                t_inc = delta_t / float(self.p_)

                for j in range(i + 2, i + self.p_ + 2):
                    self.u_[j] += float(j - i - 1) * t_inc

                for j in range(i + self.p_ + 2, len(self.u_)):
                    self.u_[j] += delta_t
        
        for i in range(P.shape[0] - 2):
            acc = self.p_ * (self.p_ - 1) * ((P[i + 2] - P[i + 1]) / (self.u_[i + self.p_ + 2] - self.u_[i + 2]) -
                                            (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])) / \
                (self.u_[i + self.p_ + 1] - self.u_[i + 2])

            if (abs(acc[0]) > self.limit_acc_ + 1e-4 or
                abs(acc[1]) > self.limit_acc_ + 1e-4 or
                abs(acc[2]) > self.limit_acc_ + 1e-4):
                
                fea = False
                if show:
                    print(f"[Realloc]: Infeasible acc {i}: {acc}")

                max_acc = -1.0
                for j in range(dimension):
                    max_acc = max(max_acc, abs(acc[j]))

                ratio = np.sqrt(max_acc / self.limit_acc_) + 1e-4
                if ratio > self.limit_ratio_:
                    ratio = self.limit_ratio_

                time_ori = self.u_[i + self.p_ + 1] - self.u_[i + 2] 
                time_new = ratio * time_ori
                delta_t = time_new - time_ori
                t_inc = delta_t / float(self.p_ - 1)

                if i == 1 or i == 2:
                    for j in range(2, 6):
                        self.u_[j] += float(j - 1) * t_inc

                    for j in range(6, len(self.u_)):
                        self.u_[j] += 4.0 * t_inc
                else:
                    for j in range(i + 3, i + self.p_ + 2):
                        self.u_[j] += float(j - i - 2) * t_inc

                    for j in range(i + self.p_ + 2, len(self.u_)):
                        self.u_[j] += delta_t

        return fea
    
    def checkFeasibility(self, show: bool):

        fea = True
        P = self.control_points_
        dimension = P.shape[1]
        max_vel, max_acc = -1.0, -1.0

        for i in range(P.shape[0] - 1):
            vel = self.p_ * (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])

            if (abs(vel[0]) > self.limit_vel_ + 1e-4 or 
                abs(vel[1]) > self.limit_vel_ + 1e-4 or
                abs(vel[2]) > self.limit_vel_ + 1e-4):
                fea = False
                if show:
                    print(f"[Check]: Infeasible vel {i}: {vel}")

                for j in range(dimension):
                    max_vel = max(max_vel, abs(vel[j]))
        
        for i in range(P.shape[0] - 2):
            acc = self.p_ * (self.p_ - 1) * ((P[i + 2] - P[i + 1]) / (self.u_[i + self.p_ + 2] - self.u_[i + 2]) -
                                            (P[i + 1] - P[i]) / (self.u_[i + self.p_ + 1] - self.u_[i + 1])) / \
                (self.u_[i + self.p_ + 1] - self.u_[i + 2])
            
            if (abs(acc[0]) > self.limit_acc_ + 1e-4 or
                abs(acc[1]) > self.limit_acc_ + 1e-4 or
                abs(acc[2]) > self.limit_acc_ + 1e-4):
                fea = False
                if show:
                    print(f"[Check]: Infeasible acc {i}: {acc}")    
                
                for j in range(dimension):
                    max_acc = max(max_acc, abs(acc[j]))
        
        return fea
                    
        
                    

from ..utils import Vector3d
import numpy as np
from typing import List

class Polynomial:

    def __init__(self, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray, time: float):

       self.cx_ = cx
       self.cy_ = cy
       self.cz_ = cz
       self.time_ = time

    def getTBasis(self, t: float, n: int, k: int):

        coeff = 1
        for i in range(n, n - k, -1):
            coeff *= i
        return coeff * pow(t, n - k)

    def getTime(self):
        return self.time_
    
    def evaluate(self, t: float, k: int):

        tv = np.zeros(6)
        for i in range(k, 6):
            tv[i] = self.getTBasis(t, i, k)
        pt = Vector3d(0, 0, 0)
        pt.x = np.dot(self.cx_, tv)
        pt.y = np.dot(self.cy_, tv)
        pt.z = np.dot(self.cz_, tv)
        return pt

class PolynomialTraj:

    def __init__(self):

        self.segments_ : List[Polynomial] = []
        self.times_ : List[float] = []
        self.time_sum_ : float = 0.0
        self.sample_points_ : List[Vector3d] = []
        self.length_ : float = 0.0
    
    def reset(self):
        self.segments_.clear()
        self.times_.clear()
        self.time_sum_ = -1
        self.sample_points_.clear()
        self.length_ = -1
    
    def addSegment(self, poly: Polynomial):

        self.segments_.append(poly)
        self.times_.append(poly.getTime())
    
    def evaluate(self, t: float, k: int):

        idx = 0
        ts = t
        while self.times_[idx] + 1e-4 < t:
            ts -= self.times_[idx]
            idx += 1
        return self.segments_[idx].evaluate(ts, k)
    
    def getTotalTime(self):
        self.time_sum_ = 0
        for t in self.times_:
            self.time_sum_ += t
        return self.time_sum_
    
    def getSamplePoints(self, points: List[Vector3d]):

        eval_t = 0.0
        total_t = self.getTotalTime()
        points.clear()
        while eval_t < total_t:
            pt = self.evaluate(eval_t, 0)
            points.append(pt)
            eval_t += 0.01
        self.sample_points_ = points
        return points
    
    def getLength(self):

        pts = []
        if len(self.sample_points_) == 0:
            pts = self.getSamplePoints(pts)
            
        self.length_ = 0.0
        p_prev = self.sample_points_[0]
        p_cur = Vector3d(0, 0, 0)
        
        for i in range(1, len(self.sample_points_)):
            p_cur = self.sample_points_[i]
            self.length_ += (p_cur - p_prev).norm()
            p_prev = p_cur
            
        return self.length_
    
    def getMeanSpeed(self):

        if self.time_sum_ < 0:
            self.getTotalTime()
        if self.length_ < 0:
            self.getLength()
        return self.length_ / self.time_sum_
    
    def getIntegralCost(self, k: int):

        cost = 0.0
        if self.time_sum_ < 0:
            self.getTotalTime()
        ts = 0
        while ts < self.time_sum_:
            um = self.evaluate(ts, k)
            cost += um.norm2() * 0.01
            ts += 0.01
        return cost
    
    def getMeanAndMaxDerivative(self, mean_d: float, max_d: float, k: int):

        mean_d = 0.0
        max_d = 0.0
        sample_num = 0
        if self.time_sum_ < 0:
            self.getTotalTime()
        ts = 0
        while ts < self.time_sum_:
            ds = self.evaluate(ts, k).norm()
            mean_d += ds
            max_d = max(max_d, ds)
            ts += 0.01
            sample_num += 1
        mean_d /= sample_num
        return mean_d, max_d
    
    @staticmethod
    def waypointsTraj(positions: np.ndarray, start_vel: Vector3d, end_vel: Vector3d, start_acc: Vector3d, end_acc: Vector3d, times: np.ndarray, poly_traj: "PolynomialTraj"):

        seg_num = times.shape[0]
        
        def factorial(x: int) -> int:
            fac = 1
            for i in range(x, 0, -1):
                fac = fac * i
            return fac
        
        Dx = np.zeros((seg_num, 6)).reshape(-1)
        Dy = np.zeros((seg_num, 6)).reshape(-1)
        Dz = np.zeros((seg_num, 6)).reshape(-1)
        
        for k in range(seg_num):
            Dx[k * 6] = positions[k, 0]
            Dy[k * 6] = positions[k, 1]
            Dz[k * 6] = positions[k, 2]
            Dx[k * 6 + 1] = positions[k + 1, 0]
            Dy[k * 6 + 1] = positions[k + 1, 1]
            Dz[k * 6 + 1] = positions[k + 1, 2]
            
            if k == 0:
                Dx[k * 6 + 2] = start_vel.x
                Dy[k * 6 + 2] = start_vel.y
                Dz[k * 6 + 2] = start_vel.z
                Dx[k * 6 + 4] = start_acc.x
                Dy[k * 6 + 4] = start_acc.y
                Dz[k * 6 + 4] = start_acc.z
            elif k == seg_num - 1:
                Dx[k * 6 + 3] = end_vel.x
                Dy[k * 6 + 3] = end_vel.y
                Dz[k * 6 + 3] = end_vel.z
                Dx[k * 6 + 5] = end_acc.x
                Dy[k * 6 + 5] = end_acc.y
                Dz[k * 6 + 5] = end_acc.z
        
        A = np.zeros((seg_num * 6, seg_num * 6))
        for k in range(seg_num):
            Ab = np.zeros((6, 6))
            for i in range(3):
                Ab[2 * i, i] = factorial(i)
                for j in range(i, 6):
                    Ab[2 * i + 1, j] = factorial(j) / factorial(j - i) * pow(times[k], j - i)
            A[k * 6:(k + 1) * 6, k * 6:(k + 1) * 6] = Ab

        num_f = 2 * seg_num + 4  # 2*seg_num for position, 4 for start/end vel/acc
        num_p = 2 * seg_num - 2  # (seg_num - 1)
        num_d = 6 * seg_num
        
        Ct = np.zeros((num_d, num_f + num_p))
        
        # Stack start point constraints
        Ct[0, 0] = 1
        Ct[2, 1] = 1  
        Ct[4, 2] = 1
        Ct[1, 3] = 1
        Ct[3, 2 * seg_num + 4] = 1
        Ct[5, 2 * seg_num + 5] = 1

        # Stack end point constraints
        Ct[6 * (seg_num - 1) + 0, 2 * seg_num + 0] = 1
        Ct[6 * (seg_num - 1) + 1, 2 * seg_num + 1] = 1
        Ct[6 * (seg_num - 1) + 2, 4 * seg_num + 0] = 1
        Ct[6 * (seg_num - 1) + 3, 2 * seg_num + 2] = 1
        Ct[6 * (seg_num - 1) + 4, 4 * seg_num + 1] = 1
        Ct[6 * (seg_num - 1) + 5, 2 * seg_num + 3] = 1

        # Stack intermediate constraints
        for j in range(2, seg_num):
            Ct[6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0] = 1
            Ct[6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1] = 1
            Ct[6 * (j - 1) + 2, 2 * seg_num + 4 + 2 * (j - 2) + 0] = 1
            Ct[6 * (j - 1) + 3, 2 * seg_num + 4 + 2 * (j - 1) + 0] = 1
            Ct[6 * (j - 1) + 4, 2 * seg_num + 4 + 2 * (j - 2) + 1] = 1
            Ct[6 * (j - 1) + 5, 2 * seg_num + 4 + 2 * (j - 1) + 1] = 1

        C = Ct.T

        # Compute transformed coefficients
        Dx1 = C @ Dx
        Dy1 = C @ Dy 
        Dz1 = C @ Dz

        # Matrix mapping coefficient to jerk (J = pTQp)
        Q = np.zeros((seg_num * 6, seg_num * 6))

        for k in range(seg_num):
            for i in range(3, 6):
                for j in range(3, 6):
                    Q[k * 6 + i, k * 6 + j] = (i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2)) / (i + j - 5) * pow(times[k], (i + j - 5))
        
        R = C @ np.linalg.inv(A.T) @ Q @ np.linalg.inv(A) @ Ct
        # Extract fixed derivatives
        Dxf = Dx1[:2 * seg_num + 4]
        Dyf = Dy1[:2 * seg_num + 4] 
        Dzf = Dz1[:2 * seg_num + 4]

        # Extract submatrices from R
        Rff = R[:2 * seg_num + 4, :2 * seg_num + 4]
        Rfp = R[:2 * seg_num + 4, 2 * seg_num + 4:]
        Rpf = R[2 * seg_num + 4:, :2 * seg_num + 4]
        Rpp = R[2 * seg_num + 4:, 2 * seg_num + 4:]

        # Solve for optimal free derivatives
        Dxp = -np.linalg.inv(Rpp) @ Rfp.T @ Dxf
        Dyp = -np.linalg.inv(Rpp) @ Rfp.T @ Dyf
        Dzp = -np.linalg.inv(Rpp) @ Rfp.T @ Dzf

        # Update full derivative vectors
        Dx1[2 * seg_num + 4:] = Dxp
        Dy1[2 * seg_num + 4:] = Dyp
        Dz1[2 * seg_num + 4:] = Dzp

        # Compute polynomial coefficients
        Px = np.linalg.inv(A) @ Ct @ Dx1
        Py = np.linalg.inv(A) @ Ct @ Dy1
        Pz = np.linalg.inv(A) @ Ct @ Dz1

        # Reset trajectory and add segments
        poly_traj.reset()
        for i in range(seg_num):
            # Extract 6 coefficients for each dimension
            cx = Px[i*6:(i+1)*6]
            cy = Py[i*6:(i+1)*6]
            cz = Pz[i*6:(i+1)*6]
            
            # Create polynomial segment and add to trajectory
            poly = Polynomial(cx, cy, cz, times[i])
            poly_traj.addSegment(poly)
        
        return poly_traj

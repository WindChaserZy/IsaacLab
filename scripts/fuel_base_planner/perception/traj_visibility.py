from ..utils import Vector3d, Vector3i
from ..env import RayCaster, EDTEnv
from ..bspline import NonUniformBspline
from typing import List

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

class VisibilityUtil:

    def __init__(self):

        self.visible_num_ = 0
        self.caster_ = RayCaster()
        self.edt_env_ = EDTEnv()
        self.resolution_ = 0.0
        self.offset_ = Vector3d(0.0, 0.0, 0.0)
        ###### Configs ######
        self.min_visib_ = 0.0
        self.max_safe_dist_ = 0.0
        self.safe_margin_ = 0.0
        self.max_acc_ = 0.0
        self.r0_ = 0.0
        self.wnl_ = 0.0
        self.forward_ = 0.0
        #############################

    def setEDTEnvironment(self, edt_env: EDTEnv):
        self.edt_env_ = edt_env
        self.resolution_ = self.edt_env_.sdf_map_.getResolution()
        origin, _ = self.edt_env_.sdf_map_.getRegion()
        self.offset_ = Vector3d(0.5, 0.5, 0.5) - origin / self.resolution_
    
    def getMinDistVoxel(self, q1: Vector3d, q2: Vector3d, offset: Vector3d, res: float):

        self.caster_.setInput(q1 /res, q2 / res)
        qk = Vector3d(0.0, 0.0, 0.0)
        pt_id = Vector3i(0, 0, 0)
        min_id = Vector3i(0, 0, 0)
        min_dist = 1e6

        st, qk = self.caster_.step(qk)
        while st:
            pt_id.x = int(qk.x + offset.x)
            pt_id.y = int(qk.y + offset.y)
            pt_id.z = int(qk.z + offset.z)
            dist = self.edt_env_.sdf_map_.getDistance(pt_id)
            if dist < min_dist:
                min_dist = dist
                min_id = pt_id
            st, qk = self.caster_.step(qk)

        min_pt = self.edt_env_.sdf_map_.indexToPos(min_id)
        return min_pt
    
    def getMinDistVoxelOnLine(self, q1: Vector3d, q2: Vector3d, offset: Vector3d, res: float, state: int, block: Vector3d):

        tmp = Vector3d(0.0, 0.0, 0.0)
        pt_id = Vector3i(0, 0, 0)
        min_dist = 1000000.0
        qb = Vector3d(0.0, 0.0, 0.0)
        grad = Vector3d(0.0, 0.0, 0.0)
        min_pt = Vector3d(0.0, 0.0, 0.0)
        dir = (q2 - q1).normalized()
        norm = (q2 - q1).norm()
        state = 0

        self.caster_.setInput(q1 / res, q2 / res)
        st, tmp = self.caster_.step(tmp)
        while st:
            pt_id.x = int(tmp.x + offset.x)
            pt_id.y = int(tmp.y + offset.y)
            pt_id.z = int(tmp.z + offset.z)
            dist = self.edt_env_.sdf_map_.getDistance(pt_id)

            if dist < min_dist:
                tmp = self.edt_env_.sdf_map_.indexToPos(pt_id)
                if dist < 1e-3:
                    min_dist = dist
                    min_pt = tmp
                    state = -1
                elif dist < 1.2:
                    # projection on the line?
                    dist, grad = self.edt_env_.evaluateEDTWithGrad(tmp, -1, dist, grad)
                    if grad.norm() > 1e-3:
                        qb = tmp - grad.normalized() * dist
                        proj = ((qb.x - q1.x) * dir.x + (qb.y - q1.y) * dir.y + (qb.z - q1.z) * dir.z) / norm
                        if proj > 0.01 and proj < 0.99:
                            min_dist = dist
                            min_pt = tmp
                            state = 1
                            block = qb

            st, tmp = self.caster_.step(tmp)

        return min_pt, state, block
        
    def lineVisib(self, p1: Vector3d, p2: Vector3d, pc: Vector3d):

        ray_pt = Vector3d(0.0, 0.0, 0.0)
        grad = Vector3d(0.0, 0.0, 0.0)
        pt = Vector3d(0.0, 0.0, 0.0)
        pt_id = Vector3i(0, 0, 0)
        dist = 0.0

        self.caster_.setInput(p1 / self.resolution_, p2 / self.resolution_)
        st, ray_pt = self.caster_.step(ray_pt)
        while st:
            pt_id.x = int(ray_pt.x + self.offset_.x)
            pt_id.y = int(ray_pt.y + self.offset_.y) 
            pt_id.z = int(ray_pt.z + self.offset_.z)
            pt = self.edt_env_.sdf_map_.indexToPos(pt_id)
            dist, grad = self.edt_env_.evaluateEDTWithGrad(pt, -1, dist, grad)
            if dist <= self.min_visib_:
                pc = self.edt_env_.sdf_map_.indexToPos(pt_id)
                return False, pc
            st, ray_pt = self.caster_.step(ray_pt)
        return True, pc
    
    def findVisiblePairs(self, pts: List[Vector3d], pairs: List[VisiblePair]):

        know_num = 0
        pairs.clear()
        cur_j = -1
        prev_j = -1

        for i in range(len(pts) - self.visible_num_):
            qi = pts[i]
            qj = Vector3d()
            qb = Vector3d()

            # Find first qj not seen by qi
            cur_j = -1
            for j in range(i + self.visible_num_, len(pts)):
                qj = pts[j]
                visible, qb = self.lineVisib(qi, qj, qb)
                if not visible:
                    cur_j = j
                    break

            if cur_j == -1:
                break  # All points visible, no need to pair

            vpair = VisiblePair()
            vpair.from_ = i
            vpair.to_ = cur_j

            grad = Vector3d()
            dist = 0.0
            dist, grad = self.edt_env_.evaluateEDTWithGrad(qb, -1, dist, grad)

            if grad.norm() > 1e-3:
                grad = grad.normalized()
                dir = (qj - qi).normalized()
                dot_product = grad.x * dir.x + grad.y * dir.y + grad.z * dir.z
                push = grad - dot_product * dir
                push = push.normalized()
                vpair.qb_ = qb - push * 0.2

            pairs.append(vpair)

            if cur_j != prev_j and prev_j != -1:
                break
            prev_j = cur_j
        
        return pairs
    
    def findUnknownPoint(self, traj: NonUniformBspline, point: Vector3d, time: float):

        pt = traj.evaluateDeBoorT(0.0)
        if self.edt_env_.sdf_map_.getOccupancy(pt) == self.edt_env_.sdf_map_.UNKNOWN:  # UNKNOWN
            return False

        duration = traj.getTimeSum()
        found = False
        t = 0.05
        while t <= duration + 1e-3:
            pt = traj.evaluateDeBoorT(t)
            if self.edt_env_.sdf_map_.getOccupancy(pt) == self.edt_env_.sdf_map_.UNKNOWN:  # UNKNOWN
                found = True
                point.x = pt[0]
                point.y = pt[1] 
                point.z = pt[2]
                time = t
                break
            t += 0.05
        if not found:
            return False  # All points are visible

        # Go a little bit forward
        forward = 0.0
        prev = Vector3d(point.x, point.y, point.z)
        while t <= duration + 1e-3:
            cur = traj.evaluateDeBoorT(t)
            forward += (cur - prev).norm()
            if forward > self.forward_:
                point.x = cur[0]
                point.y = cur[1]
                point.z = cur[2]
                time = t
                break
            prev = cur
            t += 0.05
        return True, point, time
    
    def findCriticalPoint(self, traj: NonUniformBspline, unknown_pt: Vector3d, unknown_t: float, pc: Vector3d, tc: float):

        pt = Vector3d(0.0, 0.0, 0.0)
        pb = Vector3d(0.0, 0.0, 0.0)
        tb = -10.0

        # coarse finding backward
        t = unknown_t - 0.2
        while t >= 1e-3:
            pt = traj.evaluateDeBoorT(t)
            if not self.lineVisib(unknown_pt, pt, pb):
                tb = t
                break
            t -= 0.2

        if tb < -5:
            print("all pt visible")
            return False

        # fine finding forward 
        t = tb + 0.01
        while t <= unknown_t + 1e-3:
            pt = traj.evaluateDeBoorT(t)
            if self.lineVisib(unknown_pt, pt, pb):
                break
            tc = t
            pc.x = pt[0]
            pc.y = pt[1]
            pc.z = pt[2]
            t += 0.01

        return True, pc, tc
    
    def findDirAndIdx(self, traj: NonUniformBspline, unknown_t: float, crit_t: float, dir: Vector3d, idx: int, min_pt: Vector3d):

        max_v = 0.0
        vel = traj.getDerivative()
        t = 0
        while t <= unknown_t:
            vt = vel.evaluateDeBoorT(t).norm()
            max_v = max(max_v, vt)
            t += 0.1
        
        dist1 = self.r0_ + self.forward_ + pow(max_v, 2) / (2 * self.max_acc_)
        unknown = traj.evaluateDeBoorT(unknown_t)
        unknown = Vector3d(unknown[0], unknown[1], unknown[2])
        crit = traj.evaluateDeBoorT(crit_t)
        crit = Vector3d(crit[0], crit[1], crit[2])
        v = (unknown - crit).normalized()
        n = 0.0
        while n <= dist1 + 1e-3:
            pv = crit + v * n
            if self.edt_env_.sdf_map_.getDistance(pv) <= self.resolution_:
                dist1 = n - self.safe_margin_
                break
            n += 0.1
        dist1 = min(self.max_safe_dist_, dist1)
        dir.x = v.x * dist1
        dir.y = v.y * dist1
        dir.z = v.z * dist1
        dt = traj.getKnotSpan()
        min_cost = 1e6
        idu = int(unknown_t / dt + 2)
        idb = int(crit_t / dt + 1)
        pts = traj.getControlPoint()
        for i in range(idu, 2, -1):
            pt = Vector3d(pts[i][0], pts[i][1], pts[i][2])
            dot_product = (pt.x - unknown.x) * v.x + (pt.y - unknown.y) * v.y + (pt.z - unknown.z) * v.z
            dl = Vector3d(v.x * dot_product, v.y * dot_product, v.z * dot_product)
            dn = pt - unknown - dl
            cost = dn.x * dn.x + dn.y * dn.y + dn.z * dn.z
            if dl.norm() < dist1:
                cost += self.wnl_ * pow(dl.norm() - dist1, 2)
            if cost < min_cost:
                min_cost = cost
                idx = i
                min_pt = pt
            if idx > idb:
                return False, idx, min_pt
        return True, idx, min_pt
    
    def calcViewConstraint(self, traj: NonUniformBspline, cons: ViewConstraint):

        cons.idx_ = -1
        unknown = Vector3d(0.0, 0.0, 0.0)
        critic = Vector3d(0.0, 0.0, 0.0) 
        dir = Vector3d(0.0, 0.0, 0.0)
        min_pt = Vector3d(0.0, 0.0, 0.0)
        unknown_t = 0.0
        crit_t = 0.0
        idx = 0

        uk, unknown, unknown_t = self.findUnknownPoint(traj, unknown, unknown_t)
        if not uk:
            return cons

        cp, critic, crit_t = self.findCriticalPoint(traj, unknown, unknown_t, critic, crit_t)
        if not cp:
            return cons

        success, idx, min_pt = self.findDirAndIdx(traj, unknown_t, crit_t, dir, idx, min_pt)
        if not success:
            return cons

        cons.pt_ = unknown
        cons.pc_ = critic
        cons.dir_ = dir
        cons.idx_ = idx
        cons.pcons_ = min_pt

        return cons
    
    def precomputeForVisibility(self, ctrl_pts: List[Vector3d], debug: bool):

        n = len(ctrl_pts) - self.visible_num_
        block_pts = [Vector3d() for _ in range(n)]
        unknown_num = 0
        res = self.edt_env_.sdf_map_.getResolution()
        origin, size = self.edt_env_.sdf_map_.getRegion()
        offset = Vector3d(0.5, 0.5, 0.5) - origin / res

        for j in range(len(ctrl_pts)-1, self.visible_num_-1, -1):
            
            qj = ctrl_pts[j]
            i = j - self.visible_num_
            d = self.edt_env_.sdf_map_.getDistance(qj)
            
            if d < 9999:
                if unknown_num < 5:
                    unknown_num += 1
                else:
                    block_pts[i].z = -10086
                    continue

            # method 1:
            qi = ctrl_pts[i]
            block = Vector3d()
            state = 0
            min_pt, state, block = self.getMinDistVoxelOnLine(qi, qj, offset, res, state, block)
            
            if state == 1:
                block_pts[i] = block
            elif state == -1:
                block_pts[i] = self.getVirtualBlockPt(ctrl_pts, i, j, min_pt)
            elif state == 0:
                block_pts[i].z = -10086
            else:
                pass
        
        return block_pts
    
    def getVirtualBlockPt(self, q: List[Vector3d], i: int, j: int, min_pt: Vector3d):

        qij = q[j] - q[i]
        min_dot = 1000.0
        qk = Vector3d()
        qkm = Vector3d()

        # Find point with minimum dot product
        for k in range(i, j+1):
            qkm = q[k] - min_pt
            dot = abs((qkm.x * qij.x + qkm.y * qij.y + qkm.z * qij.z))
            if dot < min_dot:
                min_dot = dot
                qk = q[k]

        # Project and offset
        dir = qij.normalized()
        qk_qi = qk - q[i]
        dot_prod = qk_qi.x * dir.x + qk_qi.y * dir.y + qk_qi.z * dir.z
        qp = q[i] + dir * dot_prod
        
        qp_qk = qp - qk
        qp_qk_norm = qp_qk.normalized()
        qv = qp + qp_qk_norm * 0.1

        return qv

        

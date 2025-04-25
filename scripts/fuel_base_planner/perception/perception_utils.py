from ..utils import Vector3d
from typing import List
import numpy as np
from math import sin, cos, tan

class PerceptionUtils:
    def __init__(self):
        # Camera position and yaw
        self.pos_ = Vector3d()
        self.yaw_ = 0.0
        
        # Camera FOV normals in world frame
        self.normals_ : List[Vector3d] = []

        # config to be added
        self.left_angle_ = -1.0
        self.right_angle_ = -1.0  
        self.top_angle_ = -1.0
        self.max_dist_ = -1.0
        self.vis_dist_ = -1.0
        #####################################

        # FOV plane normals in camera frame
        self.n_top_ = Vector3d(0.0, sin(np.pi/2 - self.top_angle_), cos(np.pi/2 - self.top_angle_))
        self.n_bottom_ = Vector3d(0.0, -sin(np.pi/2 - self.top_angle_), cos(np.pi/2 - self.top_angle_))
        self.n_left_ = Vector3d(sin(self.left_angle_), 0.0, cos(self.left_angle_))
        self.n_right_ = Vector3d(-sin(self.right_angle_), 0.0, cos(self.right_angle_))

        self.T_cb_ = np.array([0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
        self.T_bc_ = np.linalg.inv(self.T_cb_)

        hor = self.vis_dist_ * tan(self.left_angle_)
        vert = self.vis_dist_ * tan(self.top_angle_)
        origin = Vector3d(0.0, 0.0, 0.0)
        left_up = Vector3d(self.vis_dist_, hor, vert)
        left_down = Vector3d(self.vis_dist_, hor, -vert)
        right_up = Vector3d(self.vis_dist_, -hor, vert)
        right_down = Vector3d(self.vis_dist_, -hor, -vert)

        self.cam_vertices1_ : List[Vector3d] = []
        self.cam_vertices2_ : List[Vector3d] = []
       
        self.cam_vertices1_.append(origin)
        self.cam_vertices2_.append(left_up)
        self.cam_vertices1_.append(origin)
        self.cam_vertices2_.append(left_down)
        self.cam_vertices1_.append(origin)
        self.cam_vertices2_.append(right_up)
        self.cam_vertices1_.append(origin)
        self.cam_vertices2_.append(right_down)

        self.cam_vertices1_.append(left_up)
        self.cam_vertices2_.append(right_up)
        self.cam_vertices1_.append(right_up)
        self.cam_vertices2_.append(right_down)
        self.cam_vertices1_.append(right_down)
        self.cam_vertices2_.append(left_down)
        self.cam_vertices1_.append(left_down)
        self.cam_vertices2_.append(left_up)

    def setPose(self, pos: Vector3d, yaw: float):
        """Set camera position and yaw angle"""
        self.pos_ = pos
        self.yaw_ = yaw

        R_wb = np.array([cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1,]).reshape(3, 3)
        pc = self.pos_
        T_wb = np.eye(4)
        T_wb[:3, :3] = R_wb
        T_wb[:3, 3] = pc
        T_wc = np.matmul(T_wb, self.T_bc_)
        R_wc = T_wc[:3, :3]

        self.n_top_ = R_wc @ self.n_top_
        self.n_bottom_ = R_wc @ self.n_bottom_
        self.n_left_ = R_wc @ self.n_left_
        self.n_right_ = R_wc @ self.n_right_

    def getFOV(self, list1: List[Vector3d], list2: List[Vector3d]):
        """Get FOV vertices"""
        list1.clear()
        list2.clear()
        Rwb = np.array([cos(self.yaw_), -sin(self.yaw_), 0, sin(self.yaw_), cos(self.yaw_), 0, 0, 0, 1]).reshape(3, 3)
        for i in range(len(self.cam_vertices1_)):
            list1.append(Rwb @ self.cam_vertices1_[i] + self.pos_)
            list2.append(Rwb @ self.cam_vertices2_[i] + self.pos_)
        return list1, list2

    def insideFOV(self, point: Vector3d) -> bool:
        """Check if point is inside camera FOV"""
        dir = point - self.pos_
        if dir.norm() > self.max_dist_:
            return False
        dir.normalize()
        for normal in self.normals_:
            dot_product = normal.x * dir.x + normal.y * dir.y + normal.z * dir.z
            if dot_product < 0:
                return False
        return True

    def getFOVBoundingBox(self, bmin: Vector3d, bmax: Vector3d):
        """Get axis-aligned bounding box of FOV"""
        left = self.yaw_ + self.left_angle_
        right = self.yaw_ - self.right_angle_
        left_pt = self.pos_ + self.max_dist_ * Vector3d(cos(left), sin(left), 0)
        right_pt = self.pos_ + self.max_dist_ * Vector3d(cos(right), sin(right), 0)
        
        points = [left_pt, right_pt]
        if left > 0 and right < 0:
            points.append(self.pos_ + self.max_dist_ * Vector3d(1, 0, 0))
        elif left > np.pi/2 and right < np.pi/2:
            points.append(self.pos_ + self.max_dist_ * Vector3d(0, 1, 0))
        elif left > -np.pi/2 and right < -np.pi/2:
            points.append(self.pos_ + self.max_dist_ * Vector3d(0, -1, 0))
        elif (left > np.pi and right < np.pi) or (left > -np.pi and right < -np.pi):
            points.append(self.pos_ + self.max_dist_ * Vector3d(-1, 0, 0))  
        
        bmax, bmin = self.pos_, self.pos_
        for p in points:
            for i in range(3):
                if p[i] > bmax[i]:
                    bmax[i] = p[i]
                if p[i] < bmin[i]:
                    bmin[i] = p[i]
        return bmin, bmax

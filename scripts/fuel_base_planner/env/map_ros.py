from .sdf_map import SDFmap
from ..utils import MyTimer, Vector3d
import numpy as np

class MapROS:

    def __init__(self):

        ######### config #########
        self.fx_ = -1.0
        self.fy_ = -1.0
        self.cx_ = -1.0
        self.cy_ = -1.0
        self.depth_filter_maxdist_ = -1.0
        self.depth_filter_mindist_ = -1.0
        self.depth_filter_margin_ = -1
        self.k_depth_scaling_factor_ = -1.0
        self.skip_pixel_ = -1
        self.edsf_slice_height_ = -0.1
        self.visualization_truncate_height_ = -0.1
        self.visualization_truncate_low_ = -0.1
        self.show_occ_time_ = False
        self.show_esdf_time_ = False
        self.show_all_map_ = False
        self.frame_id_ = "world"
        ##########################

        self.proj_points_ = [Vector3d() for _ in range(int(640 * 480 / (self.skip_pixel_ * self.skip_pixel_)))]
        self.point_cloud_ = [Vector3d() for _ in range(int(640 * 480 / (self.skip_pixel_ * self.skip_pixel_)))]
        self.proj_points_cnt = 0

        self.local_updated_ = False
        self.esdf_need_update_ = False
        self.fuse_time_ = 0.0
        self.esdf_time_ = 0.0
        self.max_fuse_time_ = 0.0
        self.max_esdf_time_ = 0.0
        self.fuse_num_ = 0
        self.esdf_num_ = 0
        self.depth_image_ : np.ndarray = np.array([])

        self.camera_pos_ = Vector3d()
        self.camera_q_ = []

        # Random number generation for noise
        self.rand_noise_ = np.random.normal(0, 0.1)
        self.eng_ = np.random.default_rng()
 

    def setMap(self, map: SDFmap):

        self.map_ = map

    def depthImageCallback(self, depth_image: np.ndarray, pos_data):

        self.camera_pos_ = Vector3d(pos_data["position"][0], pos_data["position"][1], pos_data["position"][2])
        self.camera_q_ = [pos_data["orientation"][3], pos_data["orientation"][0], pos_data["orientation"][1], pos_data["orientation"][2]]

        if not self.map_.isInMap(self.camera_pos_):
            return

        self.depth_image_ = self.preprocessDepthImage(depth_image)
        self.processDepthImage()

        self.map_.inputPointCloud(self.point_cloud_, self.proj_points_cnt, self.camera_pos_)
        if self.local_updated_:
            self.map_.clearAndInflateLocalMap()
            self.esdf_need_update_ = True
            self.local_updated_ = False

        

    def preprocessDepthImage(self, depth_image: np.ndarray):

        return depth_image
    
    def processDepthImage(self):

        self.proj_points_cnt = 0
        
        cols = self.depth_image_.shape[1]
        rows = self.depth_image_.shape[0]
        
        # Convert quaternion to rotation matrix
        q = self.camera_q_
        camera_r = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])

        inv_factor = 1.0 / self.k_depth_scaling_factor_

        for v in range(self.depth_filter_margin_, rows - self.depth_filter_margin_, self.skip_pixel_):
            for u in range(self.depth_filter_margin_, cols - self.depth_filter_margin_, self.skip_pixel_):
                depth = self.depth_image_[v,u] * inv_factor

                if self.depth_image_[v,u] == 0 or depth > self.depth_filter_maxdist_:
                    depth = self.depth_filter_maxdist_
                elif depth < self.depth_filter_mindist_:
                    continue

                # Calculate point in camera frame
                pt_cur = np.array([
                    (u - self.cx_) * depth / self.fx_,
                    (v - self.cy_) * depth / self.fy_,
                    depth
                ])

                # Transform to world frame
                pt_world = camera_r @ pt_cur + np.array([self.camera_pos_.x, self.camera_pos_.y, self.camera_pos_.z])

                # Store point
                self.point_cloud_[self.proj_points_cnt].x = pt_world[0]
                self.point_cloud_[self.proj_points_cnt].y = pt_world[1] 
                self.point_cloud_[self.proj_points_cnt].z = pt_world[2]
                self.proj_points_cnt += 1
        

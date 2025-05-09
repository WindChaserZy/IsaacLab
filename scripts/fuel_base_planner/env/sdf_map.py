from collections import deque
import numpy as np
import math
from array import array
from enum import Enum
from typing import List, Optional, Tuple, Union
from .raycast import RayCaster
from ..utils import Vector3d, Vector3i

def logit(x):
    return np.log(x / (1 - x) )

class MapParam:
    def __init__(self):
        # map properties
        self.map_origin_ = Vector3d()
        self.map_size_ = Vector3d()
        self.map_min_boundary_ = Vector3d()
        self.map_max_boundary_ = Vector3d()
        self.map_voxel_num_ = Vector3i()
        self.resolution_ = 0.0
        self.resolution_inv_ = 0.0
        self.obstacles_inflation_ = 0.0
        self.virtual_ceil_height_ = 0.0
        self.ground_height_ = 0.0
        self.box_min_ = Vector3i()
        self.box_max_ = Vector3i()
        self.box_mind_ = Vector3d()
        self.box_maxd_ = Vector3d()
        self.default_dist_ = 0.0
        self.optimistic_ = False
        self.signed_dist_ = False

        # map fusion
        self.p_hit_ = 0.0
        self.p_miss_ = 0.0
        self.p_min_ = 0.0
        self.p_max_ = 0.0
        self.p_occ_ = 0.0
        self.prob_hit_log_ = 0.0
        self.prob_miss_log_ = 0.0
        self.clampmin_log_ = 0.0
        self.clampmax_log_ = 0.0
        self.min_occupancy_log_ = 0.0
        self.max_ray_length_ = 0.0
        self.local_bound_inflate_ = 0.0
        self.local_map_margin_ = 0
        self.unknown_flag_ = 0.0

class MapData:
    def __init__(self):

        self.occupancy_buffer_: list[float] = []
        self.occupancy_buffer_inflate_: array = array('b', []) 
        self.distance_buffer_neg_: list[float] = []
        self.distance_buffer_: list[float] = []
        self.tmpbuffer1_: list[float] = []
        self.tmpbuffer2_: list[float] = []

        self.count_hit_: list[int] = []
        self.count_miss_: list[int] = []
        self.count_hit_and_miss_: list[int] = []
        self.flag_rayend_: array = array('b', [])
        self.flag_visited_: array = array('b', [])
        self.raycast_num_: int = 0
        self.cache_voxel_: deque[int] = deque()
        self.local_bound_min_ = Vector3i()
        self.local_bound_max_ = Vector3i()
        self.update_min_ = Vector3d()
        self.update_max_ = Vector3d()
        self.reset_updated_box_ = False

class SDFmap:
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2

    def __init__(self):
        self.mp = MapParam()
        self.md = MapData()
        self.caster_ = RayCaster()
        self.initMap()

    def initMap(self): 
        
        # Config to be added
        x_size, y_size, z_size = 50, 50, 10
        self.mp.resolution_ = 0.1
        self.mp.obstacles_inflation_ = 0.199
        self.mp.local_bound_inflate_ = 0.5
        self.mp.local_map_margin_ = 50
        self.mp.ground_height_ = -1.0
        self.mp.default_dist_ = 0.0
        self.mp.optimistic_ = False
        self.mp.signed_dist_ = False

        self.mp.local_bound_inflate_ = max(self.mp.resolution_, self.mp.local_bound_inflate_)
        self.mp.resolution_inv_ = 1 / self.mp.resolution_
        self.mp.map_origin_ = Vector3d(-x_size / 2, -y_size / 2, self.mp.ground_height_)
        self.mp.map_size_ = Vector3d(x_size, y_size, z_size)
        self.mp.map_voxel_num_.x = int(np.ceil(self.mp.map_size_.x / self.mp.resolution_))
        self.mp.map_voxel_num_.y = int(np.ceil(self.mp.map_size_.y / self.mp.resolution_))
        self.mp.map_voxel_num_.z = int(np.ceil(self.mp.map_size_.z / self.mp.resolution_))
        self.mp.map_min_boundary_ = self.mp.map_origin_
        self.mp.map_max_boundary_ = self.mp.map_origin_ + self.mp.map_size_

        self.mp.p_hit_ = 0.65
        self.mp.p_miss_ = 0.35
        self.mp.p_min_ = 0.12
        self.mp.p_max_ = 0.90
        self.mp.p_occ_ = 0.80
        self.mp.max_ray_length_ = 4.5
        self.mp.virtual_ceil_height_ = -10
        #########################################################
        self.mp.prob_hit_log_ = logit(self.mp.p_hit_)
        self.mp.prob_miss_log_ = logit(self.mp.p_miss_)
        self.mp.clampmin_log_ = logit(self.mp.p_min_)
        self.mp.clampmax_log_ = logit(self.mp.p_max_)

        self.mp.min_occupancy_log_ = logit(self.mp.p_min_)
        self.mp.unknown_flag_ = 0.01

        buffer_size = int(self.mp.map_voxel_num_.x * self.mp.map_voxel_num_.y * self.mp.map_voxel_num_.z)
        self.md.occupancy_buffer_ = [self.mp.clampmin_log_ - self.mp.unknown_flag_] * buffer_size
        self.md.occupancy_buffer_inflate_ = array('b', [0] * buffer_size)
        self.md.distance_buffer_neg_ = [self.mp.default_dist_] * buffer_size
        self.md.distance_buffer_ = [self.mp.default_dist_] * buffer_size
        self.md.count_hit_and_miss_ = [0] * buffer_size
        self.md.count_hit_ = [0] * buffer_size
        self.md.count_miss_ = [0] * buffer_size
        self.md.flag_rayend_ = array('b', [-1] * buffer_size)
        self.md.flag_visited_ = array('b', [-1] * buffer_size)
        self.md.tmpbuffer1_ = [0.0] * buffer_size
        self.md.tmpbuffer2_ = [0.0] * buffer_size

        self.md.raycast_num_ = 0
        self.md.reset_updated_box_ = True
        self.md.update_min_ = Vector3d(0, 0, 0)
        self.md.update_max_ = Vector3d(0, 0, 0)

        # Config to be added
        self.mp.box_mind_.x = -10.0
        self.mp.box_mind_.y = -15.0
        self.mp.box_mind_.z = 0.0
        self.mp.box_maxd_.x = 10.0
        self.mp.box_maxd_.y = 15.0
        self.mp.box_maxd_.z = 2.0
        #########################################################

        self.mp.box_min_ = self.posToIndex(self.mp.box_mind_)
        self.mp.box_max_ = self.posToIndex(self.mp.box_maxd_)

        self.caster_ = RayCaster()
        self.caster_.setParams(self.mp.resolution_, self.mp.map_origin_)

    def posToIndex(self, pos: Vector3d) -> Vector3i:
        id = Vector3i()
        id.x = int(np.floor((pos.x - self.mp.map_origin_.x) * self.mp.resolution_inv_))
        id.y = int(np.floor((pos.y - self.mp.map_origin_.y) * self.mp.resolution_inv_))
        id.z = int(np.floor((pos.z - self.mp.map_origin_.z) * self.mp.resolution_inv_))
        return id

    def indexToPos(self, index: Vector3i) -> Vector3d:
        pos = Vector3d()
        pos.x = (index.x + 0.5) * self.mp.resolution_ + self.mp.map_origin_.x
        pos.y = (index.y + 0.5) * self.mp.resolution_ + self.mp.map_origin_.y
        pos.z = (index.z + 0.5) * self.mp.resolution_ + self.mp.map_origin_.z
        return pos
    
    def boundIndex(self, index: Vector3i):
        index.x = max(0, min(index.x, self.mp.map_voxel_num_.x - 1))
        index.y = max(0, min(index.y, self.mp.map_voxel_num_.y - 1))
        index.z = max(0, min(index.z, self.mp.map_voxel_num_.z - 1))
        return index
    
    def toAddress(self, x: int, y: int, z: int):
        return x * self.mp.map_voxel_num_.y * self.mp.map_voxel_num_.z + y * self.mp.map_voxel_num_.z + z
    
    def isInMap(self, pos: Union[Vector3d, Vector3i]):
        if isinstance(pos, Vector3d):
            return pos.x >= self.mp.map_min_boundary_.x + 1e-4 and pos.x <= self.mp.map_max_boundary_.x - 1e-4 and \
                   pos.y >= self.mp.map_min_boundary_.y + 1e-4 and pos.y <= self.mp.map_max_boundary_.y - 1e-4 and \
                   pos.z >= self.mp.map_min_boundary_.z + 1e-4 and pos.z <= self.mp.map_max_boundary_.z - 1e-4
        elif isinstance(pos, Vector3i):
            return pos.x >= 0 and pos.x <= self.mp.map_voxel_num_.x - 1 and \
                   pos.y >= 0 and pos.y <= self.mp.map_voxel_num_.y - 1 and \
                   pos.z >= 0 and pos.z <= self.mp.map_voxel_num_.z - 1
    
    def boundBox(self, low: Vector3d, up: Vector3d):
        low.x = max(low.x, self.mp.box_mind_.x)
        low.y = max(low.y, self.mp.box_mind_.y)
        low.z = max(low.z, self.mp.box_mind_.z)
        up.x = min(up.x, self.mp.box_maxd_.x)
        up.y = min(up.y, self.mp.box_maxd_.y)
        up.z = min(up.z, self.mp.box_maxd_.z)
        return low, up
    
    def getOccupancy(self, pos: Union[Vector3d, Vector3i]):
        if isinstance(pos, Vector3d):
            idx = self.posToIndex(pos)
            return self.getOccupancy(idx)
        elif isinstance(pos, Vector3i):
            if not self.isInMap(pos):
                return -1
            occ = self.md.occupancy_buffer_[self.toAddress(pos.x, pos.y, pos.z)]
            if occ < self.mp.clampmin_log_ - 1e-3:
                return self.UNKNOWN
            elif occ > self.mp.min_occupancy_log_:
                return self.OCCUPIED
            else:
                return self.FREE
            
    def getInflatedOccupancy(self, pos: Union[Vector3d, Vector3i]):
        if isinstance(pos, Vector3d):
            idx = self.posToIndex(pos)
            return self.getInflatedOccupancy(idx)
        elif isinstance(pos, Vector3i):
            if not self.isInMap(pos):
                return -1
            return int(self.md.occupancy_buffer_inflate_[self.toAddress(pos.x, pos.y, pos.z)])

    def setOccupied(self, pos: Vector3d, occ: int):
        if not self.isInMap(pos):
            return
        idx = self.posToIndex(pos)
        self.md.occupancy_buffer_inflate_[self.toAddress(idx.x, idx.y, idx.z)] = occ

    def inflatePoint(self, pt: Vector3i, inf_step: int, inf_pts: list[Vector3i]):
        n = 0
        for dx in range(-inf_step, inf_step + 1):
            for dy in range(-inf_step, inf_step + 1):
                for dz in range(-inf_step, inf_step + 1):
                    inf_pts[n] = Vector3i(pt.x + dx, pt.y + dy, pt.z + dz)
                    n += 1
        return inf_pts
    
    def getDistance(self, pos: Union[Vector3d, Vector3i]):
        if isinstance(pos, Vector3d):
            idx = self.posToIndex(pos)
            return self.getDistance(idx)
        elif isinstance(pos, Vector3i):
            if not self.isInMap(pos):
                return -1
            addr = self.toAddress(pos.x, pos.y, pos.z)
            return self.md.distance_buffer_[addr]

    def resetBufferZero(self):
        self.resetBuffer(self.mp.map_min_boundary_, self.mp.map_max_boundary_)
        self.md.local_bound_min_ = Vector3i(0, 0, 0)
        self.md.local_bound_max_ = Vector3i(
            self.mp.map_voxel_num_.x - 1,
            self.mp.map_voxel_num_.y - 1, 
            self.mp.map_voxel_num_.z - 1
        )

    def resetBuffer(self, min_pos = None, max_pos = None):
        if min_pos == None or max_pos == None:
            self.resetBufferZero()
            return
        
        min_id = self.posToIndex(min_pos)
        max_id = self.posToIndex(max_pos)
        min_id = self.boundIndex(min_id)
        max_id = self.boundIndex(max_id)
        for x in range(min_id.x, max_id.x + 1):
            for y in range(min_id.y, max_id.y + 1):
                for z in range(min_id.z, max_id.z + 1):
                    addr = self.toAddress(x, y, z)
                    self.md.occupancy_buffer_inflate_[addr] = 0
                    self.md.distance_buffer_[addr] = self.mp.default_dist_

    def fillESDF(self, f_get_val, f_set_val, start: int, end: int, dim: int):

        if dim == 0:
            v = [0.0] * self.mp.map_voxel_num_.x
            z = [0.0] * (self.mp.map_voxel_num_.x + 1)

        elif dim == 1:
            v = [0.0] * self.mp.map_voxel_num_.y
            z = [0.0] * (self.mp.map_voxel_num_.y + 1)

        elif dim == 2:
            v = [0.0] * self.mp.map_voxel_num_.z
            z = [0.0] * (self.mp.map_voxel_num_.z + 1)
        else:
            v = [0.0] * self.mp.map_voxel_num_.x
            z = [0.0] * (self.mp.map_voxel_num_.x + 1)
            
        k = start
        v[start] = start
        z[start] = float('-inf')
        z[start + 1] = float('inf')

        for q in range(start + 1, end + 1):
            k += 1
            while True:
                k -= 1
                s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k])
                if s > z[k]:
                    break

            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = float('inf')

        k = start

        for q in range(start, end + 1):
            while z[k + 1] < q:
                k += 1
            val = (q - v[k]) * (q - v[k]) + f_get_val(v[k])
            f_set_val(q, val)

    def updateESDF3d(self):

        min_esdf = self.md.local_bound_min_
        max_esdf = self.md.local_bound_max_

        if self.mp.optimistic_:

            for x in range(min_esdf.x, max_esdf.x + 1):
                for y in range(min_esdf.y, max_esdf.y + 1):
                    def get_val_z(z):
                        addr = self.toAddress(x, y, z)
                        return 0 if self.md.occupancy_buffer_inflate_[addr] == 1 else float('inf')
                    def set_val_z(z, val):
                        addr = self.toAddress(x, y, z)
                        self.md.tmpbuffer1_[addr] = val

                    self.fillESDF(get_val_z, set_val_z, min_esdf.z, max_esdf.z, 2)
        else:
            for x in range(min_esdf.x, max_esdf.x + 1):
                for y in range(min_esdf.y, max_esdf.y + 1):
                    def get_val_z(z):
                        addr = self.toAddress(x, y, z)
                        return 0 if (self.md.occupancy_buffer_inflate_[addr] == 1 or 
                                   self.md.occupancy_buffer_[addr] < self.mp.clampmin_log_ - 1e-3) else float('inf')
                    def set_val_z(z, val):
                        addr = self.toAddress(x, y, z)
                        self.md.tmpbuffer1_[addr] = val

                    self.fillESDF(get_val_z, set_val_z, min_esdf.z, max_esdf.z, 2)
                        
        # Fill ESDF in y direction
        for x in range(min_esdf.x, max_esdf.x + 1):
            for z in range(min_esdf.z, max_esdf.z + 1):
                def get_val_y(y):
                    addr = self.toAddress(x, y, z)
                    return self.md.tmpbuffer1_[addr]
                def set_val_y(y, val):
                    addr = self.toAddress(x, y, z)
                    self.md.tmpbuffer2_[addr] = val

                self.fillESDF(get_val_y, set_val_y, min_esdf.y, max_esdf.y, 1)

        # Fill ESDF in x direction and scale by resolution
        for y in range(min_esdf.y, max_esdf.y + 1):
            for z in range(min_esdf.z, max_esdf.z + 1):
                def get_val_x(x):
                    addr = self.toAddress(x, y, z)
                    return self.md.tmpbuffer2_[addr]
                def set_val_x(x, val):
                    addr = self.toAddress(x, y, z)
                    self.md.distance_buffer_[addr] = self.mp.resolution_ * math.sqrt(val)

                self.fillESDF(get_val_x, set_val_x, min_esdf.x, max_esdf.x, 0)
        
        if self.mp.signed_dist_:

            # Fill negative ESDF in z direction
            for x in range(min_esdf.x, max_esdf.x + 1):
                for y in range(min_esdf.y, max_esdf.y + 1):
                    def get_val_z(z):
                        addr = self.toAddress(x, y, z)
                        return 0 if self.md.occupancy_buffer_inflate_[addr] == 0 else float('inf')
                    def set_val_z(z, val):
                        addr = self.toAddress(x, y, z)
                        self.md.tmpbuffer1_[addr] = val

                    self.fillESDF(get_val_z, set_val_z, min_esdf.z, max_esdf.z, 2)

            # Fill negative ESDF in y direction
            for x in range(min_esdf.x, max_esdf.x + 1):
                for z in range(min_esdf.z, max_esdf.z + 1):
                    def get_val_y(y):
                        addr = self.toAddress(x, y, z)
                        return self.md.tmpbuffer1_[addr]
                    def set_val_y(y, val):
                        addr = self.toAddress(x, y, z)
                        self.md.tmpbuffer2_[addr] = val

                    self.fillESDF(get_val_y, set_val_y, min_esdf.y, max_esdf.y, 1)

            # Fill negative ESDF in x direction and scale by resolution
            for y in range(min_esdf.y, max_esdf.y + 1):
                for z in range(min_esdf.z, max_esdf.z + 1):
                    def get_val_x(x):
                        addr = self.toAddress(x, y, z)
                        return self.md.tmpbuffer2_[addr]
                    def set_val_x(x, val):
                        addr = self.toAddress(x, y, z)
                        self.md.distance_buffer_neg_[addr] = self.mp.resolution_ * math.sqrt(val)

                    self.fillESDF(get_val_x, set_val_x, min_esdf.x, max_esdf.x, 0)

            # Merge negative distance with positive
            for x in range(min_esdf.x, max_esdf.x + 1):
                for y in range(min_esdf.y, max_esdf.y + 1):
                    for z in range(min_esdf.z, max_esdf.z + 1):
                        idx = self.toAddress(x, y, z)
                        if self.md.distance_buffer_neg_[idx] > 0.0:
                            self.md.distance_buffer_[idx] += (-self.md.distance_buffer_neg_[idx] + self.mp.resolution_)
    
    def setCacheOccupancy(self, adr: int, occ: int):

        if self.md.count_hit_[adr] == 0 and self.md.count_miss_[adr] == 0:
            self.md.cache_voxel_.append(adr)

        if occ == 1:
            self.md.count_hit_[adr] += 1
        elif occ == 0:
            self.md.count_miss_[adr] = 1

    def closestPointInMap(self, pt: Vector3d, camera_pt: Vector3d) -> Vector3d:
        diff = pt - camera_pt
        max_tc = self.mp.map_max_boundary_ - camera_pt
        min_tc = self.mp.map_min_boundary_ - camera_pt
        min_t = 1000000
        
        if abs(diff.x) > 0:
            t1 = max_tc.x / diff.x
            if t1 > 0 and t1 < min_t:
                min_t = t1
            t2 = min_tc.x / diff.x
            if t2 > 0 and t2 < min_t:
                min_t = t2
            
        if abs(diff.y) > 0:
            t1 = max_tc.y / diff.y
            if t1 > 0 and t1 < min_t:
                min_t = t1
            t2 = min_tc.y / diff.y
            if t2 > 0 and t2 < min_t:
                min_t = t2
            
        if abs(diff.z) > 0:
            t1 = max_tc.z / diff.z
            if t1 > 0 and t1 < min_t:
                min_t = t1
            t2 = min_tc.z / diff.z
            if t2 > 0 and t2 < min_t:
                min_t = t2
            
            
        return camera_pt + diff * (min_t - 1e-3)

    def inputPointCloud(self, points: list[Vector3d], point_num: int, camera_pose: Vector3d):

        if point_num == 0:
            return
        
        self.md.raycast_num_ += 1

        update_min, update_max = camera_pose, camera_pose
        if self.md.reset_updated_box_:
            self.md.update_min_ = update_min
            self.md.update_max_ = update_max
            self.md.reset_updated_box_ = False
        
        tmp_flag = 0
        for i in range(point_num):
            pt_w = points[i]
            # Find closest point in map if point is outside
            if not self.isInMap(pt_w):
                # Find closest point in map and set free
                pt_w = self.closestPointInMap(pt_w, camera_pose)
                length = (pt_w - camera_pose).norm()
                if length > self.mp.max_ray_length_:
                    pt_w = camera_pose + (pt_w - camera_pose) * (self.mp.max_ray_length_ / length)
                if pt_w.z < 0.2:
                    continue
                tmp_flag = 0
            else:
                length = (pt_w - camera_pose).norm()
                if length > self.mp.max_ray_length_:
                    pt_w = camera_pose + (pt_w - camera_pose) * (self.mp.max_ray_length_ / length)
                    if pt_w.z < 0.2:
                        continue
                    tmp_flag = 0
                else:
                    tmp_flag = 1
        
            idx = self.posToIndex(pt_w)
            vox_adr = self.toAddress(idx.x, idx.y, idx.z)
            self.setCacheOccupancy(vox_adr, tmp_flag)

            for k in range(3):
                if k == 0:
                    update_min.x = min(update_min.x, pt_w.x)
                    update_max.x = max(update_max.x, pt_w.x)
                elif k == 1:
                    update_min.y = min(update_min.y, pt_w.y)
                    update_max.y = max(update_max.y, pt_w.y)
                else:
                    update_min.z = min(update_min.z, pt_w.z)
                    update_max.z = max(update_max.z, pt_w.z)

            # Raycasting between camera center and point
            if self.md.flag_rayend_[vox_adr] == self.md.raycast_num_:
                continue
            else:
                self.md.flag_rayend_[vox_adr] = self.md.raycast_num_

            self.caster_.input(pt_w, camera_pose)
            self.caster_.nextId(idx)
            while self.caster_.nextId(idx):
                self.setCacheOccupancy(self.toAddress(idx.x, idx.y, idx.z), 0)

        bound_inf = Vector3d(self.mp.local_bound_inflate_, self.mp.local_bound_inflate_, 0)
        self.md.local_bound_min_ = self.posToIndex(update_min - bound_inf)
        self.md.local_bound_max_ = self.posToIndex(update_max + bound_inf)
        self.md.local_bound_min_ = self.boundIndex(self.md.local_bound_min_)
        self.md.local_bound_max_ = self.boundIndex(self.md.local_bound_max_)

        for k in range(3):
            if k == 0:
                self.md.update_min_.x = min(update_min.x, self.md.update_min_.x)
                self.md.update_max_.x = max(update_max.x, self.md.update_max_.x)
            elif k == 1:
                self.md.update_min_.y = min(update_min.y, self.md.update_min_.y)
                self.md.update_max_.y = max(update_max.y, self.md.update_max_.y)
            else:
                self.md.update_min_.z = min(update_min.z, self.md.update_min_.z)
                self.md.update_max_.z = max(update_max.z, self.md.update_max_.z)

        while len(self.md.cache_voxel_) > 0:
            adr = self.md.cache_voxel_.popleft()
            log_odds_update = self.mp.prob_hit_log_ if self.md.count_hit_[adr] >= self.md.count_miss_[adr] else self.mp.prob_miss_log_
            self.md.count_hit_[adr] = 0
            self.md.count_miss_[adr] = 0

            if self.md.occupancy_buffer_[adr] < self.mp.clampmin_log_ - 1e-3:
                self.md.occupancy_buffer_[adr] = self.mp.min_occupancy_log_

            self.md.occupancy_buffer_[adr] = min(
                max(self.md.occupancy_buffer_[adr] + log_odds_update, self.mp.clampmin_log_),
                self.mp.clampmax_log_)
        
    def clearAndInflateLocalMap(self):

        inf_step = np.ceil(self.mp.obstacles_inflation_ / self.mp.resolution_)
        inf_pts = [Vector3i()] * int(math.pow(2 * inf_step + 1, 3))

        # Clear inflation in local bound
        for x in range(self.md.local_bound_min_.x, self.md.local_bound_max_.x + 1):
            for y in range(self.md.local_bound_min_.y, self.md.local_bound_max_.y + 1):
                for z in range(self.md.local_bound_min_.z, self.md.local_bound_max_.z + 1):
                    self.md.occupancy_buffer_inflate_[self.toAddress(x, y, z)] = 0

       
        for x in range(self.md.local_bound_min_.x, self.md.local_bound_max_.x + 1):
            for y in range(self.md.local_bound_min_.y, self.md.local_bound_max_.y + 1):
                for z in range(self.md.local_bound_min_.z, self.md.local_bound_max_.z + 1):
                    id1 = self.toAddress(x, y, z)
                    if self.md.occupancy_buffer_[id1] > self.mp.min_occupancy_log_:
                        # Inflate point
                        inf_pts = self.inflatePoint(Vector3i(x, y, z), inf_step, inf_pts)
                        for inf_pt in inf_pts:
                            idx_inf = self.toAddress(inf_pt.x, inf_pt.y, inf_pt.z)
                            if idx_inf >= 0 and idx_inf < (self.mp.map_voxel_num_.x * 
                                                         self.mp.map_voxel_num_.y * 
                                                         self.mp.map_voxel_num_.z):
                                self.md.occupancy_buffer_inflate_[idx_inf] = 1

        if self.mp.virtual_ceil_height_ > -0.5:
            ceil_id = int(np.floor((self.mp.virtual_ceil_height_ - self.mp.map_origin_.z) * self.mp.resolution_inv_))
            for x in range(self.md.local_bound_min_.x, self.md.local_bound_max_.x + 1):
                for y in range(self.md.local_bound_min_.y, self.md.local_bound_max_.y + 1):
                    addr = self.toAddress(x, y, ceil_id)
                    self.md.occupancy_buffer_[addr] = self.mp.clampmax_log_

    def getResolution(self):
        return self.mp.resolution_
    
    def getVoxelNum(self):
        return self.mp.map_voxel_num_.x * self.mp.map_voxel_num_.y * self.mp.map_voxel_num_.z
                        
    def getRegion(self):
        return self.mp.map_origin_, self.mp.map_size_
    
    def getBox(self):
        return self.mp.box_mind_, self.mp.box_maxd_
    
    def getUpdatedBox(self, reset: bool):
        if reset:
            self.md.reset_updated_box_ = True
        return self.md.update_min_, self.md.update_max_
    
    def getDistWithGrad(self, pos: Vector3d, grad: Vector3d):
        if not self.isInMap(pos):
            grad.x = 0
            grad.y = 0 
            grad.z = 0
            return 0

        # trilinear interpolation
        pos_m = pos - 0.5 * self.mp.resolution_ * Vector3d(1, 1, 1)
        idx = self.posToIndex(pos_m)
        idx_pos = self.indexToPos(idx)
        diff = Vector3d((pos - idx_pos).x * self.mp.resolution_inv_,
                       (pos - idx_pos).y * self.mp.resolution_inv_,
                       (pos - idx_pos).z * self.mp.resolution_inv_)

        values = [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    current_idx = Vector3i(idx.x + x, idx.y + y, idx.z + z)
                    values[x][y][z] = self.getDistance(current_idx)

        v00 = (1 - diff.x) * values[0][0][0] + diff.x * values[1][0][0]
        v01 = (1 - diff.x) * values[0][0][1] + diff.x * values[1][0][1]
        v10 = (1 - diff.x) * values[0][1][0] + diff.x * values[1][1][0]
        v11 = (1 - diff.x) * values[0][1][1] + diff.x * values[1][1][1]
        v0 = (1 - diff.y) * v00 + diff.y * v10
        v1 = (1 - diff.y) * v01 + diff.y * v11
        dist = (1 - diff.z) * v0 + diff.z * v1

        grad.z = (v1 - v0) * self.mp.resolution_inv_
        grad.y = ((1 - diff.z) * (v10 - v00) + diff.z * (v11 - v01)) * self.mp.resolution_inv_
        grad.x = (1 - diff.z) * (1 - diff.y) * (values[1][0][0] - values[0][0][0])
        grad.x += (1 - diff.z) * diff.y * (values[1][1][0] - values[0][1][0])
        grad.x += diff.z * (1 - diff.y) * (values[1][0][1] - values[0][0][1])
        grad.x += diff.z * diff.y * (values[1][1][1] - values[0][1][1])
        grad.x *= self.mp.resolution_inv_

        return dist, grad

    def isInBox(self, pos: Union[Vector3d, Vector3i]):
        if isinstance(pos, Vector3d):
            for i in range(3):
                if pos[i] < self.mp.box_mind_[i] or pos[i] > self.mp.box_maxd_[i]:
                    return False
            return True
        elif isinstance(pos, Vector3i):
            for i in range(3):
                if pos[i] < self.mp.box_min_[i] or pos[i] > self.mp.box_max_[i]:
                    return False
            return True

            
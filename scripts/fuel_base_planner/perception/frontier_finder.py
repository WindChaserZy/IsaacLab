import numpy as np
import math
from typing import List, Tuple
from ..env import EDTEnv, RayCaster, SDFmap
from ..utils import Vector3d, Vector3i
from .perception_utils import PerceptionUtils
from .graph_node import ViewNode

class Viewpoint:
    def __init__(self, pos: Vector3d, yaw: float, visib_num: int):
        self.pos_ = pos           
        self.yaw_ = yaw           
        self.visib_num_ = visib_num  

class Frontier:
    def __init__(self):
        self.cells_: List[Vector3d] = []             
        self.filtered_cells_: List[Vector3d] = []   
        self.average_: Vector3d = Vector3d()        
        self.id_: int = -1                           
        self.viewpoints_: List[Viewpoint] = []       
        self.box_min_: Vector3d = Vector3d()        
        self.box_max_: Vector3d = Vector3d()       
        self.paths_: List[List[Vector3d]] = []       
        self.costs_: List[float] = []

class FrontierFinder:

    def __init__(self, edt_env: EDTEnv = EDTEnv()):
        
        self.frontier_flag_: List[str] = []
        self.frontiers_: List[Frontier] = []
        self.dormant_frontiers_: List[Frontier] = []
        self.tmp_frontiers_: List[Frontier] = []
        self.removed_ids_: List[int] = []
        self.next_frontier_: Frontier = Frontier()
        self.first_new_ftr_ = Frontier() 
    
        self.edt_env_: EDTEnv = edt_env
        self.raycaster_: RayCaster = RayCaster()
        self.percep_utils_: PerceptionUtils = PerceptionUtils()

        self.init()

    def init(self):
        
        
        voxel_num = self.edt_env_.sdf_map_.getVoxelNum()
        self.frontier_flag_ = ['0'] * voxel_num
        #### config to be added
        self.cluster_min_ = -1
        self.cluster_size_xy_ = -1.0
        self.cluster_size_z_ = -1.0
        self.min_candidate_dist_ = -1.0
        self.min_candidate_clearance_ = -1.0
        self.candidate_dphi_ = -1.0
        self.candidate_rmax_ = -1.0
        self.candidate_rmin_ = -1.0
        self.candidate_rnum_ = -1
        self.down_sample_ = -1
        self.min_visib_num_ = -1
        self.min_view_finish_fraction_ = -1.0
        #################################
        self.resolution_ = self.edt_env_.sdf_map_.getResolution()
        origin, size = self.edt_env_.sdf_map_.getRegion()
        self.raycaster_.setParams(self.resolution_, origin)

    def searchFrontiers(self):
        self.tmp_frontiers_.clear()
        update_min, update_max = self.edt_env_.sdf_map_.getUpdatedBox(True)

        def resetFlag(iter: Frontier, frontiers: List[Frontier]):
            for cell in iter.cells_:
                idx = self.edt_env_.sdf_map_.posToIndex(cell)
                self.frontier_flag_[self.toadr(idx)] = str(0)
            idx = frontiers.index(iter)
            frontiers.remove(iter)
            if idx < len(frontiers):
                iter = frontiers[idx]
            else:
                iter = Frontier()
            return iter, frontiers
        
        self.removed_ids_.clear()

        i = 0
        rmv_idx = 0
        while i < len(self.frontiers_):
            if (self.haveOverlap(self.frontiers_[i].box_min_, self.frontiers_[i].box_max_, update_min, update_max) and 
                self.isFrontierChanged(self.frontiers_[i])):
                iter, self.frontiers_ = resetFlag(self.frontiers_[i], self.frontiers_)
                self.removed_ids_.append(rmv_idx)
            else:
                i += 1
                rmv_idx += 1
        
        iter = 0
        while iter < len(self.dormant_frontiers_):
            if (self.haveOverlap(self.dormant_frontiers_[iter].box_min_, self.dormant_frontiers_[iter].box_max_, update_min, update_max) and
                self.isFrontierChanged(self.dormant_frontiers_[iter])):
                self.dormant_frontiers_[iter], self.dormant_frontiers_ = resetFlag(self.dormant_frontiers_[iter], self.dormant_frontiers_)
            else:
                iter += 1
        
        search_min = update_min - Vector3d(1, 1, 0.5)
        search_max = update_max + Vector3d(1, 1, 0.5)
        box_min, box_max = self.edt_env_.sdf_map_.getBox()
        for k in range(3):
            search_min[k] = max(search_min[k], box_min[k])
            search_max[k] = min(search_max[k], box_max[k])
        
        min_id = self.edt_env_.sdf_map_.posToIndex(search_min)
        max_id = self.edt_env_.sdf_map_.posToIndex(search_max)
        for i in range(min_id.x, max_id.x + 1):
            for j in range(min_id.y, max_id.y + 1):
                for k in range(min_id.z, max_id.z + 1):
                    cur = Vector3i(i, j, k)
                    if self.frontier_flag_[self.toadr(cur)] == str(0) and self.knownfree(cur) and self.isNeighborUnknown(cur):
                        self.expandFrontier(cur)
        
        self.tmp_frontiers_ = self.splitLargeFrontiers(self.tmp_frontiers_)
    
    def expandFrontier(self, first: Vector3i):
        pos = self.edt_env_.sdf_map_.indexToPos(first)
        expand = []
        expand.append(pos)
        cell_queue = []
        cell_queue.append(first)
        self.frontier_flag_[self.toadr(first)] = str(1)

        # Search frontier cluster based on region growing (distance clustering)
        while len(cell_queue) > 0:
            cur = cell_queue[0]
            cell_queue.pop(0)
            nbrs = self.allNeighbors(cur)
            for nbr in nbrs:
                # Qualified cell should be inside bounding box and frontier cell not clustered
                adr = self.toadr(nbr)
                if (self.frontier_flag_[adr] == str(1) or 
                    not self.edt_env_.sdf_map_.isInBox(nbr) or
                    not (self.knownfree(nbr) and self.isNeighborUnknown(nbr))):
                    continue

                pos = self.edt_env_.sdf_map_.indexToPos(nbr)
                if pos.z < 0.4:  # Remove noise close to ground
                    continue
                expand.append(pos)
                cell_queue.append(nbr)
                self.frontier_flag_[adr] = str(1)

        if len(expand) > self.cluster_min_:
            # Compute detailed info
            frontier = Frontier()
            frontier.cells_ = expand
            frontier = self.computeFrontierInfo(frontier)
            self.tmp_frontiers_.append(frontier)

    def splitLargeFrontiers(self, frontiers: List[Frontier]):
        splits, tmps = [], []
        for frontier in frontiers:
            # Check if frontier needs to be split horizontally
            split_result, _, splits = self.splitHorizontally(frontier, splits)
            if split_result:
                tmps.extend(splits)
                splits.clear()
            else:
                tmps.append(frontier)
        return tmps
            

    def splitHorizontally(self, frontier: Frontier, splits: List[Frontier]):

        mean = frontier.average_.head2()
        need_split = False
        for cell in frontier.filtered_cells_:
            if (cell.head2() - mean).norm() > self.cluster_size_xy_:
                need_split = True
                break
        if not need_split:
            return False, frontier, splits
        
        cov = np.zeros((2, 2))
        for cell in frontier.filtered_cells_:
            diff = cell.head2() - mean
            cov[0][0] += diff.x * diff.x
            cov[0][1] += diff.x * diff.y
            cov[1][0] += diff.y * diff.x
            cov[1][1] += diff.y * diff.y
        cov /= len(frontier.filtered_cells_)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Find index of largest eigenvalue
        max_idx = np.argmax(eigenvalues)
        max_eigenvalue = eigenvalues[max_idx]
        
        # Get corresponding eigenvector (first principal component)
        first_pc = eigenvectors[:, max_idx]
        
        ftr1, ftr2 = Frontier(), Frontier()
        for cell in frontier.cells_:
            diff = cell.head2() - mean
            diff = np.array([diff.x, diff.y])
            if np.dot(diff, first_pc) > 0:
                ftr1.cells_.append(cell)
            else:
                ftr2.cells_.append(cell)
        ftr1 = self.computeFrontierInfo(ftr1)
        ftr2 = self.computeFrontierInfo(ftr2)
        splits2 = []
        if self.splitHorizontally(ftr1, splits2)[0]:
            splits.extend(splits2)
            splits2.clear()
        else:
            splits.append(ftr1)

        if self.splitHorizontally(ftr2, splits2)[0]:
            splits.extend(splits2)
        else:
            splits.append(ftr2)
        
        return True, frontier, splits
                
    def isInBoxes(self, boxes: List[Tuple[Vector3d, Vector3d]], idx: Vector3i):
        
        pt = self.edt_env_.sdf_map_.indexToPos(idx)
        for box in boxes:
            inbox = True
            for i in range(3):
                inbox = inbox and pt[i] > box[0][i] and pt[i] < box[1][i]
                if not inbox:
                    break
            if inbox:
                return True
        return False

    def updateFrontierCostMatrix(self):
        if len(self.removed_ids_) > 0:
            # Delete path and cost for removed clusters
            for ftr in self.frontiers_[:self.frontiers_.index(self.first_new_ftr_)]:
                cost_iter = 0
                path_iter = 0
                for removed_id in self.removed_ids_:
                    # Step iterator to the item to be removed
                    while cost_iter < removed_id:
                        cost_iter += 1
                        path_iter += 1
                   
                    ftr.costs_.pop(cost_iter)
                    ftr.paths_.pop(path_iter)
            
            self.removed_ids_.clear()
        
        def updateCost(ftr1: Frontier, ftr2: Frontier):
            # Search path from old cluster's top viewpoint to new cluster
            vu1 = ftr1.viewpoints_[0]
            vu2 = ftr2.viewpoints_[0]

            # Compute cost between viewpoints
            path = []
            cost = ViewNode().computeCost(
                vu1.pos_, vu2.pos_,
                vu1.yaw_, vu2.yaw_,
                Vector3d(0, 0, 0), 0, path)

            # Insert costs and paths for both frontiers
            ftr1.costs_.append(cost)
            ftr1.paths_.append(path)

            # Reverse path for second frontier
            path.reverse()
            ftr2.costs_.append(cost) 
            ftr2.paths_.append(path)

            return ftr1, ftr2
        
        # Update costs between old and new frontiers
        for i, old_ftr in enumerate(self.frontiers_[:self.frontiers_.index(self.first_new_ftr_)]):
            for j, new_ftr in enumerate(self.frontiers_[self.frontiers_.index(self.first_new_ftr_):]):
                old_ftr, new_ftr = updateCost(old_ftr, new_ftr)
                self.frontiers_[i] = old_ftr
                self.frontiers_[j] = new_ftr

    # Update costs between new frontiers
        for i, ftr1 in enumerate(self.frontiers_[self.frontiers_.index(self.first_new_ftr_):]):
            for j, ftr2 in enumerate(self.frontiers_[self.frontiers_.index(self.first_new_ftr_) + i:]):
                if ftr1 == ftr2:
                    self.frontiers_[self.frontiers_.index(self.first_new_ftr_) + i].costs_.append(0)
                    self.frontiers_[self.frontiers_.index(self.first_new_ftr_) + i].paths_.append([])

                else:
                    ftr1, ftr2 = updateCost(ftr1, ftr2)
                    self.frontiers_[self.frontiers_.index(self.first_new_ftr_) + i] = ftr1
                    self.frontiers_[self.frontiers_.index(self.first_new_ftr_) + i + j] = ftr2

    def mergeFrontiers(self, ftr1: Frontier, ftr2: Frontier):
        total_cells = len(ftr1.cells_) + len(ftr2.cells_)
        ftr1.average_ = Vector3d(
            (ftr1.average_.x * len(ftr1.cells_) + ftr2.average_.x * len(ftr2.cells_)) / total_cells,
            (ftr1.average_.y * len(ftr1.cells_) + ftr2.average_.y * len(ftr2.cells_)) / total_cells,
            (ftr1.average_.z * len(ftr1.cells_) + ftr2.average_.z * len(ftr2.cells_)) / total_cells
        )
        ftr1.cells_.extend(ftr2.cells_)
        ftr1 = self.computeFrontierInfo(ftr1)
        return ftr1

    def canBeMerged(self, ftr1: Frontier, ftr2: Frontier):
        total_cells = len(ftr1.cells_) + len(ftr2.cells_)
        merged_avg = Vector3d(
            (ftr1.average_.x * len(ftr1.cells_) + ftr2.average_.x * len(ftr2.cells_)) / total_cells,
            (ftr1.average_.y * len(ftr1.cells_) + ftr2.average_.y * len(ftr2.cells_)) / total_cells,
            (ftr1.average_.z * len(ftr1.cells_) + ftr2.average_.z * len(ftr2.cells_)) / total_cells
        )
        for c1 in ftr1.cells_:
            diff = c1 - merged_avg
            if diff.head2().norm() > self.cluster_size_xy_ or diff.z > self.cluster_size_z_:
                return False
        for c2 in ftr2.cells_:
            diff = c2 - merged_avg
            if diff.head2().norm() > self.cluster_size_xy_ or diff.z > self.cluster_size_z_:
                return False
        return True

    def computeFrontiersToVisit(self):
        self.first_new_ftr_ = self.frontiers_[-1]
        new_num, new_dormant_num = 0, 0
        for tmp_ftr in self.tmp_frontiers_:
            tmp_ftr = self.sampleViewpoints(tmp_ftr)
            if len(tmp_ftr.viewpoints_) > 0:
                new_num += 1
                # Sort viewpoints by visibility number (descending)
                tmp_ftr.viewpoints_.sort(key=lambda x: x.visib_num_, reverse=True)
                self.frontiers_.append(tmp_ftr)
                if self.first_new_ftr_ == self.frontiers_[-1]:
                    self.first_new_ftr_ = tmp_ftr
            else:
                # No viewpoints found, add to dormant frontiers
                self.dormant_frontiers_.append(tmp_ftr)
                new_dormant_num += 1
        idx = 0
        for i, ft in enumerate(self.frontiers_):
            self.frontiers_[i].id_ = idx
            idx += 1

    def getTopViewpointsInfo(self, cur_pos: Vector3d, points: List[Vector3d], yaws: List[float], averages: List[Vector3d]):
        
        points.clear()
        yaws.clear()
        averages.clear()
        for frontier in self.frontiers_:
            no_view = True
            for view in frontier.viewpoints_:
                if (view.pos_ - cur_pos).norm() < self.min_candidate_dist_:
                    continue
                points.append(view.pos_)
                yaws.append(view.yaw_)
                averages.append(frontier.average_)
                no_view = False
                break
            if no_view:
                view = frontier.viewpoints_[0]
                points.append(view.pos_)
                yaws.append(view.yaw_)
                averages.append(frontier.average_)
        
        return points, yaws, averages

    def getViewpointsInfo(self, cur_pos: Vector3d, ids: List[int], view_num: int, max_dacay: float, points: List[List[Vector3d]], yaws: List[List[float]]):
        points.clear()
        yaws.clear()
        for id in ids:
            for frontier in self.frontiers_:
                if frontier.id_ == id:
                    pts, ys = [], []
                    visib_thresh = frontier.viewpoints_[0].visib_num_ * max_dacay
                    for view in frontier.viewpoints_:
                        if len(pts) >= view_num or view.visib_num_ < visib_thresh:
                            break
                        if (view.pos_ - cur_pos).norm() < self.min_candidate_dist_:
                            continue
                        pts.append(view.pos_)
                        ys.append(view.yaw_)
                    if len(pts) == 0:
                        for view in frontier.viewpoints_:
                            if len(pts) >= view_num or view.visib_num_ < visib_thresh:
                                break
                            pts.append(view.pos_)
                            ys.append(view.yaw_)
                    points.append(pts)
                    yaws.append(ys)
        return points, yaws

    def getFrontiers(self, clusters: List[List[Vector3d]]):
        clusters.clear()
        for frontier in self.frontiers_:
            clusters.append(frontier.cells_)
        return clusters

    def getDormantFrontiers(self, clusters: List[List[Vector3d]]):
        clusters.clear()
        for frontier in self.dormant_frontiers_:
            clusters.append(frontier.cells_)
        return clusters

    def getFrontierBoxes(self, boxes: List[Tuple[Vector3d, Vector3d]]):
        boxes.clear()
        for frontier in self.frontiers_:
            center = (frontier.box_min_ + frontier.box_max_) / 2
            scale = frontier.box_max_ - frontier.box_min_
            boxes.append((center, scale))
        return boxes

    def getPathForTour(self, pos: Vector3d, frontier_ids: List[int], path: List[Vector3d]):
        # Make a frontier_indexer to access the frontier list easier
        frontier_indexer = []
        for frontier in self.frontiers_:
            frontier_indexer.append(frontier)

        # Compute the path from current pos to the first frontier
        segment : List[Vector3d] = []
        _, segment = ViewNode().searchPath(pos, frontier_indexer[frontier_ids[0]].viewpoints_[0].pos_, segment)
        path.extend(segment)

        # Get paths of tour passing all clusters
        for i in range(len(frontier_ids) - 1):
            # Move path to next cluster
            path_iter = frontier_indexer[frontier_ids[i]].paths_[frontier_ids[i + 1]]
            path.extend(path_iter)
        
        return path

    def getFullCostMatrix(self, cur_pos: Vector3d, cur_vel: Vector3d, cur_yaw: Vector3d, mat: np.ndarray):
        dimen = len(self.frontiers_)
        mat.resize((dimen + 1, dimen + 1))
        # Fill block for clusters
        i = 1
        for frontier in self.frontiers_:
            j = 1
            for cost in frontier.costs_:
                mat[i,j] = cost
                j += 1
            i += 1
        # Fill block from current state to clusters
        mat[:,0] = 0
        j = 1
        for frontier in self.frontiers_:
            vj = frontier.viewpoints_[0]
            path = []
            mat[0,j] = ViewNode().computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path)
            j += 1

        return mat

    def sampleViewpoints(self, frontier: Frontier):
        rc = self.candidate_rmin_
        dr = (self.candidate_rmax_ - self.candidate_rmin_) / self.candidate_rnum_
        
        while rc <= self.candidate_rmax_ + 1e-3:
            phi = -math.pi
            while phi < math.pi:
                sample_pos = frontier.average_ + rc * Vector3d(math.cos(phi), math.sin(phi), 0)

                # Check if viewpoint is qualified (in box and safe region)
                if (not self.edt_env_.sdf_map_.isInBox(sample_pos) or
                    self.edt_env_.sdf_map_.getInflatedOccupancy(sample_pos) == 1 or 
                    self.isNearUnknown(sample_pos)):
                    phi += self.candidate_dphi_
                    continue

                # Compute average yaw
                cells = frontier.filtered_cells_
                ref_dir = (cells[0] - sample_pos).normalized()
                avg_yaw = 0.0
                
                for i in range(1, len(cells)):
                    dir = (cells[i] - sample_pos).normalized()
                    yaw = math.acos(ref_dir.x * dir.x + ref_dir.y * dir.y + ref_dir.z * dir.z)
                    if ref_dir.x * dir.y - ref_dir.y * dir.x < 0:
                        yaw = -yaw
                    avg_yaw += yaw

                avg_yaw = avg_yaw / len(cells) + math.atan2(ref_dir.y, ref_dir.x)
                self.wrapYaw(avg_yaw)

                # Count visible cells and add viewpoint if above threshold
                visib_num = self.countVisibleCells(sample_pos, avg_yaw, cells)
                if visib_num > self.min_visib_num_:
                    vp = Viewpoint(sample_pos, avg_yaw, visib_num)
                    frontier.viewpoints_.append(vp)

                phi += self.candidate_dphi_
            rc += dr
        return frontier

    def isFrontierCovered(self):
        update_min, update_max = self.edt_env_.sdf_map_.getUpdatedBox(False)
        def checkChanges(frontiers: List[Frontier]) -> bool:
            for ftr in frontiers:
                if not self.haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max):
                    continue
                
                change_thresh = int(self.min_view_finish_fraction_ * len(ftr.cells_))
                change_num = 0
                
                for cell in ftr.cells_:
                    idx = self.edt_env_.sdf_map_.posToIndex(cell)
                    if not (self.knownfree(idx) and self.isNeighborUnknown(idx)):
                        change_num += 1
                        if change_num >= change_thresh:
                            return True
                            
            return False
        if checkChanges(self.frontiers_) or checkChanges(self.dormant_frontiers_):
            return True
        return False

    def isNearUnknown(self, pos: Vector3d):
        vox_num = math.floor(self.min_candidate_clearance_ / self.resolution_)
        for x in range(-vox_num, vox_num + 1):
            for y in range(-vox_num, vox_num + 1):
                for z in range(-1, 2):
                    vox = Vector3d(
                        pos.x + x * self.resolution_,
                        pos.y + y * self.resolution_,
                        pos.z + z * self.resolution_
                    )
                    if self.edt_env_.sdf_map_.getOccupancy(vox) == SDFmap.UNKNOWN:
                        return True
        return False

    def countVisibleCells(self, pos: Vector3d, yaw: float, cluster: List[Vector3d]):
        self.percep_utils_.setPose(pos, yaw)
        visib_num = 0
        idx = Vector3i()
        for cell in cluster:
            if not self.percep_utils_.insideFOV(cell):
                continue
            self.raycaster_.input(cell, pos)
            visib = True
            while self.raycaster_.nextId(idx):
                if self.edt_env_.sdf_map_.getInflatedOccupancy(idx) == 1 or self.edt_env_.sdf_map_.getOccupancy(idx) == SDFmap.UNKNOWN:
                    visib = False
                    break
            if visib:
                visib_num += 1
        return visib_num

    def wrapYaw(self, yaw: float):
        while yaw < -math.pi:
            yaw += 2 * math.pi
        while yaw > math.pi:
            yaw -= 2 * math.pi
        return yaw

    def toadr(self, idx: Vector3i):
        return self.edt_env_.sdf_map_.toAddress(idx.x, idx.y, idx.z)

    def haveOverlap(self, min1: Vector3d, max1: Vector3d, min2: Vector3d, max2: Vector3d):
        bmin, bmax = Vector3d(), Vector3d()
        for i in range(3):
            bmin[i] = max(min1[i], min2[i])
            bmax[i] = min(max1[i], max2[i])
        return bmin.x <= bmax.x +1e-3 and bmin.y <= bmax.y +1e-3 and bmin.z <= bmax.z +1e-3
                
    def isFrontierChanged(self, iter: Frontier):
        for cell in iter.cells_:
            idx = self.edt_env_.sdf_map_.posToIndex(cell)
            if not self.knownfree(idx) and self.isNeighborUnknown(idx):
                return True
        return False

    def knownfree(self, idx: Vector3i):
        return self.edt_env_.sdf_map_.getOccupancy(idx) == SDFmap.FREE
    
    def inmap(self, idx: Vector3i):
        return self.edt_env_.sdf_map_.isInBox(idx)

    def isNeighborUnknown(self, voxel: Vector3i):
        nbrs = self.sixNeighbors(voxel)
        for nbr in nbrs:
            if self.edt_env_.sdf_map_.getOccupancy(nbr) == SDFmap.UNKNOWN:
                return True
        return False

    def sixNeighbors(self, voxel: Vector3i):
        neighbors = []
        neighbors.append(Vector3i(voxel.x - 1, voxel.y, voxel.z))
        neighbors.append(Vector3i(voxel.x + 1, voxel.y, voxel.z))
        neighbors.append(Vector3i(voxel.x, voxel.y - 1, voxel.z))
        neighbors.append(Vector3i(voxel.x, voxel.y + 1, voxel.z))
        neighbors.append(Vector3i(voxel.x, voxel.y, voxel.z - 1))
        neighbors.append(Vector3i(voxel.x, voxel.y, voxel.z + 1))
        return neighbors
    
    def tenNeighbors(self, voxel: Vector3i):
        neighbors = [Vector3i() for _ in range(10)]
        count = 0
        for x in range(-1,2):
            for y in range(-1,2):
                if x == 0 and y == 0:
                    continue
                neighbors[count] = voxel + Vector3i(x, y, 0)
                count += 1
        return neighbors
    
    def allNeighbors(self, voxel: Vector3i):

        neighbors = [Vector3i() for _ in range(26)]
        count = 0
        for x in range(-1,2):
            for y in range(-1,2):
                for z in range(-1,2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    neighbors[count] = voxel + Vector3i(x, y, z)
                    count += 1
        return neighbors
        
    def computeFrontierInfo(self, ftr: Frontier):
        ftr.average_.setZero()
        ftr.box_max_ = ftr.cells_[0]
        ftr.box_min_ = ftr.cells_[0]
        for cell in ftr.cells_:
            ftr.average_ += cell
            for i in range(3):
                ftr.box_max_[i] = max(ftr.box_max_[i], cell[i])
                ftr.box_min_[i] = min(ftr.box_min_[i], cell[i])
        ftr.average_ /= len(ftr.cells_)
        ftr.filtered_cells_ = self.downsample(ftr.cells_, ftr.filtered_cells_)

        return ftr
        
    def downsample(self, cluster_in: List[Vector3d], cluster_out: List[Vector3d]):
        # Downsample points using voxel grid
        leaf_size = self.edt_env_.sdf_map_.getResolution() * self.down_sample_        
        # Create voxel grid
        voxel_map = {}
        
        # Add points to voxel grid
        for cell in cluster_in:
            # Get voxel indices
            vx = int(cell.x / leaf_size)
            vy = int(cell.y / leaf_size) 
            vz = int(cell.z / leaf_size)
            voxel_key = (vx, vy, vz)
            
            # Add point to voxel
            if voxel_key not in voxel_map:
                voxel_map[voxel_key] = []
            voxel_map[voxel_key].append(cell)
            
        # Clear output cluster
        cluster_out.clear()
        
        # Compute centroids for each voxel
        for points in voxel_map.values():
            centroid = Vector3d()
            for p in points:
                centroid += p
            centroid /= len(points)
            cluster_out.append(centroid)
        
        return cluster_out

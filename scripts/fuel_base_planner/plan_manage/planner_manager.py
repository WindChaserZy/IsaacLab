from ..utils import Vector3d
from .plan_container import PlanParameters, LocalTrajData, GlobalTrajData, MidPlanData
from ..env import EDTEnv, SDFmap
from ..searching import AStar, KinodynamicAstar, TopologyPRM
from ..bspline_opt import BsplineOptimizer
from ..perception import FrontierFinder, HeadingPlanner, VisibilityUtil
from ..bspline import NonUniformBspline
from ..poly_traj import PolynomialTraj
from typing import List
import time
from time import perf_counter
import numpy as np
import math
import threading

class FastPlannerManager:

    def __init__(self):

        self.pp_ = PlanParameters()
        self.local_data_ = LocalTrajData()
        self.global_data_ = GlobalTrajData()
        self.plan_data_ = MidPlanData()
        ######## config #################
        self.pp_.max_vel_ = -1.0
        self.pp_.max_acc_ = -1.0
        self.pp_.max_jerk_ = -1.0
        self.pp_.accept_vel_ = self.pp_.max_vel_ + 0.5
        self.pp_.accept_acc_ = self.pp_.max_acc_ + 0.5
        self.pp_.max_yawdot_ = -1.0
        self.pp_.dynamic_ = -1
        self.pp_.clearance_ = -1.0
        self.pp_.local_traj_len_ = -1.0
        self.pp_.ctrl_pt_dist = -1.0
        self.pp_.bspline_degree_ = 3
        self.pp_.min_time_ = False

        use_geometric_path = True
        use_kinodynamic_path = True
        use_topo_path = True
        use_optimization = True
        use_active_perception = True
        #################################
        self.local_data_.traj_id_ = 0
        self.sdf_map_ = SDFmap()
        self.edt_environment_ = EDTEnv()
        self.edt_environment_.setMap(self.sdf_map_)
        if use_geometric_path:
            self.path_finder_ = AStar(self.edt_environment_)
        if use_kinodynamic_path:
            self.kino_path_finder_ = KinodynamicAstar()
            self.kino_path_finder_.setEnvironment(self.edt_environment_)
        if use_topo_path:
            self.topo_prm_ = TopologyPRM()
            self.topo_prm_.setEnviroment(self.edt_environment_)
        if use_optimization:
            self.bspline_optimizers_ : List[BsplineOptimizer] = [BsplineOptimizer() for _ in range(10)]
            for i in range(10):
                self.bspline_optimizers_[i].setEnvironment(self.edt_environment_)
        if use_active_perception:
            self.frontier_finder_ = FrontierFinder()
            self.frontier_finder_.edt_env_ = self.edt_environment_
            self.heading_planner_ = HeadingPlanner()
            self.heading_planner_.setMap(self.sdf_map_)
            self.visibility_util_ = VisibilityUtil()
            self.visibility_util_.setEDTEnvironment(self.edt_environment_)
            self.plan_data_.view_cons_.idx_ = -1
    
    def setGlobalWaypoints(self, waypoints: List[Vector3d]):
        self.plan_data_.global_waypoints_ = waypoints

    def checkTrajCollision(self, distance: float):
        
        t_now = time.perf_counter() - self.local_data_.start_time_
        cur_pt = self.local_data_.position_traj_.evaluateDeBoorT(t_now)
        radius = 0.0
        fut_pt = Vector3d()
        fut_t = 0.02

        while radius < 6.0 and t_now + fut_t < self.local_data_.duration_:
            fut_pt = self.local_data_.position_traj_.evaluateDeBoorT(t_now + fut_t)
            if self.sdf_map_.getInflatedOccupancy(fut_pt) == 1:
                distance = radius
                print(f"collision at: {fut_pt}")
                return False, distance
            radius = (fut_pt - cur_pt).norm()
            fut_t += 0.02

        return True, distance
    
    def kinodynamicReplan(self, start_pt: Vector3d, start_vel: Vector3d, start_acc: Vector3d, end_pt: Vector3d, end_vel: Vector3d, time_lb: float):

        if (start_pt - end_pt).norm() < 1e-2:
            print("Close goal")
            return False
        
        init_pos = start_pt
        init_vel = start_vel
        init_acc = start_acc

        t1 = time.perf_counter()
        
        self.kino_path_finder_.reset()
        
        status = self.kino_path_finder_.search(start_pt, start_vel, start_acc, end_pt, end_vel, True)
        
        if status == KinodynamicAstar.NO_PATH:
            print("search 1 fail")
            # Retry
            self.kino_path_finder_.reset()
            status = self.kino_path_finder_.search(start_pt, start_vel, start_acc, end_pt, end_vel, False)
            if status == KinodynamicAstar.NO_PATH:
                print("[Kino replan]: Can't find path.")
                return False

        self.plan_data_.kino_path_ = self.kino_path_finder_.getKinoTraj(0.01)

        t_search = time.perf_counter() - t1
        t1 = time.perf_counter()

        # Parameterize path to B-spline
        ts = self.pp_.ctrl_pt_dist / self.pp_.max_vel_
        point_set = []
        start_end_derivatives = []
        ts, point_set, start_end_derivatives = self.kino_path_finder_.getSamples(ts, point_set, start_end_derivatives)
        ctrl_pts = np.zeros((len(point_set), 3))
        ctrl_pts = NonUniformBspline.parameterizeToBspline(
            ts, point_set, start_end_derivatives, self.pp_.bspline_degree_, ctrl_pts)
        init = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, ts)

        # B-spline-based optimization
        cost_function = BsplineOptimizer.NORMAL_PHASE
        if self.pp_.min_time_:
            cost_function |= BsplineOptimizer.MINTIME
        start, end = init.getBoundaryStates(2, 0, [], [])
        self.bspline_optimizers_[0].setBoundaryState(start, end)
        if time_lb > 0:
            self.bspline_optimizers_[0].setTimeLowerBound(time_lb)

        self.bspline_optimizers_[0].optimize(ctrl_pts, ts, cost_function, 1, 1)
        self.local_data_.position_traj_.setUniformBspline(ctrl_pts, self.pp_.bspline_degree_, ts)

        start2, end2 = self.local_data_.position_traj_.getBoundaryStates(2, 0, [], [])
        print(f"State error: ({(start2[0] - start[0]).norm()}, {(start2[1] - start[1]).norm()}, {(start2[2] - start[2]).norm()})")
        
        self.updateTrajInfo()
        return True
    
    def updateTrajInfo(self):

        self.local_data_.velocity_traj_ = self.local_data_.position_traj_.getDerivative()
        self.local_data_.acceleration_traj_ = self.local_data_.velocity_traj_.getDerivative()
        
        self.local_data_.start_pos_ = self.local_data_.position_traj_.evaluateDeBoorT(0.0)
        self.local_data_.duration_ = self.local_data_.position_traj_.getTimeSum()

        self.local_data_.traj_id_ += 1

    def planExploreTraj(self, tour: List[Vector3d], cur_vel: Vector3d, cur_acc: Vector3d, time_lb: float):

        if len(tour) == 0:
            print("Empty path to traj planner")
            return False
        
        pt_num = len(tour)
        pos = np.zeros((pt_num, 3))
        for i in range(pt_num):
            pos[i] = np.array(tour[i].listify())
        
        zero = Vector3d()
        times = np.zeros(pt_num - 1)
        for i in range(pt_num - 1):
            times[i] = (pos[i + 1] - pos[i]).norm() / (self.pp_.max_vel_ * 0.5)
        
        init_traj = PolynomialTraj()
        init_traj = PolynomialTraj.waypointsTraj(pos, cur_vel, zero, cur_acc, zero, times, init_traj)

        points, boundary_deri = [], []
        duration = init_traj.getTotalTime()
        seg_num = init_traj.getLength() * self.pp_.ctrl_pt_dist
        seg_num = max(8, seg_num)
        dt = duration / seg_num

        ts = 0.0
        while ts <= duration + 1e-4:
            points.append(init_traj.evaluate(ts, 0))
            ts += dt
        boundary_deri.append(init_traj.evaluate(0.0, 1))
        boundary_deri.append(init_traj.evaluate(duration, 1))
        boundary_deri.append(init_traj.evaluate(0.0, 2))
        boundary_deri.append(init_traj.evaluate(duration, 2))

        ctrl_pts = np.array([])
        ctrl_pts = NonUniformBspline.parameterizeToBspline(
            ts, points, boundary_deri, self.pp_.bspline_degree_, ctrl_pts)
        tmp_traj = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, ts)
        cost_func = BsplineOptimizer.NORMAL_PHASE
        if self.pp_.min_time_:
            cost_func |= BsplineOptimizer.MINTIME
        start, end = tmp_traj.getBoundaryStates(2, 0, [], [])
        self.bspline_optimizers_[0].setBoundaryState(start, end)
        if time_lb > 0:
            self.bspline_optimizers_[0].setTimeLowerBound(time_lb)
        self.bspline_optimizers_[0].optimize(ctrl_pts, dt, cost_func, 1, 1)
        self.local_data_.position_traj_.setUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)

        self.updateTrajInfo()

    def planGlobalTraj(self, start_pos: Vector3d):

        self.plan_data_.clearTopoPaths()
        points = self.plan_data_.global_waypoints_
        if len(points) == 0:
            print("No global waypoints")
        points.insert(0, start_pos)

        inter_points = []
        dist_thresh =4.0
        for i in range(len(points) - 1):
            inter_points.append(points[i])
            dist = (points[i + 1] - points[i]).norm()
            if dist > dist_thresh:
                id_num = math.floor(dist / dist_thresh) + 1
                for j in range(1, id_num):
                    ratio = j / id_num
                    inter_pt = Vector3d(
                        points[i].x * (1.0 - ratio) + points[i + 1].x * ratio,
                        points[i].y * (1.0 - ratio) + points[i + 1].y * ratio,
                        points[i].z * (1.0 - ratio) + points[i + 1].z * ratio
                    )
                    inter_points.append(inter_pt)
        inter_points.append(points[-1])

        if len(inter_points) == 2:
            mid = (inter_points[0] + inter_points[1]) * 0.5
            inter_points.insert(1, mid)
        
        pt_num = len(inter_points)
        pos = np.zeros((pt_num, 3))
        for i in range(pt_num):
            pos[i] = np.array(inter_points[i].listify())
        zero = Vector3d()
        time = np.zeros(pt_num - 1)
        for i in range(pt_num - 1):
            time[i] = (pos[i + 1] - pos[i]).norm() / (self.pp_.max_vel_ * 0.5)
        time[0] += self.pp_.max_vel_ / (2 * self.pp_.max_acc_)
        time[-1] += self.pp_.max_vel_ / (2 * self.pp_.max_acc_)
        
        gl_traj = PolynomialTraj()
        gl_traj = PolynomialTraj.waypointsTraj(pos, zero, zero, zero, zero, time, gl_traj)

        self.global_data_.setGlobalTraj(gl_traj)
        ctrl_pts, dt, duration = self.paramLocalTraj(0.0)
        bspline = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)
        self.global_data_.setLocalTraj(bspline, 0.0, duration, 0.0)
        self.local_data_.position_traj_ = bspline
        self.local_data_.start_time_ = perf_counter()

        self.updateTrajInfo()
        return True
    
    def topoReplan(self, collide: bool):

        time_now = perf_counter()
        t_now = time_now - self.global_data_.global_start_time_
        ctrl_pts, local_traj_dt, local_traj_duration = self.paramLocalTraj(t_now)
        init_traj = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, local_traj_dt)
        self.local_data_.start_time_ = time_now

        if not collide:
            init_traj = self.refineTraj(init_traj)
            time_change = init_traj.getTimeSum() - local_traj_duration
            self.local_data_.position_traj_ = init_traj
            self.global_data_.setLocalTraj(self.local_data_.position_traj_, time_now, local_traj_duration + time_change + time_now, time_change)
        else:
            self.plan_data_.initial_local_segment_ = init_traj
            colli_start, colli_end, start_pts, end_pts = self.findCollisionRange()
            if len(colli_start) == 1 and len(colli_end) == 0:
                self.local_data_.position_traj_ = init_traj
                self.global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0)
            else:
                self.plan_data_.clearTopoPaths()
                graph, raw_paths, filtered_paths, select_paths = [], [], [], []
                graph, raw_paths, filtered_paths, select_paths = self.topo_prm_.findTopoPaths(colli_start[0], colli_end[-1], start_pts, end_pts, graph, raw_paths, filtered_paths, select_paths)
                if len(select_paths) == 0:
                    print("No topo path found")
                    return False
                self.plan_data_.addTopoPath(graph, raw_paths, filtered_paths, select_paths)
                t1 = perf_counter()
                if len(self.plan_data_.topo_traj_pos1_) >= len(select_paths):
                    self.plan_data_.topo_traj_pos1_ = self.plan_data_.topo_traj_pos1_[:len(select_paths)]
                else:
                    self.plan_data_.topo_traj_pos1_.extend([NonUniformBspline()] * (len(select_paths) - len(self.plan_data_.topo_traj_pos1_)))
                if len(self.plan_data_.topo_traj_pos2_) >= len(select_paths):
                    self.plan_data_.topo_traj_pos2_ = self.plan_data_.topo_traj_pos2_[:len(select_paths)]
                else:
                    self.plan_data_.topo_traj_pos2_.extend([NonUniformBspline()] * (len(select_paths) - len(self.plan_data_.topo_traj_pos2_)))
                # Optimize trajectories in parallel
                optimize_threads = []
                for i in range(len(select_paths)):
                    thread = threading.Thread(target=self.optimizeTopoBspline, args=(t_now, local_traj_duration, select_paths[i], i))
                    optimize_threads.append(thread)
                    thread.start()

                # Wait for all optimization threads to complete
                for thread in optimize_threads:
                    thread.join()

                t_opt = perf_counter() - t1
                print(f"[planner]: optimization time: {t_opt}")

                # Select and refine best trajectory
                best_traj = NonUniformBspline()
                best_traj = self.selectBestTraj(best_traj)
                best_traj = self.refineTraj(best_traj)
                time_change = best_traj.getTimeSum() - local_traj_duration

                # Update trajectory data
                self.local_data_.position_traj_ = best_traj
                self.global_data_.setLocalTraj(self.local_data_.position_traj_, t_now,
                                             local_traj_duration + time_change + t_now, time_change)

        self.updateTrajInfo()

    
    def paramLocalTraj(self, t: float):
        dt, duration = 0.0
        point_set = []
        start_end_derivative = []
        
        # Get trajectory info in sphere
        point_set, start_end_derivative = self.global_data_.getTrajInfoInSphere(
            t, 
            self.pp_.local_traj_len_,
            self.pp_.ctrl_pt_dist, point_set, start_end_derivative, dt, duration
        )

        # Parameterize to B-spline
        ctrl_pts = NonUniformBspline.parameterizeToBspline(
            dt,
            point_set, 
            start_end_derivative,
            self.pp_.bspline_degree_,
            np.zeros((0,0))
        )

        self.plan_data_.local_start_end_derivative_ = start_end_derivative

        return ctrl_pts, dt, duration
    
    def selectBestTraj(self, traj: NonUniformBspline):

        trajs = self.plan_data_.topo_traj_pos2_
        # Sort trajectories by jerk and select best one
        trajs.sort(key=lambda tj: tj.getJerk())
        return trajs[0]
    
    def refineTraj(self, traj: NonUniformBspline):

        t1 = perf_counter()
        self.plan_data_.no_visib_traj_ = traj

        cost_function = BsplineOptimizer.NORMAL_PHASE
        if self.pp_.min_time_:
            cost_function |= BsplineOptimizer.MINTIME

        # Get control points and timing info
        ctrl_pts = traj.getControlPoint()
        dt = traj.getKnotSpan()
        
        # Get boundary states
        start1, end1 = [], []
        traj.getBoundaryStates(2, 0, start1, end1)

        # Optimize trajectory
        self.bspline_optimizers_[0].setBoundaryState(start1, end1)
        self.bspline_optimizers_[0].optimize(ctrl_pts, dt, cost_function, 2, 2)
        traj.setUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)

        # Get final boundary states
        start2, end2 = [], []
        traj.getBoundaryStates(2, 2, start2, end2)

        return traj
    
    def findCollisionRange(self):

        colli_start = []
        colli_end = []
        start_pts = []
        end_pts = []

        last_safe = True
        safe = True
        initial_traj = self.plan_data_.initial_local_segment_
        t_m, t_mp = initial_traj.getTimeSpan()

        # Find range of collision
        t_s = -1.0
        tc = t_m
        while tc <= t_mp + 1e-4:
            ptc = initial_traj.evaluateDeBoor(tc)
            safe = False if self.edt_environment_.evaluateCoarseEDT(ptc, -1.0) < self.topo_prm_.clearance_ else True

            if last_safe and not safe:
                colli_start.append(initial_traj.evaluateDeBoor(tc - 0.05))
                if t_s < 0.0:
                    t_s = tc - 0.05
            elif not last_safe and safe:
                colli_end.append(ptc)
                t_e = tc

            last_safe = safe
            tc += 0.05

        if len(colli_start) == 0:
            return

        if len(colli_start) == 1 and len(colli_end) == 0:
            return

        # Find start and end safe segment
        dt = initial_traj.getKnotSpan()
        sn = math.ceil((t_s - t_m) / dt)
        dt = (t_s - t_m) / sn

        tc = t_m
        while tc <= t_s + 1e-4:
            start_pts.append(initial_traj.evaluateDeBoor(tc))
            tc += dt

        dt = initial_traj.getKnotSpan()
        sn = math.ceil((t_mp - t_e) / dt)
        dt = (t_mp - t_e) / sn

        if dt > 1e-4:
            tc = t_e
            while tc <= t_mp + 1e-4:
                end_pts.append(initial_traj.evaluateDeBoor(tc))
                tc += dt
        else:
            end_pts.append(initial_traj.evaluateDeBoor(t_mp))

        return colli_start, colli_end, start_pts, end_pts
                    
    def optimizeTopoBspline(self, start_t: float, duration: float, guide_path: List[Vector3d], traj_id: int):

        t1 = perf_counter()

        # Re-parameterize B-spline according to the length of guide path
        seg_num = math.floor(self.topo_prm_.pathLength(guide_path) / self.pp_.ctrl_pt_dist)
        seg_num = max(6, seg_num)  # Min number required for optimizing
        dt = duration / float(seg_num)
        ctrl_pts = self.reparamLocalTraj(start_t, duration, dt)

        tmp_traj = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)
        start = []
        end = []
        start, end = tmp_traj.getBoundaryStates(2, 0, start, end)

        # Discretize the guide path and align it with B-spline control points
        tmp_pts = []
        guide_pts = []
        if self.pp_.bspline_degree_ == 3 or self.pp_.bspline_degree_ == 5:
            tmp_pts = self.topo_prm_.pathToGuidePts(guide_path, int(ctrl_pts.shape[0]) - 2, tmp_pts)
            guide_pts.extend(tmp_pts[2:-2])
            if len(guide_pts) != int(ctrl_pts.shape[0]) - 6:
                print("Warning: Incorrect guide for 3 degree")
        elif self.pp_.bspline_degree_ == 4:
            tmp_pts = self.topo_prm_.pathToGuidePts(guide_path, int(2 * ctrl_pts.shape[0]) - 7, tmp_pts)
            for i in range(len(tmp_pts)):
                if i % 2 == 1 and i >= 5 and i <= len(tmp_pts) - 6:
                    guide_pts.append(tmp_pts[i])
            if len(guide_pts) != int(ctrl_pts.shape[0]) - 8:
                print("Warning: Incorrect guide for 4 degree")

        tm1 = perf_counter() - t1
        t1 = perf_counter()

        # First phase, path-guided optimization
        start, end = self.bspline_optimizers_[traj_id].setBoundaryState(start, end)
        self.bspline_optimizers_[traj_id].setGuidePath(guide_pts)
        self.bspline_optimizers_[traj_id].optimize(ctrl_pts, dt, BsplineOptimizer.GUIDE_PHASE, 0, 1)
        self.plan_data_.topo_traj_pos1_[traj_id] = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)

        tm2 = perf_counter() - t1
        t1 = perf_counter()

        # Second phase, smooth+safety+feasibility
        cost_func = BsplineOptimizer.NORMAL_PHASE
        start, end = self.bspline_optimizers_[traj_id].setBoundaryState(start, end)
        self.bspline_optimizers_[traj_id].optimize(ctrl_pts, dt, cost_func, 1, 1)
        self.plan_data_.topo_traj_pos2_[traj_id] = NonUniformBspline(ctrl_pts, self.pp_.bspline_degree_, dt)
        
        
    def reparamLocalTraj(self, start_t: float, duration: float, dt: float) -> np.ndarray:

        point_set = []
        start_end_derivative = []

        self.global_data_.getTrajInfoInDuration(start_t, duration, dt, point_set, start_end_derivative)
        self.plan_data_.local_start_end_derivative_ = start_end_derivative

        # Parameterize to B-spline
        ctrl_pts = np.zeros((len(point_set), 3))
        ctrl_pts = NonUniformBspline.parameterizeToBspline(dt, point_set, start_end_derivative, self.pp_.bspline_degree_, ctrl_pts)

        return ctrl_pts

    def planYaw(self, start_yaw: Vector3d):

        t1 = perf_counter()

        # Calculate waypoints of heading
        pos = self.local_data_.position_traj_
        duration = pos.getTimeSum()

        dt_yaw = 0.3
        seg_num = math.ceil(duration / dt_yaw)
        dt_yaw = duration / seg_num

        forward_t = 2.0
        last_yaw = start_yaw[0]
        waypts = []
        waypt_idx = []

        # seg_num -> seg_num - 1 points for constraint excluding the boundary states
        for i in range(seg_num):
            tc = i * dt_yaw
            pc = pos.evaluateDeBoorT(tc)
            tf = min(duration, tc + forward_t)
            pf = pos.evaluateDeBoorT(tf)
            pd = pf - pc

            if np.linalg.norm(pd) > 1e-6:
                waypt = np.array([np.arctan2(pd[1], pd[0]), 0.0, 0.0])
                waypt[0] = self.calcNextYaw(last_yaw, waypt[0])
            else:
                waypt = waypts[-1]
            
            last_yaw = waypt[0]
            waypts.append(waypt)
            waypt_idx.append(i)

        # Calculate initial control points with boundary state constraints
        yaw = np.zeros((seg_num + 3, 1))

        states2pts = np.array([
            [1.0, -dt_yaw, (1/3.0) * dt_yaw * dt_yaw],
            [1.0, 0.0, -(1/6.0) * dt_yaw * dt_yaw],
            [1.0, dt_yaw, (1/3.0) * dt_yaw * dt_yaw]
        ])
        yaw[0:3, 0:1] = states2pts @ start_yaw

        end_v = self.local_data_.velocity_traj_.evaluateDeBoorT(duration - 0.1)
        end_yaw = np.array([np.arctan2(end_v[1], end_v[0]), 0, 0])
        end_yaw[0] = self.calcNextYaw(last_yaw, end_yaw[0])
        yaw[seg_num:seg_num+3, 0:1] = states2pts @ end_yaw

        # Solve
        self.bspline_optimizers_[1].setWaypoints(waypts, waypt_idx)
        cost_func = BsplineOptimizer.SMOOTHNESS | BsplineOptimizer.WAYPOINTS | \
                    BsplineOptimizer.START | BsplineOptimizer.END

        start = [Vector3d(start_yaw[0], 0, 0), 
                Vector3d(start_yaw[1], 0, 0), 
                Vector3d(start_yaw[2], 0, 0)]
        end = [Vector3d(end_yaw[0], 0, 0),
               Vector3d(end_yaw[1], 0, 0), 
               Vector3d(end_yaw[2], 0, 0)]
        self.bspline_optimizers_[1].setBoundaryState(start, end)
        yaw, dt_yaw = self.bspline_optimizers_[1].optimize(yaw, dt_yaw, cost_func, 1, 1)

        # Update traj info
        self.local_data_.yaw_traj_.setUniformBspline(yaw, self.pp_.bspline_degree_, dt_yaw)
        self.local_data_.yawdot_traj_ = self.local_data_.yaw_traj_.getDerivative()
        self.local_data_.yawdotdot_traj_ = self.local_data_.yawdot_traj_.getDerivative()

        path_yaw = [waypts[i][0] for i in range(len(waypts))]
        self.plan_data_.path_yaw_ = path_yaw
        self.plan_data_.dt_yaw_ = dt_yaw
        self.plan_data_.dt_yaw_path_ = dt_yaw

    def planYawExplore(self, start_yaw: Vector3d, end_yaw: float, lookfwd: bool, relax_time: float):

        seg_num = 12
        dt_yaw = self.local_data_.duration_ / seg_num
        start_yaw3d = start_yaw.copy()
        
        # Normalize start yaw angle to [-pi, pi]
        while start_yaw3d[0] < -np.pi:
            start_yaw3d[0] += 2 * np.pi
        while start_yaw3d[0] > np.pi:
            start_yaw3d[0] -= 2 * np.pi
        last_yaw = start_yaw3d[0]

        # Initialize yaw trajectory control points
        yaw = np.zeros((seg_num + 3, 1))

        # Set initial state constraints
        states2pts = np.array([
            [1.0, -dt_yaw, (1/3.0) * dt_yaw * dt_yaw],
            [1.0, 0.0, -(1/6.0) * dt_yaw * dt_yaw],
            [1.0, dt_yaw, (1/3.0) * dt_yaw * dt_yaw]
        ])
        yaw[0:3, 0:1] = states2pts @ np.array(start_yaw3d.listify())

        # Add waypoint constraints for look-forward mode
        waypts = []
        waypt_idx = []
        if lookfwd:
            forward_t = 2.0
            relax_num = int(relax_time / dt_yaw)
            for i in range(1, seg_num - relax_num):
                tc = i * dt_yaw
                pc = self.local_data_.position_traj_.evaluateDeBoorT(tc)
                tf = min(self.local_data_.duration_, tc + forward_t)
                pf = self.local_data_.position_traj_.evaluateDeBoorT(tf)
                pd = pf - pc
                pd_norm = np.linalg.norm(pd)
                
                if pd_norm > 1e-6:
                    waypt = Vector3d(np.arctan2(pd[1], pd[0]), 0.0, 0.0)
                    waypt[0] = self.calcNextYaw(last_yaw, waypt[0])
                else:
                    waypt = waypts[-1]

                last_yaw = waypt[0]
                waypts.append(waypt)
                waypt_idx.append(i)

        # Set final state constraints
        end_yaw3d = Vector3d(end_yaw, 0, 0)
        end_yaw3d[0] = self.calcNextYaw(last_yaw, end_yaw3d[0])
        yaw[seg_num:seg_num+3, 0:1] = states2pts @ np.array(end_yaw3d.listify())

        # Check for rapid yaw changes
        if abs(start_yaw3d[0] - end_yaw3d[0]) >= np.pi:
            print("Error: Yaw change rapidly!")
        
        cost_func = BsplineOptimizer.SMOOTHNESS | BsplineOptimizer.WAYPOINTS | \
                    BsplineOptimizer.START | BsplineOptimizer.END
        
        start = [Vector3d(start_yaw3d[0], 0, 0),
                 Vector3d(start_yaw3d[1], 0, 0),
                 Vector3d(start_yaw3d[2], 0, 0)]
        end = [Vector3d(end_yaw3d[0], 0, 0),
               Vector3d(0, 0, 0)]
        self.bspline_optimizers_[1].setBoundaryState(start, end)
        self.bspline_optimizers_[1].setWaypoints(waypts, waypt_idx)
        yaw, dt_yaw = self.bspline_optimizers_[1].optimize(yaw, dt_yaw, cost_func, 1, 1)
        
        self.local_data_.yaw_traj_.setUniformBspline(yaw, 3, dt_yaw)
        self.local_data_.yawdot_traj_ = self.local_data_.yaw_traj_.getDerivative()
        self.local_data_.yawdotdot_traj_ = self.local_data_.yawdot_traj_.getDerivative()

        self.plan_data_.dt_yaw_ = dt_yaw

    def calcNextYaw(self, last_yaw: float, yaw: float):

        round_last = last_yaw
        while round_last < -np.pi:
            round_last += 2 * np.pi
        while round_last > np.pi:
            round_last -= 2 * np.pi

        diff = yaw - round_last
        if abs(diff) <= np.pi:
            yaw = last_yaw + diff
        elif diff > np.pi:
            yaw = last_yaw + diff - 2 * np.pi
        else:  # diff < -np.pi
            yaw = last_yaw + diff + 2 * np.pi

        return yaw
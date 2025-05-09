from .expl_data import ExplorationParam, ExplorationData
from ..utils import Vector3d, Vector3i
from ..env import EDTEnv, SDFmap, RayCaster
from ..perception import FrontierFinder, ViewNode, GraphSearch
from ..plan_manage import FastPlannerManager
from ..searching import AStar
from time import perf_counter
import math
from typing import List
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy

class FastExplorationManager:

    NO_FRONTIER = 0
    FAIL = 1
    SUCCEED = 2

    def __init__(self):

        self.planner_manager_ = FastPlannerManager()
        self.edt_environment_ = self.planner_manager_.edt_environment_
        self.sdf_map_ = self.edt_environment_.sdf_map_
        self.frontier_finder_ = FrontierFinder(self.edt_environment_)

        self.ed_ = ExplorationData()
        self.ep_ = ExplorationParam()

        ######### config #########

        self.ep_.refine_local_ = True
        self.ep_.refined_num_ = 7
        self.ep_.refined_radius_ = 5.0
        self.ep_.top_view_num_ = 15
        self.ep_.max_decay_ = -1.0
        self.ep_.tsp_dir_ = "null"
        self.ep_.relax_time_ = 1.0

        ViewNode.vm_ = 2.0
        ViewNode.am_ = 2.0
        ViewNode.yd_ = 60 * 3.1415926 / 180
        ViewNode.ydd_ = 90 * 3.1415926 / 180
        ViewNode.w_dir_ = 1.5
        ######### config #########
        ViewNode.astar_ = AStar(self.edt_environment_)
        ViewNode.map_ = self.sdf_map_

        self.resolution_ = self.sdf_map_.getResolution()
        origin, size = self.sdf_map_.getRegion()
        ViewNode.caster_ = RayCaster()
        ViewNode.caster_.setParams(self.resolution_, origin)

        self.planner_manager_.path_finder_.lambda_heu_ = 1.0
        self.planner_manager_.path_finder_.max_search_time_ = 1.0


    def planExploreMotion(self, pos: Vector3d, vel: Vector3d, acc: Vector3d, yaw: Vector3d):

        t1 = perf_counter()
        t2 = t1
        self.ed_.views_.clear()
        self.ed_.global_tour_.clear()

        self.frontier_finder_.searchFrontiers()

        frontier_time = perf_counter() - t1
        t1 = perf_counter()

        self.frontier_finder_.computeFrontiersToVisit()
        self.ed_.frontiers_ = self.frontier_finder_.getFrontiers(self.ed_.frontiers_)
        self.ed_.frontier_boxes_ = self.frontier_finder_.getFrontierBoxes(self.ed_.frontier_boxes_)
        self.ed_.dead_frontiers_ = self.frontier_finder_.getDormantFrontiers(self.ed_.dead_frontiers_)

        if len(self.ed_.frontiers_) == 0:
            print("No coverable frontiers found")
            return self.NO_FRONTIER

        self.ed_.points_, self.ed_.yaws_, self.ed_.averages_ = self.frontier_finder_.getTopViewpointsInfo(pos, self.ed_.points_, self.ed_.yaws_, self.ed_.averages_)

        # Add view points for visualization
        for i in range(len(self.ed_.points_)):
            self.ed_.views_.append(
                self.ed_.points_[i] + 2.0 * Vector3d(math.cos(self.ed_.yaws_[i]), math.sin(self.ed_.yaws_[i]), 0))

        view_time = perf_counter() - t1
        print(f"Frontier: {len(self.ed_.frontiers_)}, t: {frontier_time}, viewpoint: {len(self.ed_.points_)}, t: {view_time}")

        # Do global and local tour planning and retrieve the next viewpoint
        next_pos = Vector3d()
        next_yaw = 0.0

        if len(self.ed_.points_) > 1:
            # Find global tour passing through all viewpoints using TSP
            indices = []
            indices = self.findGlobalTour(pos, vel, yaw, indices)

            if self.ep_.refine_local_:
                # Refine next few viewpoints in global tour
                t1 = perf_counter()

                self.ed_.refined_ids_.clear()
                self.ed_.unrefined_points_.clear()
                knum = min(len(indices), self.ep_.refined_num_)
                for i in range(knum):
                    tmp = self.ed_.points_[indices[i]]
                    self.ed_.unrefined_points_.append(tmp)
                    self.ed_.refined_ids_.append(indices[i])
                    if (tmp - pos).norm() > self.ep_.refined_radius_ and len(self.ed_.refined_ids_) >= 2:
                        break

                # Get top N viewpoints for next K frontiers
                self.ed_.n_points_.clear()
                n_yaws = []
                self.ed_.n_points_, n_yaws = self.frontier_finder_.getViewpointsInfo(
                    pos, self.ed_.refined_ids_, self.ep_.top_view_num_, self.ep_.max_decay_, 
                    self.ed_.n_points_, n_yaws)

                self.ed_.refined_points_.clear()
                self.ed_.refined_views_.clear()
                refined_yaws = []
                self.ed_.refined_points_, refined_yaws = self.refineLocalTour(pos, vel, yaw, self.ed_.n_points_, n_yaws, self.ed_.refined_points_, refined_yaws)
                next_pos = self.ed_.refined_points_[0]
                next_yaw = refined_yaws[0]

                # Get markers for view visualization
                for i in range(len(self.ed_.refined_points_)):
                    view = self.ed_.refined_points_[i] + 2.0 * Vector3d(
                        math.cos(refined_yaws[i]), math.sin(refined_yaws[i]), 0)
                    self.ed_.refined_views_.append(view)

                self.ed_.refined_views1_.clear()
                self.ed_.refined_views2_.clear()
                for i in range(len(self.ed_.refined_points_)):
                    v1, v2 = [], []
                    self.frontier_finder_.percep_utils_.setPose(self.ed_.refined_points_[i], refined_yaws[i])
                    self.frontier_finder_.percep_utils_.getFOV(v1, v2)
                    self.ed_.refined_views1_.extend(v1)
                    self.ed_.refined_views2_.extend(v2)

                local_time = perf_counter() - t1
                print(f"Local refine time: {local_time}")

            else:
                # Choose next viewpoint from global tour
                next_pos = self.ed_.points_[indices[0]]
                next_yaw = self.ed_.yaws_[indices[0]]

        elif len(self.ed_.points_) == 1:
            # Only 1 destination, no need for TSP
            self.frontier_finder_.updateFrontierCostMatrix()
            self.ed_.global_tour_ = [pos, self.ed_.points_[0]]
            self.ed_.refined_tour_.clear()
            self.ed_.refined_views1_.clear() 
            self.ed_.refined_views2_.clear()

            if self.ep_.refine_local_:
                # Find min cost viewpoint for next frontier
                self.ed_.refined_ids_ = [0]
                self.ed_.unrefined_points_ = [self.ed_.points_[0]]
                self.ed_.n_points_.clear()
                n_yaws = []
                self.ed_.n_points_, n_yaws = self.frontier_finder_.getViewpointsInfo(
                    pos, [0], self.ep_.top_view_num_, self.ep_.max_decay_,
                    self.ed_.n_points_, n_yaws)

                min_cost = 100000
                min_cost_id = -1
                tmp_path = []
                for i in range(len(self.ed_.n_points_[0])):
                    tmp_cost = ViewNode.computeCost(
                        pos, self.ed_.n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path)
                    if tmp_cost < min_cost:
                        min_cost = tmp_cost
                        min_cost_id = i

                next_pos = self.ed_.n_points_[0][min_cost_id]
                next_yaw = n_yaws[0][min_cost_id]
                self.ed_.refined_points_ = [next_pos]
                self.ed_.refined_views_ = [next_pos + 2.0 * Vector3d(
                    math.cos(next_yaw), math.sin(next_yaw), 0)]

            else:
                next_pos = self.ed_.points_[0]
                next_yaw = self.ed_.yaws_[0]

        else:
            print("Empty destination.")

        t1 = perf_counter()

        # Compute time lower bound of yaw and use in trajectory generation
        diff = abs(next_yaw - yaw[0])
        time_lb = min(diff, 2 * math.pi - diff) / ViewNode.yd_

        # Generate trajectory of x,y,z
        self.planner_manager_.path_finder_.reset()
        if self.planner_manager_.path_finder_.search(pos, next_pos) != AStar.REACH_END:
            print("No path to next viewpoint")
            return self.FAIL

        self.ed_.path_next_goal_ = self.planner_manager_.path_finder_.getPath()
        self.ed_.path_next_goal_ = self.shortenPath(self.ed_.path_next_goal_)

        radius_far = 5.0
        radius_close = 1.5
        length = AStar.pathLength(self.ed_.path_next_goal_)

        if length < radius_close:
            # Next viewpoint is very close, no need to search kinodynamic path
            # Just use waypoints-based optimization
            self.planner_manager_.planExploreTraj(self.ed_.path_next_goal_, vel, acc, time_lb)
            self.ed_.next_goal_ = next_pos

        elif length > radius_far:
            # Next viewpoint is far away, select intermediate goal on geometric path
            print("Far goal.")
            len2 = 0.0
            truncated_path = [self.ed_.path_next_goal_[0]]
            for i in range(1, len(self.ed_.path_next_goal_)):
                if len2 >= radius_far:
                    break
                cur_pt = self.ed_.path_next_goal_[i]
                len2 += (cur_pt - truncated_path[-1]).norm()
                truncated_path.append(cur_pt)

            self.ed_.next_goal_ = truncated_path[-1]
            self.planner_manager_.planExploreTraj(truncated_path, vel, acc, time_lb)

        else:
            # Search kino path to exactly next viewpoint and optimize
            print("Mid goal")
            self.ed_.next_goal_ = next_pos

            if not self.planner_manager_.kinodynamicReplan(pos, vel, acc, self.ed_.next_goal_, Vector3d(), time_lb):
                return self.FAIL

        if self.planner_manager_.local_data_.position_traj_.getTimeSum() < time_lb - 0.1:
            print("Lower bound not satisfied!")

        self.planner_manager_.planYawExplore(yaw, next_yaw, True, self.ep_.relax_time_)

        traj_plan_time = perf_counter() - t1
        t1 = perf_counter()

        yaw_time = perf_counter() - t1
        print(f"Traj: {traj_plan_time}, yaw: {yaw_time}")
        total = perf_counter() - t2
        print(f"Total time: {total}")
        if total > 0.1:
            print("Total time too long!!!")

        return self.SUCCEED
        
    def findGlobalTour(self, cur_pos: Vector3d, cur_vel: Vector3d, cur_yaw: Vector3d, indices: List[int]):
        
        self.frontier_finder_.updateFrontierCostMatrix()
        cost_mat = np.zeros((len(self.ed_.frontiers_), len(self.ed_.frontiers_)))
        cost_mat = self.frontier_finder_.getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat)
        dimension = cost_mat.shape[0]

        manager = pywrapcp.RoutingIndexManager(dimension, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(cost_mat[from_node][to_node])
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)
        indices.clear()
        if solution:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
        else:
            print("No solution found")

        self.ed_.global_tour_ = self.frontier_finder_.getPathForTour(cur_pos, indices, self.ed_.global_tour_)
        return indices
    
    def refineLocalTour(self, cur_pos: Vector3d, cur_vel: Vector3d, cur_yaw: Vector3d, n_points: List[List[Vector3d]], n_yaws: List[List[float]], refined_points: List[Vector3d], refined_yaws: List[float]):

        g_search = GraphSearch()
        last_group = []
        cur_group = []

        # Add the current state
        first = ViewNode(cur_pos, cur_yaw[0])
        first.vel_ = cur_vel
        g_search.addNode(first)
        last_group.append(first)
        final_node = ViewNode()

        # Add viewpoints
        print("Local tour graph: ", end='')
        for i in range(len(n_points)):
            # Create nodes for viewpoints of one frontier
            for j in range(len(n_points[i])):
                node = ViewNode(n_points[i][j], n_yaws[i][j])
                g_search.addNode(node)
                # Connect a node to nodes in last group
                for nd in last_group:
                    g_search.addEdge(nd.id_, node.id_)
                cur_group.append(node)

                # Only keep the first viewpoint of the last local frontier
                if i == len(n_points) - 1:
                    final_node = node
                    break

            # Store nodes for this group for connecting edges
            print(f"{len(cur_group)}, ", end='')
            last_group = cur_group.copy()
            cur_group.clear()
        print("")

        # Search optimal sequence
        path = []
        path = g_search.DijkstraSearch(first.id_, final_node.id_, path)

        # Return searched sequence
        for i in range(1, len(path)):
            refined_points.append(path[i].pos_)
            refined_yaws.append(path[i].yaw_)

        # Extract optimal local tour (for visualization)
        self.ed_.refined_tour_.clear()
        self.ed_.refined_tour_.append(cur_pos)
        ViewNode.astar_.lambda_heu_ = 1.0
        ViewNode.astar_.setResolution(0.2)
        for pt in refined_points:
            path = []
            if ViewNode.searchPath(self.ed_.refined_tour_[-1], pt, path):
                self.ed_.refined_tour_.extend(path)
            else:
                self.ed_.refined_tour_.append(pt)
        ViewNode.astar_.lambda_heu_ = 10000

        return refined_points, refined_yaws
    
    def shortenPath(self, path: List[Vector3d]) -> List[Vector3d]:
        if len(path) == 0:
            print("Empty path to shorten")
            return path

        # Shorten the tour, only critical intermediate points are reserved
        dist_thresh = 3.0
        short_tour = [path[0]]

        for i in range(1, len(path) - 1):
            if (path[i] - short_tour[-1]).norm() > dist_thresh:
                short_tour.append(path[i])
            else:
                # Add waypoints to shorten path only to avoid collision
                ViewNode.caster_.input(short_tour[-1], path[i + 1])
                idx = Vector3i()
                while ViewNode.caster_.nextId(idx):
                    if ViewNode.map_.getInflatedOccupancy(idx) == 1 or \
                       ViewNode.map_.getOccupancy(idx) == ViewNode.map_.UNKNOWN:
                        short_tour.append(path[i])
                        break

        if (path[-1] - short_tour[-1]).norm() > 1e-3:
            short_tour.append(path[-1])

        # Ensure at least three points in the path
        if len(short_tour) == 2:
            short_tour.insert(1, (short_tour[0] + short_tour[1]) / 2)

        path = short_tour.copy()
        return path
from ..utils import Vector3d, Vector3i
from typing import List
import numpy as np

class FSMData:

    def __init__(self):

        self.trigger_ : bool = False
        self.have_odom_ : bool = False
        self.static_state_ : bool = False

        self.state_str_  = []
        self.odom_pos_ = Vector3d()
        self.odom_vel_ = Vector3d()
        self.odom_orient_ = np.array([0.0, 0.0, 0.0, 1.0])  
        self.odom_yaw_ = 0.0

        # Start state
        self.start_pt_ = Vector3d()
        self.start_vel_ = Vector3d() 
        self.start_acc_ = Vector3d()
        self.start_yaw_ = 0.0

        self.start_poss = []  
        self.newest_traj_ = None 

class FSMParam:

    def __init__(self):

        self.replan_thresh1_ = 0.0
        self.replan_thresh2_ = 0.0
        self.replan_thresh3_ = 0.0
        self.replan_time_ = 0.0

        self.frontiers_ : List[List[Vector3d]] = []
        self.dead_frontiers_ : List[List[Vector3d]] = []
        self.frontier_boxes_ : List[tuple[Vector3d, Vector3d]] = []
        self.points_ : List[Vector3d] = []
        self.averages_ : List[Vector3d] = []
        self.views_ : List[Vector3d] = []
        self.yaws_ : List[float] = []
        self.global_tour_ : List[Vector3d] = []

        self.refined_ids_ : List[int] = []
        self.n_points_ : List[List[Vector3d]] = []
        self.unrefined_points_ : List[Vector3d] = []
        self.refined_points_ : List[Vector3d] = []
        self.refined_views_ : List[Vector3d] = []
        self.refined_views1_ : List[Vector3d] = []
        self.refined_views2_ : List[Vector3d] = []
        self.refined_tour_ : List[Vector3d] = []

        self.next_goal_ = Vector3d()
        self.path_next_goal_ : List[Vector3d] = []

        self.views_vis1_ : List[Vector3d] = []
        self.views_vis2_ : List[Vector3d] = []
        self.centers_ : List[Vector3d] = []
        self.scales_ : List[Vector3d] = []

class ExplorationParam:

    def __init__(self):

        self.refine_local_ = False
        self.refined_num_ = 0
        self.refined_radius_ = 0.0
        self.top_view_num_ = 0
        self.max_decay_ = 0.0
        self.tsp_dir_ = ""  # resource dir of tsp solver
        self.relax_time_ = 0.0

class ExplorationData:

    def __init__(self):

        self.frontiers_ : List[List[Vector3d]] = []
        self.dead_frontiers_ : List[List[Vector3d]] = []
        self.frontier_boxes_ : List[tuple[Vector3d, Vector3d]] = []
        self.points_ : List[Vector3d] = []
        self.averages_ : List[Vector3d] = []
        self.views_ : List[Vector3d] = []
        self.yaws_ : List[float] = []
        self.global_tour_ : List[Vector3d] = []

        self.refined_ids_ : List[int] = []
        self.n_points_ : List[List[Vector3d]] = []
        self.unrefined_points_ : List[Vector3d] = []
        self.refined_points_ : List[Vector3d] = []
        self.refined_views_ : List[Vector3d] = []
        self.refined_views1_ : List[Vector3d] = []
        self.refined_views2_ : List[Vector3d] = []
        self.refined_tour_ : List[Vector3d] = []

        self.next_goal_ = Vector3d()
        self.path_next_goal_ : List[Vector3d] = []

        self.views_vis1_ : List[Vector3d] = []
        self.views_vis2_ : List[Vector3d] = []
        self.centers_ : List[Vector3d] = []
        self.scales_ : List[Vector3d] = []

from .expl_data import FSMParam, FSMData
from ..plan_manage import FastPlannerManager
from .fast_exploration_manager import FastExplorationManager
from ..utils import Vector3d, MyTimer
from ..env import MapROS
from ..bspline import NonUniformBspline
import numpy as np


class FastExplorationFSM:

    INIT = 0
    WAIT_TRIGGER = 1
    PLAN_TRAJ = 2
    PUB_TRAJ = 3
    EXEC_TRAJ = 4
    FINISH = 5

    def __init__(self):

        self.fp_ = FSMParam()
        self.fd_ = FSMData()

        ####### config #######
        self.fp_.replan_thresh1_ = 0.5
        self.fp_.replan_thresh2_ = 0.5
        self.fp_.replan_thresh3_ = 1.5
        self.fp_.replan_time_ = 0.2

        self.expl_manager_ = FastExplorationManager()
        self.map_ros_ = MapROS()
        self.planner_manager_ = self.expl_manager_.planner_manager_
        self.map_ros_.setMap(self.planner_manager_.edt_environment_.sdf_map_)
        self.state_ = self.INIT
        self.fd_.have_odom_ = False
        self.fd_.static_state_ = True
        self.fd_.trigger_ = False
        self.force = [0, 0, 0, 0]

        self.exec_timer_ = MyTimer(self, 0.01, self.FSMCallback)
        self.safety_timer_ = MyTimer(self, 0.05, self.safetyCallback)
        self.frontier_timer_ = MyTimer(self, 0.1, self.frontierCallback)

        self.exec_timer_.start()
        self.safety_timer_.start()
        self.frontier_timer_.start()

    def FSMCallback(self, t):

        if self.state_ == self.INIT:
            if not self.fd_.have_odom_:
                return
            else:
                self.transitState(self.PLAN_TRAJ)

        elif self.state_ == self.WAIT_TRIGGER:
            if self.fd_.trigger_:
                self.transitState(self.PLAN_TRAJ)

        elif self.state_ == self.FINISH:
            print("Finish")
            return
        
        elif self.state_ == self.PLAN_TRAJ:
            if self.fd_.static_state_:
                # Plan from static state (hover)
                self.fd_.start_pt_ = self.fd_.odom_pos_
                self.fd_.start_vel_ = self.fd_.odom_vel_
                self.fd_.start_acc_ = Vector3d()
                
                self.fd_.start_yaw_ = Vector3d(self.fd_.odom_yaw_, 0.0, 0.0)
            else:
                # Replan from non-static state, starting from 'replan_time' seconds later
                info = self.planner_manager_.local_data_
                t_r = self.fp_.replan_time_
                
                self.fd_.start_pt_ = info.position_traj_.evaluateDeBoorT(t_r)
                self.fd_.start_vel_ = info.velocity_traj_.evaluateDeBoorT(t_r)
                self.fd_.start_acc_ = info.acceleration_traj_.evaluateDeBoorT(t_r)
                self.fd_.start_yaw_ = Vector3d(
                    info.yaw_traj_.evaluateDeBoorT(t_r)[0],
                    info.yawdot_traj_.evaluateDeBoorT(t_r)[0],
                    info.yawdotdot_traj_.evaluateDeBoorT(t_r)[0]
                )

            res = self.expl_manager_.planExploreMotion(
                self.fd_.start_pt_,
                self.fd_.start_vel_,
                self.fd_.start_acc_,
                self.fd_.start_yaw_
            ) ############# Bspline discarded

            info = self.planner_manager_.local_data_

            pos_bspline = info.position_traj_
            vel_bspline = info.velocity_traj_
            acc_bspline = info.acceleration_traj_
            # Get current position, velocity and acceleration from B-splines
            cur_pos = pos_bspline.evaluateDeBoorT(0.0)  # Current position
            cur_vel = vel_bspline.evaluateDeBoorT(0.0)  # Current velocity  
            cur_acc = acc_bspline.evaluateDeBoorT(0.0)  # Current acceleration

            # Drone parameters
            mass = 0.027  # kg
            arm_length = 0.225  # m 
            k_thrust = 1.0  # Thrust coefficient
            k_torque = 0.0245  # Torque coefficient
            g = 9.81  # Gravity

            # Calculate total thrust needed (mass * acceleration + gravity compensation)
            thrust = mass * (cur_acc[2] + g)  # Total thrust needed

            # Calculate roll and pitch angles from desired acceleration
            roll = np.arcsin((cur_acc[0] * np.cos(self.fd_.odom_yaw_) + 
                            cur_acc[1] * np.sin(self.fd_.odom_yaw_)) / g)
            pitch = np.arcsin((cur_acc[1] * np.cos(self.fd_.odom_yaw_) - 
                             cur_acc[0] * np.sin(self.fd_.odom_yaw_)) / g)

            # Calculate individual motor thrusts
            f1 = thrust/4 + (thrust * arm_length/4) * (np.tan(roll) + np.tan(pitch))  # Front right
            f2 = thrust/4 + (thrust * arm_length/4) * (-np.tan(roll) + np.tan(pitch)) # Front left
            f3 = thrust/4 + (thrust * arm_length/4) * (-np.tan(roll) - np.tan(pitch)) # Rear left
            f4 = thrust/4 + (thrust * arm_length/4) * (np.tan(roll) - np.tan(pitch))  # Rear right

            # Calculate motor torques
            tau1 = f1 * k_torque  # Front right
            tau2 = -f2 * k_torque # Front left  
            tau3 = f3 * k_torque  # Rear left
            tau4 = -f4 * k_torque # Rear right

            self.force = [tau1, tau2, tau3, tau4]
            
            if res == FastExplorationManager.SUCCEED:
                self.transitState(self.PUB_TRAJ)
            elif res == FastExplorationManager.NO_FRONTIER:
                self.transitState(self.FINISH)
                self.fd_.static_state_ = True
            elif res == FastExplorationManager.FAIL:
                # Still in PLAN_TRAJ state, keep replanning
                print("Plan fail")
                self.fd_.static_state_ = True
        
        elif self.state_ == self.PUB_TRAJ:

            self.transitState(self.EXEC_TRAJ)

        elif self.state_ == self.EXEC_TRAJ:

            self.transitState(self.PLAN_TRAJ)


    def transitState(self, new_state: int):
        self.state_ = new_state

    def safetyCallback(self, t):

        if self.state_ == self.EXEC_TRAJ:
            dist = 0.0
            safe, dist = self.planner_manager_.checkTrajCollision(dist)
            if not safe:
                self.transitState(self.PLAN_TRAJ)

    def frontierCallback(self, t):

        if not hasattr(self.frontierCallback, 'delay'):
            self.frontierCallback.delay = 0
        if self.frontierCallback.delay > 5:
            return
        self.frontierCallback.delay += 1

        if self.state_ == self.WAIT_TRIGGER or self.state_ == self.FINISH:
            ft = self.expl_manager_.frontier_finder_
            ed = self.expl_manager_.ed_
            ft.searchFrontiers()
            ft.computeFrontiersToVisit()
            ft.updateFrontierCostMatrix()
            ft.getFrontiers(ed.frontiers_)
            ft.getFrontierBoxes(ed.frontier_boxes_)

    def odometryCallback(self, data):

        # Update odometry position
        self.fd_.odom_pos_ = Vector3d(data["position"][0],
            data["position"][1], 
            data["position"][2])

        # Update odometry velocity
        self.fd_.odom_vel_ = Vector3d(data["velocity"][0],
            data["velocity"][1],
            data["velocity"][2])

        # Update orientation quaternion 
        self.fd_.odom_orient_ = np.array([
            data["orientation"][3],
            data["orientation"][0],
            data["orientation"][1],
            data["orientation"][2]
        ])

        # Calculate yaw from quaternion
        # Using atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
        qw = self.fd_.odom_orient_[0]
        qx = self.fd_.odom_orient_[1] 
        qy = self.fd_.odom_orient_[2]
        qz = self.fd_.odom_orient_[3]
        self.fd_.odom_yaw_ = np.arctan2(2 * (qw * qz + qx * qy), 
                                       1 - 2 * (qy * qy + qz * qz))

        self.fd_.have_odom_ = True
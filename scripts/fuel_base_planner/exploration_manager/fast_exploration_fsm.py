from .expl_data import FSMParam, FSMData
from ..plan_manage import FastPlannerManager
from .fast_exploration_manager import FastExplorationManager
from ..utils import Vector3d, MyTimer


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
        self.fp_.replan_thresh1_ = -1.0
        self.fp_.replan_thresh2_ = -1.0
        self.fp_.replan_thresh3_ = -1.0
        self.fp_.replan_time_ = -1.0

        self.expl_manager_ = FastExplorationManager()
        self.planner_manager_ = self.expl_manager_.planner_manager_
        self.state_ = self.INIT
        self.fd_.have_odom_ = False
        self.fd_.static_state_ = True
        self.fd_.trigger_ = False

        self.exec_timer_ = MyTimer(self, 0.01, self.FSMCallback)
        self.safety_timer_ = MyTimer(self, 0.05, self.safetyCallback)
        self.frontier_timer_ = MyTimer(self, 0.1, self.frontierCallback)

        self.exec_timer_.start()
        self.safety_timer_.start()
        self.frontier_timer_.start()

    def FSMCallback(self):

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

    def safetyCallback(self):

        if self.state_ == self.EXEC_TRAJ:
            dist = 0.0
            safe, dist = self.planner_manager_.checkTrajCollision(dist)
            if not safe:
                self.transitState(self.PLAN_TRAJ)

    def frontierCallback(self):

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


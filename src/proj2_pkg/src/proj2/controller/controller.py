#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import numpy as np
import sys

import tf2_ros
import tf
from std_srvs.srv import Empty as EmptySrv
import rospy
from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
from proj2.planners import RRTPlanner, BicycleConfigurationSpace, analysis 

class BicycleModelController(object):
    def __init__(self):
        """
        Executes a plan made by the planner
        """
        self.pub = rospy.Publisher('/bicycle/cmd_vel', BicycleCommandMsg, queue_size=10)
        self.sub = rospy.Subscriber('/bicycle/state', BicycleStateMsg, self.subscribe)
        self.state = BicycleStateMsg()
        self.primitive_duration = 1.5
        self.since_last_clap = 0

        #position_data = np.loadtxt("../src/proj2_pkg/src/proj2/data/position_data_1_table.txt")
    #curvature_fit(position_data)
        forward_data = np.loadtxt("./src/proj2_pkg/src/proj2/data/forward.txt")
        curved_data = np.loadtxt("./src/proj2_pkg/src/proj2/data/curved.txt")
        (self.phi1, self.v1) = analysis.determine_phi_v_primitives(forward_data)
        (self.phi2, self.v2) = analysis.determine_phi_v_primitives(curved_data)
        self.should_clap = True 
        rospy.on_shutdown(self.shutdown)

    def execute_plan(self, plan):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        if len(plan) == 0:
            return
        rate = rospy.Rate(int(1 / plan.dt))
        start_t = rospy.Time.now()
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            if t > plan.times[-1]:
                break
            state, cmd = plan.get(t)
            if cmd[1] == self.phi2 and (self.since_last_clap > self.primitive_duration or self.was_forward):
                self.clap()
                self.since_last_clap = 0
                self.was_forward = False  
            elif cmd[1] == self.phi1:
                self.was_forward = True 
            else:
                self.since_last_clap += plan.dt 
                self.was_forward = False 
            rate.sleep()

    def clap(self): 
        rospy.logerr("CLAP CLAP CLAP")

    def cmd(self, msg):
        """
        Sends a command to the turtlebot / turtlesim

        Parameters
        ----------
        msg : numpy.ndarray
        """
        self.pub.publish(BicycleCommandMsg(*msg))

    def subscribe(self, msg):
        """
        callback fn for state listener.  Don't call me...
        
        Parameters
        ----------
        msg : :obj:`BicycleStateMsg`
        """
        self.state = np.array([msg.x, msg.y, msg.theta, msg.phi])

    def shutdown(self):
        rospy.loginfo("Shutting Down")
        self.cmd((0, 0))

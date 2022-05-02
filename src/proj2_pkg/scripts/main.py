#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
import sys
import argparse

from std_srvs.srv import Empty as EmptySrv
import rospy

from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
from proj2.planners import RRTPlanner, BicycleConfigurationSpace
from proj2.controller import BicycleModelController

def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, default=200, help='Desired position in x')
    parser.add_argument('-y', type=float, default=200, help='Desired position in y')
    parser.add_argument('-theta', type=float, default=0.0, help='Desired angle')
    parser.add_argument('-phi', type=float, default=0.0, help='Desired angle of the (imaginary) steering wheel')
    return parser.parse_args()

if __name__ == '__main__':
    rospy.init_node('planning', anonymous=False)
    args = parse_args()

    # reset state
    print('Waiting for converter/reset service ...')
    rospy.wait_for_service('/converter/reset')
    print('found!')
    reset = rospy.ServiceProxy('/converter/reset', EmptySrv)
    reset()

    if not rospy.has_param("/environment/obstacles"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    obstacles = rospy.get_param("/environment/obstacles")

    if not rospy.has_param("/environment/low_lims"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    xy_low = rospy.get_param("/environment/low_lims")

    if not rospy.has_param("/environment/high_lims"):
        raise ValueError("No environment information loaded on parameter server. Did you run init_env.launch?")
    xy_high = rospy.get_param("/environment/high_lims")

    if not rospy.has_param("/bicycle_converter/converter/max_steering_angle"):
        raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    phi_max = rospy.get_param("/bicycle_converter/converter/max_steering_angle")

    if not rospy.has_param("/bicycle_converter/converter/max_steering_rate"):
        raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    u2_max = rospy.get_param("/bicycle_converter/converter/max_steering_rate")

    if not rospy.has_param("/bicycle_converter/converter/max_linear_velocity"):
        raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
    u1_max = rospy.get_param("/bicycle_converter/converter/max_linear_velocity")

    print("Obstacles:", obstacles)
    
    controller = BicycleModelController()

    rospy.sleep(1)

    print("Initial State")
    print(controller.state)

    goal = np.array([args.x, args.y, args.theta, args.phi])

    config = BicycleConfigurationSpace( low_lims = xy_low + [-1000, -phi_max],
                                        high_lims = xy_high + [1000, phi_max],
                                        input_low_lims = [-u1_max, -u2_max],
                                        input_high_lims = [u1_max, u2_max],
                                        obstacles = obstacles,
                                        robot_radius = 10,
                                        primitive_duration = controller.primitive_duration)

    planner = RRTPlanner(config, expand_dist=20, max_iter=10000)
    plan = planner.plan_to_pose(controller.state, goal, dt=0.1, prefix_time_length=1)

    print("Predicted Initial State")
    print(plan.start_position())
    print("Predicted Final State")
    print(plan.end_position())

    planner.plot_execution()

    controller.execute_plan(plan)
    print("Final State")
    print(controller.state)

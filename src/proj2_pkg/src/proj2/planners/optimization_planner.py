#!/usr/bin/env python
"""
Starter code for EECS C106B Spring 2022 Project 2.
Author: Valmik Prabhu, Amay Saxena

Implements the optimization-based path planner. You shouldn't need to edt this file.
Just place your completed optimization_planner_casadi.py file in the planners/ folder 
(same folder as this file). It should then work out of the box.
"""

import scipy as sp
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

from .configuration_space import BicycleConfigurationSpace, Plan, expanded_obstacles
from .optimization_planner_casadi import plan_to_pose

class OptimizationPlanner(object):
    def __init__(self, config_space):
        self.config_space = config_space

        self.input_low_lims = self.config_space.input_low_lims
        self.input_high_lims = self.config_space.input_high_lims

    def plan_to_pose(self, start, goal, dt=0.01, N=1000):
        """
            Uses your optimization based path planning algorithm to plan from the 
            start configuration to the goal configuration.
            This function interfaces with your python code from homework.
            
            Please place your completed optimization_planner_casadi.py file in the 
            planners/ folder (same folder as this file).

            Args:
                start: starting configuration of the robot.
                goal: goal configuration of the robot.
                dt: Discretization time step. How much time we would like between
                    subsequent time-stamps.
                N: How many waypoints would we like to have in our path from start
                   to goal
        """

        print("======= Planning with OptimizationPlanner =======")

        # Expand obstacles to account for the radius of the robot.
        with expanded_obstacles(self.config_space.obstacles, self.config_space.robot_radius + 0.05):

            self.plan = None

            q_opt, u_opt = plan_to_pose(np.array(start), np.array(goal), 
                self.config_space.low_lims, self.config_space.high_lims, 
                self.input_low_lims, self.input_high_lims, self.config_space.obstacles, 
                L=self.config_space.robot_length, n=N, dt=dt)

            times = []
            target_positions = []
            open_loop_inputs = []
            t = 0

            for i in range(0, N):
                qi = np.array([q_opt[0][i], q_opt[1][i], q_opt[2][i], q_opt[3][i]])
                ui = np.array([u_opt[0][i], u_opt[1][i]])
                times.append(t)
                target_positions.append(qi)
                open_loop_inputs.append(ui)
                t = t + dt

            # We add one extra step since q_opt has one more state that u_opt
            qi = np.array([q_opt[0][N], q_opt[1][N], q_opt[2][N], q_opt[3][N]])
            ui = np.array([0.0, 0.0])
            times.append(t)
            target_positions.append(qi)
            open_loop_inputs.append(ui)

            self.plan = Plan(np.array(times), np.array(target_positions), np.array(open_loop_inputs), dt)
        return self.plan

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already.
        """
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

def main():
    """Use this function if you'd like to test without ROS.
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([9, 9, 0, 0])
    xy_low = [0, 0]
    xy_high = [10, 10]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = [[6, 3.5, 1.5], [3.5, 6.5, 1]]

    config = BicycleConfigurationSpace( xy_low + [-1000, -phi_max],
                                        xy_high + [1000, phi_max],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15)

    planner = OptimizationPlanner(config)
    plan = planner.plan_to_pose(start, goal)
    planner.plot_execution()

if __name__ == '__main__':
    main()

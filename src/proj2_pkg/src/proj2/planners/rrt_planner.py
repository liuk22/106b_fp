#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import sys
import time

from sklearn import random_projection
import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from configuration_space import HexbugConfigurationSpace, Plan

class RRTGraph(object):

    def __init__(self, *nodes):
        self.nodes = [n for n in nodes] # The first node is the start configuration of the path planning. 
        self.parent = defaultdict(lambda: None)
        self.path = defaultdict(lambda: None)

    def add_node(self, new_config, parent, path):
        new_config = tuple(new_config)
        parent = tuple(parent)
        self.nodes.append(new_config)
        self.parent[new_config] = parent
        self.path[(parent, new_config)] = path

    def get_edge_paths(self):
        for pair in self.path:
            yield self.path[pair]

    def construct_path_to(self, c):
        c = tuple(c)
        return Plan.chain_paths(self.construct_path_to(self.parent[c]), self.path[(self.parent[c], c)]) if self.parent[c] else None

class RRTPlanner(object):

    def __init__(self, config_space, expand_dist, max_iter=10000):
        # config_space should be an object of type ConfigurationSpace
        # (or a subclass of ConfigurationSpace).
        self.config_space = config_space
        # Maximum number of iterations to run RRT for:
        self.max_iter = max_iter
        # Exit the algorithm once a node is sampled within this 
        # distance of the goal:
        self.expand_dist = expand_dist


    def plan_to_pose(self, start, goal, prefix_time_length=100):
        """
            Uses the RRT algorithm to plan from the start configuration
            to the goal configuration.
        """
        print("======= Planning with RRT =======")
        self.graph = RRTGraph(start)
        self.plan = None
        print("Iteration:", 0)
        for it in range(self.max_iter):
            sys.stdout.write("\033[F")
            print("Iteration:", it + 1)
            if rospy.is_shutdown():
                print("Stopping path planner.")
                break
            rand_config = self.config_space.sample_config(goal, self.graph.nodes) # ADD EXISTING POINTS AS A SECOND ARG HERE
            if self.config_space.check_collision(rand_config):
                continue
            closest_config = self.config_space.nearest_config_to(self.graph.nodes, rand_config)
            path = self.config_space.local_plan(closest_config, rand_config)
            if self.config_space.check_path_collision(path):
                continue
            delta_path = path.get_prefix(prefix_time_length)
            new_config = delta_path.end_position()
            self.graph.add_node(new_config, closest_config, delta_path)
            if self.config_space.distance(new_config, goal) <= self.expand_dist:
                self.plan = self.graph.construct_path_to(new_config)
                return self.plan
        print("Failed to find plan in allotted number of iterations.")
        return None


    def execute_plan(self, plan):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        if len(plan) == 0:
            return
        rate = rospy.Rate(10) # Magic number here
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
                self.since_last_clap += 1
                self.was_forward = False 
            rate.sleep()

    def plot_execution(self):
        """
        Creates a plot of the RRT graph on the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already, so that self.graph
        is populated. If planning was successful, then self.plan will be populated 
        and it will be plotted as well.
        """
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        for path in self.graph.get_edge_paths():
            xs = path.positions[:, 0]
            ys = path.positions[:, 1]
            ax.plot(xs, ys, color='orange')

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

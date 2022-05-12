#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import numpy as np
from contextlib import contextmanager
import analysis

class Plan(object):
    """Data structure to represent a motion plan. Stores plans in the form of
    three arrays of the same length: times, positions, and open_loop_inputs.

    The following invariants are assumed:
        - at time times[i] the plan prescribes that we be in position
          positions[i] and perform input open_loop_inputs[i].
        - times starts at zero. Each plan is meant to represent the motion
          from one point to another over a time interval starting at 
          time zero. If you wish to append together multiple paths
          c1 -> c2 -> c3 -> ... -> cn, you should use the chain_paths
          method.
    """

    def __init__(self, times, target_positions, open_loop_inputs):
        self.times = times
        self.positions = target_positions
        self.open_loop_inputs = open_loop_inputs

    def __iter__(self):
        # I have to do this in an ugly way because python2 sucks and
        # I hate it.
        for t, p, c in zip(self.times, self.positions, self.open_loop_inputs):
            yield t, p, c

    def __len__(self):
        return len(self.times)

    def get(self, t):
        """Returns the desired position and open loop input at time t.
        """
        index = int(np.sum(self.times <= t))
        index = index - 1 if index else 0
        return self.positions[index], self.open_loop_inputs[index]

    def end_position(self):
        return self.positions[-1]

    def start_position(self):
        return self.positions[0]

    def get_prefix(self, until_time):
        """Returns a new plan that is a prefix of this plan up until the
        time until_time.
        """
        times = self.times[self.times <= until_time]
        positions = self.positions[self.times <= until_time]
        open_loop_inputs = self.open_loop_inputs[self.times <= until_time]
        return Plan(times, positions, open_loop_inputs)

    @classmethod
    def chain_paths(self, *paths):
        """Chain together any number of plans into a single plan.
        """
        def chain_two_paths(path1, path2):
            """Chains together two plans to create a single plan. Requires
            that path1 ends at the same configuration that path2 begins at.
            """
            if not path1 and not path2:
                return None
            elif not path1:
                return path2
            elif not path2:
                return path1
            assert np.allclose(path1.end_position(), path2.start_position()), "Cannot append paths with inconsistent start and end positions."
            times = np.concatenate((path1.times, path1.times[-1] + path2.times[1:]), axis=0)
            positions = np.concatenate((path1.positions, path2.positions[1:]), axis=0)
            open_loop_inputs = np.concatenate((path1.open_loop_inputs, path2.open_loop_inputs[1:]), axis=0)

            return Plan(times, positions, open_loop_inputs)
        chained_path = None
        for path in paths:
            chained_path = chain_two_paths(chained_path, path)
        return chained_path

@contextmanager
def expanded_obstacles(obstacle_list, delta):
    """Context manager that edits obstacle list to increase the radius of
    all obstacles by delta.
    
    Assumes obstacles are circles in the x-y plane and are given as lists
    of [x, y, r] specifying the center and radius of the obstacle. So
    obstacle_list is a list of [x, y, r] lists.

    Note we want the obstacles to be lists instead of tuples since tuples
    are immutable and we would be unable to change the radii.

    Usage:
        with expanded_obstacles(obstacle_list, 0.1):
            # do things with expanded obstacle_list. While inside this with 
            # block, the radius of each element of obstacle_list has been
            # expanded by 0.1 meters.
        # once we're out of the with block, obstacle_list will be
        # back to normal
    """
    for obs in obstacle_list:
        obs[2] += delta
    yield obstacle_list
    for obs in obstacle_list:
        obs[2] -= delta

class ConfigurationSpace(object):
    """ An abstract class for a Configuration Space. 
    
        DO NOT FILL IN THIS CLASS

        Instead, fill in the BicycleConfigurationSpace at the bottom of the
        file which inherits from this class.
    """

    def __init__(self, dim, low_lims, high_lims, obstacles):
        """
        Parameters
        ----------
        dim: dimension of the state space: number of state variables.
        low_lims: the lower bounds of the state variables. Should be an
                iterable of length dim.
        high_lims: the higher bounds of the state variables. Should be an
                iterable of length dim.
        obstacles: A list of obstacles. This could be in any representation
            we choose, based on the application. In this project, for the bicycle
            model, we assume each obstacle is a circle in x, y space, and then
            obstacles is a list of [x, y, r] lists specifying the center and 
            radius of each obstacle.

        """
        self.dim = dim
        self.low_lims = np.array(low_lims)
        self.high_lims = np.array(high_lims)
        self.obstacles = obstacles


    def distance(self, c1, c2):
        """
            Implements the chosen metric for this configuration space.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.

            Returns the distance between configurations c1 and c2 according to
            the chosen metric.
        """
        pass

    def sample_config(self, *args):
        """
            Samples a new configuration from this C-Space according to the
            chosen probability measure.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.

            Returns a new configuration sampled at random from the configuration
            space.
        """
        pass

    def check_collision(self, c):
        """
            Checks to see if the specified configuration c is in collision with
            any obstacles.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.
        """
        pass

    def check_path_collision(self, path):
        """
            Checks to see if a specified path through the configuration space is 
            in collision with any obstacles.
            This method should be implemented whenever this ConfigurationSpace
            is subclassed.
        """
        pass

    def local_plan(self, c1, c2):
        """
            Constructs a plan from configuration c1 to c2.

            This is the local planning step in RRT. This should be where you extend
            the trajectory of the robot a little bit starting from c1. This may not
            constitute finding a complete plan from c1 to c2. Remember that we only
            care about moving in some direction while respecting the kinemtics of
            the robot. You may perform this step by picking a number of motion
            primitives, and then returning the primitive that brings you closest
            to c2.
        """
        pass

    def nearest_config_to(self, config_list, config):
        """
            Finds the configuration from config_list that is closest to config.
        """
        return min(config_list, key=lambda c: self.distance(c, config))

class HexbugConfigurationSpace(ConfigurationSpace):
    """
        The configuration space for a Bicycle modeled robot
        Obstacles should be tuples (x, y, r), representing circles of 
        radius r centered at (x, y)
        We assume that the robot is circular and has radius equal to robot_radius
        The state of the robot is defined as (x, y, theta, phi).
    """
    def __init__(self, low_lims, high_lims, input_low_lims, input_high_lims, obstacles, robot_radius, primitive_duration, goal_bias=0.5):
        dim = 4
        super(HexbugConfigurationSpace, self).__init__(dim, low_lims, high_lims, obstacles)
        self.robot_radius = robot_radius
        self.robot_length = 1
        self.input_low_lims = input_low_lims
        self.input_high_lims = input_high_lims
        forward_data = np.loadtxt("../data/forward_may_4.txt")
        curved_data = np.loadtxt("../data/backward_may_4.txt")

        (self.phi1, self.v1) = analysis.determine_phi_v_primitives(forward_data)
        self.phi1 *= -1 # vision shows it's going wrong way
        (self.phi2, self.v2) = analysis.determine_phi_v_primitives(curved_data)

        self.primitive_duration = primitive_duration
        self.goal_bias = goal_bias

    def config2image_coords(self, config_xy):
        new_xy = np.array(config_xy).astype(int)
        new_xy = tuple(new_xy)
        

        #new_xy[1] = self.high_lims[1] - config_xy[1]
        return new_xy


    def distance(self, c1, c2):
        """
        c1 and c2 should be numpy.ndarrays of size (4,)
        """
        x1, y1, theta1, _ = c1
        x2, y2, theta2, _ = c2

        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        beta = 0.5
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + beta*((a2 - a1)**2 + (b2 - b1)**2)) 

    def sample_config(self, *args):
        """
        Pick a random configuration from within our state boundaries.

        You can pass in any number of additional optional arguments if you
        would like to implement custom sampling heuristics. By default, the
        RRT implementation passes in the goal as an additional argument,
        which can be used to implement a goal-biasing heuristic.
        """
        # Use goal bias
        p = self.goal_bias
        goal = args[0]
        existing_pts = args[1]
        min_d = min([self.distance(xi, goal) for xi in existing_pts])
        sample_pt = None 
        if np.random.rand() < p:
            rand_r = np.random.rand() * min_d
            rand_alpha = np.random.rand() * 2 * np.pi 
            rand_theta = np.random.rand() * np.pi/2  - np.pi/4
            sample_pt = rand_r * np.cos(rand_alpha) + goal[0], rand_r * np.sin(rand_alpha)+ goal[1], rand_theta + goal[2], 0
        else:
            x = np.random.rand() * (self.high_lims[0] - self.low_lims[0]) + self.low_lims[0]
            y = np.random.rand() * (self.high_lims[1] - self.low_lims[1]) + self.low_lims[1]
            theta = np.random.rand() * (self.high_lims[2] - self.low_lims[2]) + self.low_lims[2]
            sample_pt = x, y, theta, 0
        return np.array(sample_pt)
            
    def check_collision(self, c):
        """
        Returns true if a configuration c is in collision
        c should be a numpy.ndarray of size (4,)
        """
        x, y, _, _ = c
        for obsX, obsY, obsR in self.obstacles:
            if (x - obsX)**2 + (y - obsY)**2 <= (obsR + self.robot_radius)**2:
                return True
        return False

    def check_path_collision(self, path):
        """
        Returns true if the input path is in collision. The path
        is given as a Plan object. See configuration_space.py
        for details on the Plan interface.

        You should also ensure that the path does not exceed any state bounds,
        and the open loop inputs don't exceed input bounds.
        """
        for t, p, c in path:
            if self.check_collision(p):
                return True
            if any(p > self.high_lims) or any(p < self.low_lims):
                return True
            if any(c > self.input_high_lims) or any(c < self.input_low_lims):
                return True
        return False

    def local_plan(self, c1, c2):
        """
        Constructs a local plan from c1 to c2. Usually, you want to
        just come up with any plan without worrying about obstacles,
        because the algorithm checks to see if the path is in collision,
        in which case it is discarded.

        This should return a cofiguration_space.Plan object.
        """

        c1 = np.array(c1)
        c2 = np.array(c2)
        timesteps = int(self.primitive_duration + 1)
        times = np.linspace(0, self.primitive_duration, timesteps)
        
        # Create a set of motion primitives from c1.
        def generate_target_positions(ol_inputs):
            target_positions = np.zeros((timesteps, 4))
            target_positions[0, :] = c1 # starting position 
            for t in range(1, timesteps):
                x, y, theta, _ = target_positions[t-1, :]
                theta += ol_input[t, 1]
                speeds = np.array([np.cos(theta), np.sin(theta), 0, 0]) * ol_inputs[t, 0] 
                target_positions[t, :] = target_positions[t-1, :] + speeds * 1
                target_positions[t, 2] = theta
            return target_positions

        input_primitives = {
            'forward': np.vstack((self.v1 * np.ones(timesteps), self.phi1 * np.ones(timesteps))).T,
            'back_j': np.vstack((self.v2 * -1 * np.ones(timesteps), self.phi2 * np.ones(timesteps))).T
        }
        
        min_d = float('inf')
        closest_plan = None
        for name in input_primitives:
            ol_input = input_primitives[name]
            target_positions = generate_target_positions(ol_input)
            plan = Plan(times, target_positions, ol_input)
            d = self.distance(plan.end_position(), c2)
            if d < min_d:
                min_d = d
                closest_plan = plan
        return closest_plan 
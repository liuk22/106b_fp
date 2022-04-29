#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
Adapted for Spring 2020 by Amay Saxena
"""
import numpy as np
from scipy.integrate import quad
import sys
from copy import copy
import matplotlib.pyplot as plt
from configuration_space import Plan, BicycleConfigurationSpace

class SinusoidPlanner():
    def __init__(self, config_space):
        """
        Turtlebot planner that uses sequential sinusoids to steer to a goal pose.

        config_space should be a BicycleConfigurationSpace object.
        Parameters
        ----------
        l : float
            length of car
        """
        self.config_space = config_space
        self.l = config_space.robot_length
        self.max_phi = config_space.high_lims[3]
        self.max_u1 = config_space.input_high_lims[0]
        self.max_u2 = config_space.input_high_lims[1]

    def plan_to_pose(self, start_state, goal_state, dt = 0.01, delta_t=2):
        """
        Plans to a specific pose in (x,y,theta,phi) coordinates.  You 
        may or may not have to convert the state to a v state with state2v()
        You may want to plan each component separately
        so that you can reset phi in case there's drift in phi.

        You will need to edit some or all of this function to take care of
        configuration

        Parameters
        ----------
        start_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        goal_state: numpy.ndarray of shape (4,) [x, y, theta, phi]
        dt : float
            how many seconds between trajectory timesteps
        delta_t : float
            how many seconds each trajectory segment should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """

        print("======= Planning with SinusoidPlanner =======")

        self.plan = None
        # This bit hasn't been exhaustively tested, so you might hit a singularity anyways
        x_s, y_s, theta_s, phi_s = start_state
        x_g, y_g, theta_g, phi_g = goal_state
        max_abs_angle = max(abs(theta_g), abs(theta_s))
        min_abs_angle = min(abs(theta_g), abs(theta_s))
        
        # We can only change phi up to some threshold
        self.phi_dist = min(
            abs(phi_g - self.max_phi),
            abs(phi_g + self.max_phi)
        )
        # # dummy steering with alternative model
        # y_path2 = self.steer_x(start_state, goal_state, dt=dt, delta_t=delta_t, altModel=True)
        # phi_path2 = self.steer_phi(y_path2.end_position(), goal_state, dt=dt, delta_t=delta_t, altModel=True)
        # alpha_path2 = self.steer_alpha(phi_path2.end_position(), goal_state, dt=dt, delta_t=delta_t, altModel=True)
        # x_path2 = self.steer_y(alpha_path2.end_position(), goal_state, dt=dt, delta_t=delta_t, altModel=True)
        # print("end position", x_path2.end_position())
        # self.plan = Plan.chain_paths(y_path2 , phi_path2 , alpha_path2 , x_path2)
        # return self.plan

        # Point turn
        # if start_state[1] == goal_state[0] and start_state[0] == goal_state[1] and start_state[2] == 0 and goal_state[2] == 3.142 and start_state[3] == goal_state[3]: 
        #     target1 = np.array([1, 1, np.pi/4, 0])
        #     target2 = np.array([1, 1, 3 * np.pi/4, 0])

        #     # First, steer with the canonical model to 45degrees
        #     x_path1 = self.steer_x(start_state, target1, dt=dt, delta_t=delta_t)
        #     phi_path1 = self.steer_phi(x_path1.end_position(), target1, dt=dt, delta_t=delta_t)
        #     alpha_path1 = self.steer_alpha(phi_path1.end_position(), target1, dt=dt, delta_t=delta_t)
        #     y_path1 = self.steer_y(alpha_path1.end_position(), target1, dt=dt, delta_t=delta_t)

        #     # print("end position", y_path1.end_position())
        #     # Then, steer with the alternate model from 45 to 135
        #     y_path2 = self.steer_x(y_path1.end_position(), target2, dt=dt, delta_t=delta_t, altModel=True)
        #     phi_path2 = self.steer_phi(y_path2.end_position(), target2, dt=dt, delta_t=delta_t)
        #     alpha_path2 = self.steer_alpha(phi_path2.end_position(), target2, dt=dt, delta_t=delta_t, altModel=True)
        #     x_path2 = self.steer_y(alpha_path2.end_position(), target2, dt=dt, delta_t=delta_t, altModel=True)
        #     print("end position", x_path2.end_position())

        #     # # Finally, steer with canonical model from 135 to 180
        #     # x_path3 = self.steer_x(x_path2.end_position(), goal_state, dt=dt, delta_t=delta_t)
        #     # phi_path3 = self.steer_phi(x_path3.end_position(), goal_state, dt=dt, delta_t=delta_t)
        #     # alpha_path3 = self.steer_alpha(phi_path3.end_position(), goal_state, dt=dt, delta_t=delta_t)
        #     # y_path3 = self.steer_y(alpha_path3.end_position(), goal_state, dt=dt, delta_t=delta_t)
            
        #     self.plan = Plan.chain_paths(x_path1, phi_path1, alpha_path1, y_path1, y_path2 , phi_path2 , alpha_path2 , x_path2)#, x_path3, phi_path3, alpha_path3, y_path3)
        #     return self.plan

        if (max_abs_angle > np.pi/2) and (min_abs_angle < np.pi/2):
            raise ValueError("You'll cause a singularity here. You should add something to this function to fix it")

        if abs(phi_s) > self.max_phi or abs(phi_g) > self.max_phi:
            raise ValueError("Either your start state or goal state exceeds steering angle bounds")

        x_path =        self.steer_x(
                            start_state, 
                            goal_state, 
                            dt=dt, 
                            delta_t=delta_t
                        )

        phi_path =      self.steer_phi(
                            x_path.end_position(), 
                            goal_state,  
                            dt=dt, 
                            delta_t=delta_t
                        )
        alpha_path =    self.steer_alpha(
                            phi_path.end_position(), 
                            goal_state, 
                            dt=dt, 
                            delta_t=delta_t
                        )

        y_path =        self.steer_y(
                            alpha_path.end_position(), 
                            goal_state,
                            dt=dt,
                            delta_t=delta_t
                        )

        self.plan = Plan.chain_paths(x_path, phi_path, alpha_path, y_path)
        
        return self.plan

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already, so that self.graph
        is populated. If planning was successful, then self.plan will be populated 
        and it will be plotted as well.
        """
        ax = plt.subplot(1, 1, 1)

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

    def steer_x(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2, altModel=False):
        """
        Create a Plan to move the turtlebot in the x direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state, altModel=altModel)
        goal_state_v = self.state2v(goal_state, altModel=altModel)
        delta_x = goal_state_v[0] - start_state_v[0]

        v1 = delta_x/delta_t
        v2 = 0

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt, altModel=altModel)

    def steer_phi(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2, altModel=False):
        """
        Create a trajectory to move the turtlebot in the phi direction

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state, altModel=altModel)
        goal_state_v = self.state2v(goal_state, altModel=altModel)
        delta_phi = goal_state_v[1] - start_state_v[1]

        v1 = 0
        v2 = delta_phi/delta_t

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1, v2])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt, altModel=altModel)

    def steer_alpha(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2, altModel=False):
        """
        Create a trajectory to move the turtlebot in the alpha direction.  
        Remember dot{alpha} = f(phi(t))*u_1(t) = f(frac{a_2}{omega}*sin(omega*t))*a_1*sin(omega*t)
        also, f(phi) = frac{1}{l}tan(phi)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        # alternate model 
        start_state_v = self.state2v(start_state, altModel=altModel)
        goal_state_v = self.state2v(goal_state, altModel=altModel)
        delta_alpha = goal_state_v[2] - start_state_v[2]

        omega = 2*np.pi / delta_t

        a2 = min(1, self.phi_dist*omega)
        
        f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model]
        if altModel:
            f = lambda phi: -(1/self.l)*np.tan(phi) # This is from the car model]
        phi_fn = lambda t: (a2/omega)*np.sin(omega*t) + start_state_v[1]
        integrand = lambda t: f(phi_fn(t))*np.sin(omega*t) # The integrand to find beta
        beta1 = (omega/np.pi) * quad(integrand, 0, delta_t)[0]

        a1 = (delta_alpha*omega)/(np.pi*beta1)
              
        v1 = lambda t: a1*np.sin(omega * t)
        v2 = lambda t: a2*np.cos(omega * t)

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt, altModel=altModel)


    def steer_y(self, start_state, goal_state, t0 = 0, dt = 0.01, delta_t = 2, altModel=False):
        """
        Create a trajectory to move the turtlebot in the y direction. 
        Remember, dot{y} = g(alpha(t))*v1 = frac{alpha(t)}{sqrt{1-alpha(t)^2}}*a_1*sin(omega*t)
        See the doc for more math details

        Parameters
        ----------
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            current state of the turtlebot
        goal_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            desired state of the turtlebot
        t0 : float
            what timestep this trajectory starts at
        dt : float
            how many seconds between each trajectory point
        delta_t : float
            how many seconds the trajectory should run for

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        start_state_v = self.state2v(start_state, altModel=altModel)
        goal_state_v = self.state2v(goal_state, altModel=altModel)
        delta_y = goal_state_v[3] - start_state_v[3]

        omega = 2*np.pi / delta_t
        
        a2 = min(1, self.phi_dist*omega) # from previous part
        def best_a1(a2):

            def actual_beta1(a1): 
                phi_fn = lambda tau: (a2/(2*omega))*np.sin(2*omega*tau) + start_state_v[1]
                
                f = lambda phi: (1/self.l)*np.tan(phi) # This is from the car model
                if altModel:
                    f = lambda phi: -(1/self.l)*np.tan(phi) # This is from the car model
                g = lambda alpha: alpha / ((1 - alpha**2)**(1/2)) # This is from the car model

                integrand1 = lambda tau: f(phi_fn(tau)) * a1 * np.sin(omega * tau) # integrand to find alpha
                def alpha_fn(t):
                    #if altModel: return start_state_v[2] - quad(integrand1, 0, t)[0]
                    return start_state_v[2] + quad(integrand1, 0, t)[0]

                integrand2 = lambda t: g(alpha_fn(t)) * np.sin(omega*t) # The integrand to find beta
                beta1 = omega / np.pi * quad(integrand2, 0, delta_t)[0]
                return beta1

            def desired_beta1(a1): 
                return delta_y * omega / (np.pi * a1)

            a1_upper = 10
            a1_lower = 1e-3
            a1_curr = (a1_upper - a1_lower) / 2 + a1_lower
             
            eps = 1e-3
            
            curr_err = float('inf')            
            while abs(curr_err) > eps and abs(a1_curr - a1_upper) > eps and abs(a1_curr - a1_lower) > eps:  
                curr_err = desired_beta1(a1_curr) - actual_beta1(a1_curr)
                if curr_err < 0: 
                    a1_upper = a1_curr
                else:
                    a1_lower = a1_curr
                print("a1_curr", a1_curr, "error", curr_err, "phi", start_state_v[3])
                a1_curr = (a1_upper - a1_lower) / 2 + a1_lower 
            
            return a1_curr
    
        a1 = best_a1(a2)

        # Steer using the a1 and a2 amplitudes we found.
        v1 = lambda t: a1*np.sin(omega*(t))
        v2 = lambda t: a2*np.cos(2*omega*(t))

        #print("v1", [v1(t-t0) for t in np.linspace(t0, t0+delta_t)])
        #print("v2", [v2(t-t0) for t in np.linspace(t0, t0+delta_t)])

        path, t = [], t0
        while t < t0 + delta_t:
            path.append([t, v1(t-t0), v2(t-t0)])
            t = t + dt
        return self.v_path_to_u_path(path, start_state, dt, altModel=altModel)
                


    def state2v(self, state, altModel=False):
        """
        Takes a state in (x,y,theta,phi) coordinates and returns a state of (x,phi,alpha,y)

        Parameters
        ----------
        state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            some state

        Returns
        -------
        4x1 :obj:`numpy.ndarray` 
            x, phi, alpha, y
        """
        x, y, theta, phi = state
        if altModel:
            return np.array([y, phi, np.cos(theta), x])
        return np.array([x, phi, np.sin(theta), y])

    def v_path_to_u_path(self, path, start_state, dt, altModel=False):
        """
        convert a trajectory in v commands to u commands

        Parameters
        ----------
        path : :obj:`list` of (float, float, float)
            list of (time, v1, v2) commands
        start_state : numpy.ndarray of shape (4,) [x, y, theta, phi]
            starting state of this trajectory
        dt : float
            how many seconds between timesteps in the trajectory

        Returns
        -------
        :obj: Plan
            See configuration_space.Plan.
        """
        def v2cmd(v1, v2, state):
            
            u1 = v1/np.cos(state[2])
            u2 = v2
            if altModel:
                u1 = v1/np.sin(state[2])
                u2 = v2
            return [u1, u2]

        curr_state = start_state
        positions = []
        times = []
        open_loop_inputs = []
        for i, (t, v1, v2) in enumerate(path):
            cmd_u = v2cmd(v1, v2, curr_state)
            positions.append(curr_state)
            open_loop_inputs.append(cmd_u)
            times.append(t)

            x, y, theta, phi = curr_state
            linear_velocity, steering_rate = cmd_u
            curr_state = [
                x     + np.cos(theta)               * linear_velocity*dt,
                y     + np.sin(theta)               * linear_velocity*dt,
                theta + np.tan(phi) / float(self.l) * linear_velocity*dt,
                phi   + steering_rate*dt
            ]

        return Plan(np.array(times), np.array(positions), np.array(open_loop_inputs), dt=dt)

def main():
    """Use this function if you'd like to test without ROS.
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([1.3, 1.1, 0, 0])# og goal = np.array([2, 1.3, 0.7, 0])
    xy_low = [0, 0]
    xy_high = [5, 5]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = []

    config = BicycleConfigurationSpace( xy_low + [-1000, -phi_max],
                                        xy_high + [1000, phi_max],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15)

    planner = SinusoidPlanner(config)
    plan = planner.plan_to_pose(start, goal, 0.01, 2.0)
    planner.plot_execution()

if __name__ == '__main__':
    main()

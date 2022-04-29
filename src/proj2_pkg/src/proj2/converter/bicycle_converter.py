#!/usr/bin/env python
"""
Starter code for EE106B Turtlebot Lab
Author: Valmik Prabhu, Chris Correa
"""
import rospy
from geometry_msgs.msg import Twist, PoseWithCovariance, TwistWithCovariance
from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
import numpy as np
import tf2_ros
import tf
from std_srvs.srv import Empty as EmptySrv, EmptyResponse
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

class BicycleConverter():
    """docstring for BicycleConverter"""

    def get_params(self):
        if not rospy.has_param("~converter/sim"):
            raise ValueError("Converter sim flag not found on parameter server")    
        self.sim = rospy.get_param("~converter/sim")

        if not rospy.has_param("~converter/length"):
            raise ValueError("Converter length not found on parameter server")    
        self.length = rospy.get_param("~converter/length")

        if not rospy.has_param("~converter/turtlebot_command_topic"):
            raise ValueError("Converter input topic not found on parameter server")    
        self.turtlebot_command_topic = rospy.get_param("~converter/turtlebot_command_topic")

        if not rospy.has_param("~converter/sim_command_topic"):
            raise ValueError("Converter input topic not found on parameter server")    
        self.sim_command_topic = rospy.get_param("~converter/sim_command_topic")

        if not rospy.has_param("~converter/sim_pose_topic"):
            raise ValueError("Converter output topic not found on parameter server")
        self.sim_pose_topic = rospy.get_param("~converter/sim_pose_topic")

        if not rospy.has_param("~converter/fixed_frame"):
            raise ValueError("Converter output topic not found on parameter server")
        self.fixed_frame = rospy.get_param("~converter/fixed_frame")

        if not rospy.has_param("~converter/robot_frame"):
            raise ValueError("Converter output topic not found on parameter server")
        self.robot_frame = rospy.get_param("~converter/robot_frame")

        if not rospy.has_param("~converter/state_topic"):
            raise ValueError("Converter output topic not found on parameter server")
        self.state_topic = rospy.get_param("~converter/state_topic")

        if not rospy.has_param("~converter/bicycle_command_topic"):
            raise ValueError("Converter output topic not found on parameter server")
        self.bicycle_command_topic = rospy.get_param("~converter/bicycle_command_topic")

        if not rospy.has_param("~converter/max_steering_angle"):
            raise ValueError("Max Steering Angle not found on parameter server")
        self.max_steering_angle = rospy.get_param("~converter/max_steering_angle")

        if not rospy.has_param("~converter/max_steering_rate"):
            raise ValueError("Max Steering Rate not found on parameter server")
        self.max_steering_rate = rospy.get_param("~converter/max_steering_rate")

        if not rospy.has_param("~converter/max_linear_velocity"):
            raise ValueError("Max Linear Velocity not found on parameter server")
        self.max_linear_velocity = rospy.get_param("~converter/max_linear_velocity")


    def __init__(self):
        self._name = rospy.get_name()
        self.last_time = rospy.Time.now()
        self.rate_hz = 200
        self.rate = rospy.Rate(self.rate_hz)
        self.get_params()

        self.state = BicycleStateMsg()
        self.command = BicycleCommandMsg()
        if self.sim:
            self.sim_subscriber = rospy.Subscriber(self.sim_pose_topic, Odometry, self.update_sim_pose)
            self.unicycle_command_topic = self.sim_command_topic

        else:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self.unicycle_command_topic = self.turtlebot_command_topic

            # resetting stuff
            self.reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', EmptyMsg, queue_size=10)
        self.command_publisher = rospy.Publisher(self.unicycle_command_topic, Twist, queue_size = 1)
        self.state_publisher = rospy.Publisher(self.state_topic, BicycleStateMsg, queue_size = 1)
        self.subscriber = rospy.Subscriber(self.bicycle_command_topic, BicycleCommandMsg, self.command_listener)
        rospy.Service('converter/reset', EmptySrv, self.reset)
        rospy.on_shutdown(self.shutdown)

    def command_listener(self, msg):
        msg.steering_rate = max(min(msg.steering_rate, self.max_steering_rate), -self.max_steering_rate)
        msg.linear_velocity = max(min(msg.linear_velocity, self.max_linear_velocity), -self.max_linear_velocity)

        self.command = msg
        self.last_time = rospy.Time.now() # Save the time of last command for safety

    def update_sim_pose(self, msg):
        self.state.x = msg.pose.pose.position.x 
        self.state.y = msg.pose.pose.position.y 
        o = msg.pose.pose.orientation
        theta = tf.transformations.euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.state.theta = theta[2]

    def run(self):
        while not rospy.is_shutdown():

            # If we aren't using sim, get the state
            if not self.sim:
                for i in range(100):
                    try:
                        pose = self.tf_buffer.lookup_transform(
                            self.fixed_frame, self.robot_frame, rospy.Time())
                        break
                    except (tf2_ros.LookupException,
                            tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException):
                        pass
                if i == 99:
                    rospy.logerr("%s: Could not extract pose from TF. Using last-known transform", self._name)
                self.state.x = pose.transform.translation.x
                self.state.y = pose.transform.translation.y
                (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                    [pose.transform.rotation.x, pose.transform.rotation.y,
                     pose.transform.rotation.z, pose.transform.rotation.w])
                self.state.theta = yaw

            # Now publish the state
            self.state_publisher.publish(self.state)

            # Now execute commands
            # Timeout to ensure that only recent commands are executed
            if (rospy.Time.now() - self.last_time).to_sec() > 1.0:
                self.command.steering_rate = 0
                self.command.linear_velocity = 0

            # We output velocity and yaw rate based on the bicycle model
            output = Twist()
            output.linear.x = self.command.linear_velocity
            output.angular.z = (1/self.length)*np.tan(self.state.phi)*self.command.linear_velocity

            self.state.phi = self.state.phi + self.command.steering_rate/self.rate_hz
            self.state.phi = max(min(self.state.phi, self.max_steering_angle), -self.max_steering_angle)

            self.command_publisher.publish(output)

            self.rate.sleep()

    def reset(self, req):
        if not self.sim:
            self.reset_odom.publish(EmptyMsg())
            self.state = BicycleStateMsg()
        else:
            self.state.phi = 0
        return EmptyResponse()

    def shutdown(self):
    	rospy.loginfo("Shutting Down")
    	self.command_publisher.publish(Twist()) # Stop moving

if __name__ == '__main__':
    rospy.init_node("Bicycle Conversion", anonymous=True)
    rospy.loginfo("To Stop Turtlebot hit Ctrl-C")

    converter = BicycleConverter()
    converter.run()
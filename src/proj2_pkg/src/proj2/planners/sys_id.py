#!/usr/bin/env python
import numpy as np
import cv2
from utils.utils import * 
import matplotlib.pyplot as plt


import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from proj2.planners import RRTPlanner, HexbugConfigurationSpace

SYSTEM_ID_MODE = False
GLOBAL_DT = 0.2

def transform_to_posestamped(transform):
    """
    SE(3) transform to PoseStamped
    """
    ps = PoseStamped()
    ps.pose.position.x = transform[0, 3] # 0.827 
    ps.pose.position.y = transform[1, 3] # -0.380
    ps.pose.position.z = transform[2, 3] # -0.209 

    quat = quaternion_from_matrix(transform[:3, :3])
    ps.pose.orientation.x = quat[0] # 0 #
    ps.pose.orientation.y = quat[1] # 1 #
    ps.pose.orientation.z = quat[2] # 0 #
    ps.pose.orientation.w = quat[3] # 0 #

    ps.header.frame_id = 'base'
    return ps 

def localize_robot_initial_track_rect(cv_bridge, camera_image_topic, camera_info_topic):
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)

    # use frame delta of first 2 frames and get bounding box 
    image_1 = rospy.wait_for_message(camera_image_topic, Image)
    image_1 = cv_bridge.imgmsg_to_cv2(image_1, desired_encoding='passthrough')
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    image_2 = rospy.wait_for_message(camera_image_topic, Image)
    image_2 = cv_bridge.imgmsg_to_cv2(image_2, desired_encoding='passthrough')
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    image_delta = cv2.absdiff(image_1, image_2_gray)
    _, image_delta = cv2.threshold(image_delta, 25, 255, cv2.THRESH_BINARY)
    image_delta = cv2.dilate(image_delta, None, iterations=2)

    # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    contour_list, hierarchy = cv2.findContours(image_delta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour_list.sort(key=cv2.contourArea)

    largest_contour = contour_list[-1]
    return (cv2.boundingRect(largest_contour), image_2)

def two_rects_to_state(b_rect_new, b_rect_old):
    xy =  np.array([(b_rect_new[0] + b_rect_new[2] / 2), 
                        (b_rect_new[1] + b_rect_new[3] / 2)]).astype(int)
    xy_old =  np.array([(b_rect_old[0] + b_rect_old[2] / 2), 
            (b_rect_old[1] + b_rect_old[3] / 2)]).astype(int)
    direction = xy - xy_old
    theta = np.arctan2(direction[1], direction[0])
    return np.array([xy[0], xy[1], theta + np.pi, 0])

def show_cv2_plan(mat, plan, planner):
    for t, p, c in plan:
        center = tuple(planner.config_space.config2image_coords(p[:2].astype(int)))
        mat = cv2.circle(mat, center=center, radius=1, color=(0, 165, 255), thickness=1)
    return mat 

def online_planning(camera_image_topic, camera_info_topic, camera_frame, planner, goal):
    bridge = CvBridge()
    vw = cv2.VideoWriter('output_2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (planner.config_space.high_lims[0], planner.config_space.high_lims[1]))
    tracking_rect, image_2 =  localize_robot_initial_track_rect(bridge, camera_image_topic, camera_info_topic)

    # KCF object tracking, get the initial theta state of robot needs two timestep approximation
    tracker = cv2.TrackerKCF_create()
    tracker.init(image_2, tracking_rect)
    image = rospy.wait_for_message(camera_image_topic, Image)
    mat = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    last_track_rect = tracking_rect
    for _ in range(3):
        image = rospy.wait_for_message(camera_image_topic, Image)
    image = rospy.wait_for_message(camera_image_topic, Image)
    mat = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    err_code, tracking_rect = tracker.update(mat)
    init_robot_state = two_rects_to_state(tracking_rect, last_track_rect)
    if not SYSTEM_ID_MODE:
        plan = planner.plan_to_pose(init_robot_state, goal)

    xys = []
    t = 0 
    total_time = 5
    while not rospy.is_shutdown(): # and t < 10 * total_time: 
        image = rospy.wait_for_message(camera_image_topic, Image)
        mat = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        last_track_rect = tracking_rect
        err_code, tracking_rect = tracker.update(mat)
        
        mat = cv2.rectangle(mat, (int(tracking_rect[0]), int(tracking_rect[1])), (int(tracking_rect[0] + tracking_rect[2]), int(tracking_rect[1] + tracking_rect[3])), 255, thickness=2)
        xy =  np.array([(tracking_rect[0] + tracking_rect[2] / 2), 
                        (tracking_rect[1] + tracking_rect[3] / 2)]).astype(int)
        xy_old =  np.array([(last_track_rect[0] + last_track_rect[2] / 2), 
                (last_track_rect[1] + last_track_rect[3] / 2)]).astype(int)
        arrow_end = xy + 5 * (xy - xy_old)
        mat = cv2.arrowedLine(mat, tuple(xy_old), tuple(arrow_end), (0, 255, 0), thickness=3)
        

        goal_in_img = planner.config_space.config2image_coords(goal[:2])
        init_robot_state_in_img = planner.config_space.config2image_coords(init_robot_state[:2].astype(int))
        mat = cv2.circle(mat, tuple(goal_in_img), radius=15, color=(0, 0, 255), thickness=3)
        mat = cv2.circle(mat, tuple(init_robot_state_in_img), radius=15, color=(0, 255, 255), thickness=3)
        mat = show_cv2_plan(mat, plan, planner)
        
        if t == 0:
            cv2.imwrite("opencv_rrt.png", mat)
        vw.write(mat)
        cv2.imshow("Camera Tracking", mat)

        cv2.waitKey(10)
        
        if SYSTEM_ID_MODE:
            xys.append(xy)
        t += 1
    vw.release()

    if SYSTEM_ID_MODE:
        xys = np.array(xys)
        np.savetxt("./src/proj2_pkg/src/proj2/data/backward_may_4.txt", xys)

if __name__ == '__main__':
    rospy.init_node("main")

    camera_topic = '/usb_cam/image_raw'
    camera_info = '/usb_cam/camera_info'
    camera_frame = '/usb_cam'
    

    goal = np.array([280, 120, 0, 0])
    config = HexbugConfigurationSpace( low_lims = [0, 0, -1000, -1000],
                                        high_lims = [1280, 720, 1000, 1000],
                                         input_low_lims = [-float('inf'), -float('inf')],
                                         input_high_lims = [float('inf'), float('inf')],
                                         obstacles = [],
                                         robot_radius = 10,
                                         primitive_duration = 15)
    planner = RRTPlanner(config, expand_dist=50, max_iter=500)

    online_planning(camera_topic, camera_info, camera_frame, planner, goal)
#!/usr/bin/env python
from cgitb import lookup
from os import system
from random import randrange, sample
from cv2 import contourArea, imshow
import numpy as np
import cv2
from numpy import diff
from utils.utils import * 
import matplotlib.pyplot as plt


import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

system_id_mode = True

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


def plan_and_execute(planner, target_transform, stage_msg): 
    planning_done = False 
    plan = None 
    while not planning_done:
        ps = transform_to_posestamped(target_transform)
        # rospy.logwarn("gripper final target:\n %s" % str(ps))
    
        plan = planner.plan_to_pose(ps)
        if not plan:
            rospy.logwarn("didn't find %s plan, retrying" % stage_msg)
            continue 
        else: 
            planning_done = True 
    planner.execute_plan(plan)


def perform_sys_id(camera_image_topic, camera_info_topic, camera_frame):
    bridge = CvBridge()
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)

    # use frame delta of first 2 frames and get bounding box 
    image_1 = rospy.wait_for_message(camera_image_topic, Image)
    image_1 = bridge.imgmsg_to_cv2(image_1, desired_encoding='passthrough')
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    rospy.logwarn(str(image_1.shape))

    return 
    image_2 = rospy.wait_for_message(camera_image_topic, Image)
    image_2 = bridge.imgmsg_to_cv2(image_2, desired_encoding='passthrough')
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    image_delta = cv2.absdiff(image_1, image_2_gray)
    _, image_delta = cv2.threshold(image_delta, 25, 255, cv2.THRESH_BINARY)
    image_delta = cv2.dilate(image_delta, None, iterations=2)

    # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
    contour_list, hierarchy = cv2.findContours(image_delta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour_list.sort(key=cv2.contourArea)

    largest_contour = contour_list[-1]
    lcbb_x, lcbb_y, lcbb_w, lcbb_h = cv2.boundingRect(largest_contour)


    # KCF object tracking

    tracking_rect = (lcbb_x, lcbb_y, lcbb_w, lcbb_h)
    tracker = cv2.TrackerKCF_create()
    tracker.init(image_2, tracking_rect)

    xys = []
    t = 0 

    while not rospy.is_shutdown() and t < 500: 
        image = rospy.wait_for_message(camera_image_topic, Image)
        mat = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        err_code, tracking_rect = tracker.update(mat)
        
        mat = cv2.rectangle(mat, (int(tracking_rect[0]), int(tracking_rect[1])), (int(tracking_rect[0] + tracking_rect[2]), int(tracking_rect[1] + tracking_rect[3])), 255, thickness=2)
        
        cv2.imshow("Camera Tracking", mat)
        cv2.waitKey(30)

        xy =  np.array([(tracking_rect[0] + tracking_rect[2])/2, 
                        (tracking_rect[1] + tracking_rect[3])/2])

        xys.append(xy)
        t += 1
    plt.figure()
    xys = np.array(xys)
    #np.savetxt("./src/fp_pkg/data/position_data_2.txt", xys)

if __name__ == '__main__':
    rospy.init_node("main")

    camera_topic = '/usb_cam/image_raw'
    camera_info = '/usb_cam/camera_info'
    camera_frame = '/usb_cam'
    
    if system_id_mode:
        # do system id
        perform_sys_id(camera_topic, camera_info, camera_frame)
    else:
        pass
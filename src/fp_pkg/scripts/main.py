#!/usr/bin/env python
from cgitb import lookup
from os import system
from random import randrange
from cv2 import contourArea
import numpy as np
import cv2

try:
    import rospy
    import tf
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, CameraInfo
    from baxter_interface import gripper as baxter_gripper
    from intera_interface import gripper as sawyer_gripper
    ros_enabled = True
except Exception as e:
    print(e)
    print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
    ros_enabled = False


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


def execute_grasp(T_world_grasp, planner, gripper):
    """
    Perform a pick and place procedure for the object. One strategy (which we have
    provided some starter code for) is to
    1. Move the gripper from its starting pose to some distance behind the object
    2. Move the gripper to the grasping pose
    3. Close the gripper
    4. Move up
    5. Place the object somewhere on the table
    6. Open the gripper. 

    As long as your procedure ends up picking up and placing the object somewhere
    else on the table, we consider this a success!

    HINT: We don't require anything fancy for path planning, so using the MoveIt
    API should suffice. Take a look at path_planner.py. The `plan_to_pose` and
    `execute_plan` functions should be useful. If you would like to be fancy,
    you can also explore the `compute_cartesian_path` functionality described in
    http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    
    Parameters
    ----------
    T_world_grasp : 4x4 :obj:`numpy.ndarray`
        pose of gripper relative to world frame when grasping object
    """


    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return

    planner = PathPlanner("right_arm")

    gripper.calibrate()
    # SE(3) transform 


    # get behind object
    calculated_grasp_transform = T_world_grasp.copy()

    r = 0.05 # 5cm behind the object. some small offset
    offset =  np.array([0, 0, r])
    behind_obj_transform = calculated_grasp_transform.copy() 
    behind_obj_transform[:3, 3] += offset
    # behind_obj_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # rospy.logerr("behind_obj_tranform" + str(behind_obj_transform))
    # rospy.logwarn("starting behind plan")
    plan_and_execute(planner, behind_obj_transform, "behind object")

    # get exactly to object/grasp
    
    rospy.logwarn("starting grasp plan")
    plan_and_execute(planner, calculated_grasp_transform, "grasp")
    close_gripper()

    # Bring the object up
    r = 0.10
    offset = np.array([0, 0, r])

    up_air_transform = calculated_grasp_transform.copy() 
    up_air_transform[:3, 3] += offset
    
    rospy.logwarn("starting up air plan")
    plan_and_execute(planner, up_air_transform, "up air")

    
    # Move object in air over
    r = 0.10 
    over_air_transform = up_air_transform.copy()
    rand_r_sample = np.array([0, r, 0])
    over_air_transform[:3, 3] += rand_r_sample
    rospy.logwarn("starting air move plan")
    plan_and_execute(planner, over_air_transform, "air move")

    # Place object back down
    put_down_transform = over_air_transform.copy()
    r = -0.05 
    rand_r_sample = np.array([0, 0, r])
    put_down_transform[:3, 3] += rand_r_sample
    rospy.logwarn("starting put down plan")
    plan_and_execute(planner, put_down_transform, "put down")
    open_gripper()


def perform_sys_id(camera_image_topic, camera_info_topic, camera_frame):
    bridge = CvBridge()
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)

    # use frame delta of first 2 frames and get bounding box 
    image_1 = rospy.wait_for_message(camera_image_topic, Image)
    image_1 = bridge.imgmsg_to_cv2(image_1, desired_encoding='passthrough')
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    image_2 = rospy.wait_for_message(camera_image_topic, Image)
    image_2 = bridge.imgmsg_to_cv2(image_2, desired_encoding='passthrough')
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    image_delta = cv2.absdiff(image_1, image_2)
    image_delta = cv2.threshold(image_delta, 25, 255, cv2.THRESH_BINARY)
    image_delta = cv2.dilate(image_delta, None, iterations=2)
    contour_list = cv2.findContours(image_delta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    contour_list.sort(key=cv2.contourArea)

    largest_contour = contour_list[-1]
    lcbb_x, lcbb_y, lcbb_w, lcbb_h = cv2.boundingRect(largest_contour)
    bb_eps = 5


    tracker = cv2.Tracker_create('KCF')

    tracker_ok = tracker.init(image_2, (lcbb_x, lcbb_y, lcbb_w, lcbb_h))

    # use KCF object tracking 
    while True:
        image = rospy.wait_for_message(camera_image_topic, Image)


        mat = bridge.imgmsg_to_cv2(image_1, desired_encoding='passthrough')

        def localize_cube_in_img(mask):
        hierarchy, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cube_contour = None 
        cube_contour_area = 0 
        for i, contour in enumerate(contours): 
            contour_area = cv2.contourArea(contour)
            if contour_area > cube_contour_area:
                cube_contour = contour 
                cube_contour_area = contour_area
        (cube_x, cube_y), cube_r = cv2.minEnclosingCircle(cube_contour)
        src_harris = cv2.cornerHarris(mask, 2, 3, 0.04)
        src_harris = src_harris < 0.01 * src_harris.min() 
        rospy.logerr(str(src_harris))
        
        func_in_cube = lambda x, y: (x - cube_x)**2 + (y - cube_y)**2 < cube_r ** 2 + 4
        corners = [] 
        for i in range(src_harris.shape[1]):
            for j in range(src_harris.shape[0]):
                if len(corners) == 4: 
                    break
                if func_in_cube(i, j) and src_harris[i, j] > 0: 
                    rospy.logwarn("found corner")
                    corners.append((i, j))
            if len(corners) == 4: 
                break 

        return (cube_x, cube_y), corners

    colored_mat_1 = cv2.bitwise_and(mat_hsv_1, mat_hsv_1, mask=mask_1)
    (square_x, square_y), corners = localize_cube_in_img(mask_1) 
    for c in corners: 
        colored_mat_1 = cv2.circle(colored_mat_1, c, 6, (0, 0, 255), 2)
    colored_mat_1 = cv2.circle(colored_mat_1, (int(square_x), int(square_y)), 6, (0, 255, 0), 2)
    cv2.imshow('image', colored_mat_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # a simplification is that the cube is always lying on the table and cannot be lying on a corner or edge. 
    # so it is always facing up wrt base frame. 
    # therefore we only need to determine its theta, its x-y plane angle 


    # side_length = ... # length of one side of cube
    # pose = ... # 4x4 homogenous transform for center of cube
    # return trimesh.primitives.Box((side_length, side_length, side_length), pose)


if __name__ == '__main__':

    robot_type = 'sawyer'
    if robot_type == 'baxter':
        camera_topic = '/cameras/left_hand_camera/image'
        camera_info = '/cameras/left_hand_camera/camera_info'
        camera_frame = '/left_hand_camera'
    elif robot_type == 'sawyer':
        camera_topic = '/usb_cam/image_raw'
        camera_info = '/usb_cam/camera_info'
        camera_frame = '/usb_cam'
        
    system_id_mode = True
    if system_id_mode:
        # do system id
        mesh = perform_sys_id(camera_topic, camera_info, camera_frame)
    else:
        pass 
    



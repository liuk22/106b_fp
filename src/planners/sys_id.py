#!/usr/bin/env python
import numpy as np
import cv2
from utils import * 
import matplotlib.pyplot as plt
from rrt_planner import RRTPlanner
from configuration_space import HexbugConfigurationSpace
import time

SYSTEM_ID_MODE = False
GLOBAL_DT = 0.2

def localize_robot_initial_track_rect(vc):
    # use frame delta of first 2 frames and get bounding box 
    ret_1, image_1 = vc.read()
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    ret_2, image_2 = vc.read()
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
    visited_p = set() 

    if plan:
        for t, p, c in plan:
            center = tuple(planner.config_space.config2image_coords(p[:2]))
            mat = cv2.circle(mat, center=center, radius=2, color=(0, 165, 255), thickness=2)
            visited_p.add(str(p)) 

    for path in planner.graph.get_edge_paths():
        for t, p, c in path:
            if str(p) not in visited_p:
                center = tuple(planner.config_space.config2image_coords(p[:2]))
                mat = cv2.circle(mat, center=center, radius=2, color=(203, 192, 255), thickness=2)
            
    return mat 

def online_planning(planner, goal):
    video_capture = cv2.VideoCapture(1)
    vw = cv2.VideoWriter('output_1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 5, (640, 480))
    tracking_rect, image_2 =  localize_robot_initial_track_rect(video_capture)

    # KCF object tracking, get the initial theta state of robot needs two timestep approximation
    tracker = cv2.TrackerKCF_create()
    tracker.init(image_2, tracking_rect)
    last_track_rect = tracking_rect
    for _ in range(10):
        ret, mat = video_capture.read()
    err_code, tracking_rect = tracker.update(mat)
    init_robot_state = two_rects_to_state(tracking_rect, last_track_rect)
    replan_timestep_horizon = 20
    if not SYSTEM_ID_MODE:
        t1 = time.time()
        plan = planner.plan_to_pose(tuple(init_robot_state), goal)
        print(f"planning with {planner.max_iter} max iterations took { time.time() - t1} seconds")
        if plan: 
            planner.execute_plan(plan, replan_timestep_horizon)
        else:
            print("couldn't find plan")
    xys = []
    t = 0 

    while True: # and t < 10 * total_time: 
        ret, mat = video_capture.read()
        last_track_rect = tracking_rect
        err_code, tracking_rect = tracker.update(mat)
        if t > 1:
            init_robot_state = two_rects_to_state(tracking_rect, last_track_rect)
        if t > 75:
            break 
        if t % replan_timestep_horizon == 0:
            tracking_rect, image_2 = localize_robot_initial_track_rect(video_capture)
            tracker = cv2.TrackerKCF_create()
            tracker.init(image_2, tracking_rect)
            last_track_rect = tracking_rect
            plan = planner.plan_to_pose(tuple(init_robot_state), goal, replan_timestep_horizon)
            if plan:
                planner.execute_plan(plan, replan_timestep_horizon)
            else:
                print("couldn't find plan")
        
        mat = cv2.rectangle(mat, (int(tracking_rect[0]), int(tracking_rect[1])), (int(tracking_rect[0] + tracking_rect[2]), int(tracking_rect[1] + tracking_rect[3])), 255, thickness=2)
        xy =  np.array([(tracking_rect[0] + tracking_rect[2] / 2), 
                        (tracking_rect[1] + tracking_rect[3] / 2)]).astype(int)
        xy_old =  np.array([(last_track_rect[0] + last_track_rect[2] / 2), 
                (last_track_rect[1] + last_track_rect[3] / 2)]).astype(int)
        arrow_end = xy + 10 * (xy - xy_old)
        mat = cv2.arrowedLine(mat, tuple(xy_old), tuple(arrow_end), (0, 255, 0), thickness=3)
        

        goal_in_img = planner.config_space.config2image_coords(goal[:2])
        init_robot_state_in_img = planner.config_space.config2image_coords(init_robot_state[:2].astype(int))
        mat = cv2.circle(mat, tuple(goal_in_img), radius=15, color=(0, 0, 255), thickness=3)
        arrow_end = goal_in_img + 10 * np.array([np.cos(goal[2]), np.sin(goal[2])])
        arrow_end = arrow_end.astype(int)
        mat = cv2.arrowedLine(mat, tuple(goal_in_img), tuple(arrow_end), (255, 255, 0), thickness=3)
        mat = cv2.circle(mat, tuple(init_robot_state_in_img), radius=15, color=(0, 255, 255), thickness=3)
        mat = show_cv2_plan(mat, plan, planner)
        
        if t == 2:
            cv2.imwrite("timestep_2.png", mat)
        if  t % replan_timestep_horizon == 0 and plan:
            cv2.imwrite(f"timestep_{t}.png", mat)
        vw.write(mat)

        cv2.imshow("Camera Tracking", mat)

        cv2.waitKey(50)
        
        if SYSTEM_ID_MODE:
            xys.append(xy)
        t += 1
    vw.release()

    if SYSTEM_ID_MODE:
        xys = np.array(xys)
        np.savetxt("./src/proj2_pkg/src/proj2/data/backward_may_4.txt", xys)

if __name__ == '__main__':
    goal = np.array([150, 120, np.pi, 0])
    config = HexbugConfigurationSpace( low_lims = [0, 0, -1000, -1000],
                                        high_lims = [1280, 720, 1000, 1000],
                                         input_low_lims = [-float('inf'), -float('inf')],
                                         input_high_lims = [float('inf'), float('inf')],
                                         obstacles = [],
                                         robot_radius = 10,
                                         primitive_duration = 15,
                                         goal_bias=0.25)
    planner = RRTPlanner(config, expand_dist=50, max_iter=100)

    online_planning(planner, goal)
#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point
import time


rospy.init_node('move_obstacles_node')


set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

obstacle1_name = 'obstacle_3'
obstacle2_name = 'obstacle_9'
obstacle3_name = 'obstacle_6'

rate = rospy.Rate(10)
start_time = time.time()

while not rospy.is_shutdown():
    current_time = time.time() - start_time

    position_obstacle1 = Point()
    if 24.0 > current_time > 13.0:
        position_obstacle1.x = 2.25 + 0.06 * (current_time - 13.0)
        position_obstacle1.y = 2.25 - 0.12 * (current_time - 13.0)
    elif current_time >= 24.0:
        position_obstacle1.x = 2.25 + 0.06 * (24.0 - 13.0)
        position_obstacle1.y = 2.25 - 0.12 * (24.0 - 13.0)
    else:
        position_obstacle1.x = 2.25
        position_obstacle1.y = 2.25

    position_obstacle2 = Point()
    if 40 > current_time > 13.0:
        position_obstacle2.x = 4.25 - 0.06 * (current_time - 13.0)
        position_obstacle2.y = 2.45 + 0.06 * (current_time - 13.0)
    elif current_time >= 40.0:
        position_obstacle2.x = 4.25 - 0.06 * (40.0 - 13.0)
        position_obstacle2.y = 2.45 + 0.06 * (40.0 - 13.0)
    else:
        position_obstacle2.x = 4.25
        position_obstacle2.y = 2.45

    position_obstacle3 = Point()
    if 46.0 > current_time > 27:
        position_obstacle3.x = 4.75 - 0.1 * (current_time - 27)
        position_obstacle3.y = 3.75 
    elif current_time >= 45.0:
        position_obstacle3.x = 4.75 - 0.1 * (46.0 - 27)
        position_obstacle3.y = 3.75
    else:
        position_obstacle3.x = 4.75
        position_obstacle3.y = 3.75

    model_state_obstacle1 = ModelState()
    model_state_obstacle1.model_name = obstacle1_name
    model_state_obstacle1.pose.position = position_obstacle1

    model_state_obstacle2 = ModelState()
    model_state_obstacle2.model_name = obstacle2_name
    model_state_obstacle2.pose.position = position_obstacle2

    model_state_obstacle3 = ModelState()
    model_state_obstacle3.model_name = obstacle3_name
    model_state_obstacle3.pose.position = position_obstacle3

    set_model_state(model_state_obstacle1)
    set_model_state(model_state_obstacle2)
    set_model_state(model_state_obstacle3)

    rate.sleep()
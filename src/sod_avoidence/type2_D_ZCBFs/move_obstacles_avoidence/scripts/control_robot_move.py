#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import quaternion_from_euler
import time
import math
import numpy as np
from scipy.interpolate import CubicSpline



def interpolate(start, end, ratio):
    return start + ratio * (end - start)

def interpolate_pose(start_pos, end_pos, start_ori_euler, end_ori_euler, ratio):
    current_pos = Point(
        interpolate(start_pos.x, end_pos.x, ratio),
        interpolate(start_pos.y, end_pos.y, ratio),
        interpolate(start_pos.z, end_pos.z, ratio)
    )
    current_ori_euler = (
        interpolate(start_ori_euler[0], end_ori_euler[0], ratio),
        interpolate(start_ori_euler[1], end_ori_euler[1], ratio),
        interpolate(start_ori_euler[2], end_ori_euler[2], ratio)
    )
    current_ori = Quaternion(*quaternion_from_euler(*current_ori_euler))
    return current_pos, current_ori

def move_robot(robot_name, waypoints, current_time):
    for i in range(len(waypoints) - 1):
        if waypoints[i]['time'] <= current_time < waypoints[i + 1]['time']:
            start_pos = waypoints[i]['position']
            end_pos = waypoints[i + 1]['position']
            start_ori_euler = waypoints[i]['orientation_euler']
            end_ori_euler = waypoints[i + 1]['orientation_euler']

            duration = waypoints[i + 1]['time'] - waypoints[i]['time']
            elapsed = current_time - waypoints[i]['time']
            ratio = elapsed / duration

            current_pos, current_ori = interpolate_pose(start_pos, end_pos, start_ori_euler, end_ori_euler, ratio)

            model_state = ModelState()
            model_state.model_name = robot_name
            model_state.pose.position = current_pos
            model_state.pose.orientation = current_ori

            try:
                set_model_state(model_state)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed for {robot_name}: {e}")
            break

def generate_spline_path(start, end, start_orientation_euler, end_orientation_euler, mid_point, mid_orientation_euler, mid_time, num_points, start_time, total_duration):
    times = [start_time, mid_time, start_time + total_duration]
    points = [start, mid_point, end]

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    spline_x = CubicSpline(times, x_vals)
    spline_y = CubicSpline(times, y_vals)

    bezier_path = []
    for i in range(num_points):
        t = start_time + i * total_duration / (num_points - 1)
        x = spline_x(t)
        y = spline_y(t)

        if i == 0:
            orientation_euler = start_orientation_euler
        elif i == num_points - 1:
            orientation_euler = end_orientation_euler
        else:
            dx_dt = spline_x(t, 1)
            dy_dt = spline_y(t, 1)
            orientation_euler = (0, 0, math.atan2(dy_dt, dx_dt))

        orientation_quat = Quaternion(*quaternion_from_euler(*orientation_euler))
        bezier_path.append({
            'time': t,
            'position': Point(x, y, 0),
            'orientation': orientation_quat,
            'orientation_euler': orientation_euler
        })
    return bezier_path

rospy.init_node('move_robots_node')
rospy.wait_for_service('/gazebo/set_model_state')
set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

start_point9 = (-4, 2.6)
start_orientation9 = (0, 0, 0)  

mid_point9 = (0, 3.0)
mid_orientation_euler9 = (0, 0.0, -math.pi /3)
mid_time9 = 17.5

end_point9 = (2, 5.0)
end_orientation9 = (0, 0, 0)  

waypoints_robot9 = generate_spline_path(
    start_point9, end_point9, start_orientation9, end_orientation9,
    mid_point9, mid_orientation_euler9, mid_time9,
    num_points=80, start_time=0, total_duration=52
)

start_point7 = (-4, 0.6)
start_orientation7 = (0, 0, 0) 

mid_point7 = (-0.5, 2.0)
mid_orientation_euler7 = (0, 0, math.pi / 4)
mid_time7 = 22.0

end_point7 = (2, 4.0)
end_orientation7 = (0, 0, 0)

waypoints_robot7 = generate_spline_path(
    start_point7, end_point7, start_orientation7, end_orientation7,
    mid_point7, mid_orientation_euler7, mid_time7,
    num_points=80, start_time=0, total_duration=51.6
)


start_point6 = (4, -1.5)
start_orientation6 = (0, 0, -math.pi) 

mid_point6 = (-0.4, 1.1)
mid_orientation_euler6 = (0, 0, -math.pi *2/3)
mid_time6 = 25

end_point6 = (-2, 4.0)
end_orientation6 = (0, 0, -math.pi)  

waypoints_robot6 = generate_spline_path(
    start_point6, end_point6, start_orientation6, end_orientation6,
    mid_point6, mid_orientation_euler6, mid_time6,
    num_points=80, start_time=0, total_duration=58
)

start_point8 = (4, 0.5)
start_orientation8 = (0, 0, math.pi) 

mid_point8 = (1, 0.0)
mid_orientation_euler8 = (0, 0, math.pi *3/4)
mid_time8 = 19.0

end_point8 = (-2, 3.0)
end_orientation8 = (0, 0, -math.pi)  

waypoints_robot8 = generate_spline_path(
    start_point8, end_point8, start_orientation8, end_orientation8,
    mid_point8, mid_orientation_euler8, mid_time8,
    num_points=80, start_time=0, total_duration=52
)

start_point5 = (4, -3.5)
start_orientation5 = (0, 0, -math.pi)  

mid_point5 = (0.5, -0.7)
mid_orientation_euler5 = (0, 0, -math.pi*2/ 3)
mid_time5 = 31.5

end_point5 = (-2, 1.6)
end_orientation5 = (0, 0, -math.pi)  

waypoints_robot5 = generate_spline_path(
    start_point5, end_point5, start_orientation5, end_orientation5,
    mid_point5, mid_orientation_euler5, mid_time5,
    num_points=80, start_time=8, total_duration=59.0
)


start_point2 = (-4, -1.4)
start_orientation2 = (0, 0, 0)  

mid_point2 = (0, -0.8)
mid_orientation_euler2 = (0, 0, math.pi / 5)
mid_time2 = 41.5

end_point2 = (2, 3.0)

end_orientation2 = (0, 0, 0)  

waypoints_robot2 = generate_spline_path(
    start_point2, end_point2, start_orientation2, end_orientation2,
    mid_point2, mid_orientation_euler2, mid_time2,
    num_points=80, start_time=10, total_duration=63
)


start_point4 = (-2, -5.0)
start_orientation4 = (0, 0, -math.pi/2)  

mid_point4 = (0, -2.0)
mid_orientation_euler4 = (0, 0, math.pi / 4)
mid_time4 = 46



end_point4 = (2, 2.0)
end_orientation4 = (0, 0, 0)  

waypoints_robot4 = generate_spline_path(
    start_point4, end_point4, start_orientation4, end_orientation4,
    mid_point4, mid_orientation_euler4, mid_time4,
    num_points=80, start_time=20, total_duration=85
    # num_points=80, start_time=0, total_duration=10

    
)


start_point1 = (0, -5.0)
start_orientation1 = (0, 0, math.pi/2) 

mid_point1 = (1, -2.5)
mid_orientation_euler1 = (0, 0, math.pi / 3)
mid_time1 = 85

end_point1 = (2, 1.0)
end_orientation1 = (0, 0, 0)  

waypoints_robot1 = generate_spline_path(
    start_point1, end_point1, start_orientation1, end_orientation1,
    mid_point1, mid_orientation_euler1, mid_time1,
    num_points=80, start_time=60, total_duration=105
)


start_point3 = (2, -5.0)
start_orientation3 = (0, 0, math.pi/2)  

mid_point3 = (-0.0, -2.3)
mid_orientation_euler3 = (0, 0, math.pi *3/ 4)
mid_time3 = 50

end_point3 = (-2, -1.0)
end_orientation3 = (0, 0, -math.pi)  

waypoints_robot3 = generate_spline_path(
    start_point3, end_point3, start_orientation3, end_orientation3,
    mid_point3, mid_orientation_euler3, mid_time3,
    num_points=80, start_time=30, total_duration=68
)
rate = rospy.Rate(10)
start_time = time.time()

while not rospy.is_shutdown():
    current_time = time.time() - start_time

    move_robot('turtlebot3_9', waypoints_robot9, current_time)
    move_robot('turtlebot3_7', waypoints_robot7, current_time)
    move_robot('turtlebot3_8', waypoints_robot8, current_time)
    move_robot('turtlebot3_6', waypoints_robot6, current_time)
    move_robot('turtlebot3_5', waypoints_robot5, current_time)
    move_robot('turtlebot3_2', waypoints_robot2, current_time)
    move_robot('turtlebot3_4', waypoints_robot4, current_time)
    move_robot('turtlebot3_1', waypoints_robot1, current_time)
    move_robot('turtlebot3_3', waypoints_robot3, current_time)






    rate.sleep()

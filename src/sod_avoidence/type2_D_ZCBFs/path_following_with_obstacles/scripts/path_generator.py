#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import quaternion_from_euler

def generate_sine_trajectory(start_x, end_x, num_points, amplitude, frequency):
    x = np.linspace(start_x, end_x, num_points)
    y = amplitude * np.sin(frequency * x)
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)
    theta = np.append(theta, theta[-1])
    return [(x[i], y[i], theta[i]) for i in range(num_points)]

def generate_elliptical_trajectory(a, b, num_points):
    # Generate points for a full ellipse
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    x = a * np.cos(t)
    y = b * np.sin(t)
    # Calculate the heading angle (theta)
    dx = -a * np.sin(t)
    dy = b * np.cos(t)
    theta = np.arctan2(dy, dx)
    
    # Shift the starting point to the bottom of the ellipse
    start_index = np.argmin(y)
    x = np.roll(x, -start_index)
    y = np.roll(y, -start_index)
    theta = np.roll(theta, -start_index)
    
    # Ensure the first point's angle is 0 (facing right)
    theta = (theta - theta[0] + 2*np.pi) % (2*np.pi)
    trajectory = [(x[i], y[i], theta[i]) for i in range(len(x))]
    return trajectory

def path_publisher():
    rospy.init_node('path_publisher', anonymous=True)
    path_pub = rospy.Publisher('trajectory', PoseArray, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # # Generate sin trajectory
    trajectory = generate_sine_trajectory(0.0, 10.0, 100, 2, 1)
    # Generate the elliptical trajectory
    # Parameters: a (semi-major axis), b (semi-minor axis), number of points
    # trajectory = generate_elliptical_trajectory(4.0, 2.5, 100)

    # Create PoseArray message
    pose_array = PoseArray()
    pose_array.header.frame_id = "odom"  
    for point in trajectory:
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        q = quaternion_from_euler(0, 0, point[2])
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        pose_array.poses.append(pose)

    while not rospy.is_shutdown():
        pose_array.header.stamp = rospy.Time.now()
        path_pub.publish(pose_array)
        rate.sleep()

if __name__ == '__main__':
    try:
        path_publisher()
    except rospy.ROSInterruptException:
        pass
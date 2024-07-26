#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

def trajectory_callback(msg):
    global marker_pub
    
    # Create a line strip marker for the path
    line_strip = Marker()
    line_strip.header = msg.header
    line_strip.ns = "path_line"
    line_strip.id = 0
    line_strip.type = Marker.LINE_STRIP
    line_strip.action = Marker.ADD
    line_strip.scale.x = 0.02  # Line width
    line_strip.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # Green color
    line_strip.pose.orientation.w = 1.0
    
    # Add all points to the line strip
    for pose in msg.poses:
        line_strip.points.append(pose.position)
    
    # Publish the line strip marker
    marker_pub.publish(line_strip)

if __name__ == '__main__':
    rospy.init_node('path_visualizer')
    
    # Subscribe to the trajectory topic
    rospy.Subscriber('/trajectory', PoseArray, trajectory_callback)
    
    # Create a publisher for the marker
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    
    rospy.spin()
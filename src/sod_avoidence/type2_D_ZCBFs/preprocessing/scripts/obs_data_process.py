#!/usr/bin/env python3
import re
import numpy as np
import rospy
import json
from std_msgs.msg import String

def extract_and_format_data(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    formatted_data = []
    for i in range(0, len(numbers), 3):
        obstacle_data = [round(float(numbers[i]), 2), round(float(numbers[i+1]), 2),
                         round(float(numbers[i+2]), 2), 0.0, 0.0]
        formatted_data.append(obstacle_data)
    return formatted_data

def obstacle_info_callback(data):
    global obstacle_tracks, timestamps
    current_time = rospy.get_time()
    if data.data:
        formatted_list = extract_and_format_data(data.data)
    else:
        formatted_list = []
    update_obstacle_history(formatted_list, current_time)

def update_obstacle_history(current_scan, current_time):
    global obstacle_tracks, timestamps
    valid_indices = [i for i in range(len(obstacle_tracks)) if (current_time - timestamps[i][-1]) < DECAY_TIME_LIMIT]
    obstacle_tracks = [obstacle_tracks[i] for i in valid_indices]
    timestamps = [timestamps[i] for i in valid_indices]

    if current_scan:
        if not obstacle_tracks:
            obstacle_tracks = [[obstacle] for obstacle in current_scan]
            timestamps = [[current_time] for _ in current_scan]
            return

        matched_current_scan = set()

        for i in range(len(obstacle_tracks)):
            track = obstacle_tracks[i]
            last_obstacle = track[-1]

            for j in range(len(current_scan)):
                if j in matched_current_scan:
                    continue
                current_obstacle = current_scan[j]
                distance = np.linalg.norm(np.array(last_obstacle[:2]) - np.array(current_obstacle[:2]))
                if distance < 0.2:
                    matched_current_scan.add(j)
                    track.append(current_obstacle)
                    timestamps[i].append(current_time)

                    if len(track) > 15:
                        track.pop(0)
                        timestamps[i].pop(0)
                    break

        for j, obstacle in enumerate(current_scan):
            if j not in matched_current_scan:
                obstacle_tracks.append([obstacle])
                timestamps.append([current_time])

    for i in range(len(obstacle_tracks)):
        track = obstacle_tracks[i]
        time_list = timestamps[i]
        obs_vx, obs_vy = calculate_velocity(track, time_list)
        track[-1][3] = obs_vx
        track[-1][4] = obs_vy

    publish_all_latest_obstacles()

def initialize_kalman():
    x = np.array([0, 0])  
    P = np.eye(2)  
    F = np.eye(2)  
    H = np.array([[1, 0]])  
    # R = np.array([[20.0]])  
    # R = np.array([[1.25]])  
    R = np.array([[1.0]])  
    Q = np.array([[1e-4, 0], [0, 1e-4]])  
    return x, P, F, H, R, Q

def kalman_filter_last_velocity(positions, obstime):
    x, P, F, H, R, Q = initialize_kalman()
    for i in range(1, len(positions)):
        dt = obstime[i] - obstime[i - 1]
        F[0, 1] = dt
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        z = np.array([positions[i]])
        y = z - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(2) - K.dot(H)).dot(P)
    
    return x[1]

def calculate_velocity(track, obstime):
    if len(obstime) <= 1:
        return 0.0, 0.0  
    else:
        x_positions = np.array([pos[0] for pos in track])
        x_positions = x_positions - x_positions[0]
        y_positions = np.array([pos[1] for pos in track])
        y_positions = y_positions - y_positions[0]
        vx = kalman_filter_last_velocity(x_positions, obstime)
        vy = kalman_filter_last_velocity(y_positions, obstime)
        if abs(vx) < 0.01:
            vx = 0
        if abs(vy) < 0.01:
            vy = 0
        return vx, vy

def publish_all_latest_obstacles():
    global obstacle_tracks
    latest_obstacles = [track[-1] for track in obstacle_tracks]
    obstacle_data = json.dumps(latest_obstacles)
    pub.publish(String(obstacle_data))


obstacle_tracks = []
timestamps = []
# DECAY_TIME_LIMIT = 3.0
DECAY_TIME_LIMIT = 2.0 #sin path / elliptical path



rospy.init_node("obstacle_processor")
sub = rospy.Subscriber("/obstacle_info", String, obstacle_info_callback)
pub = rospy.Publisher("/processed_obstacles", String, queue_size=1)

rospy.spin()
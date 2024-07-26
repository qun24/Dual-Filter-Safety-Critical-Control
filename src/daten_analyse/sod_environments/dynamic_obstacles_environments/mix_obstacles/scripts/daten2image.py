import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from cycler import cycler
import numpy as np
import os

# 共享配置
FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/dynamic_obstacles_environments'



def load_json_data(filename):
    file_path = os.path.join(FOLDER_PATH, filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        print(f"错误：在指定的文件夹中没有找到 {filename} 文件。")
        return None

def save_plot(plt, filename):
    save_path = os.path.join(FOLDER_PATH, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
def generate_trajectory_plot(target_points, obstacles):
    data = load_json_data('xy_path_sod.json')
    if data is None:
        return

    x_path = data["x_path"]
    y_path = data["y_path"]
    T = data["T"][:-1]  # 从T中删除最后一个元素

    assert len(x_path) == len(T) == len(y_path), "所有数据长度必须相同"

    plt.figure(figsize=(14, 10))
    
    # 绘制机器人轨迹
    plt.plot(x_path, y_path, color='gold', label='Robot Trajectory', linewidth=4)
    
    # 绘制目标点
    for i, point in enumerate(target_points, 1):
        plt.plot(point[0], point[1], marker='*', color='green', markersize=35)
        plt.annotate(f'$\\it{{p_{i}}}$', 
                    (point[0], point[1]), 
                    xytext=(10, 10),
                    textcoords='offset points', 
                    fontsize=20, 
                    color='black')
    
    # 定义动态障碍物的移动函数
    def get_obstacle_position(current_time, start_pos, end_pos, start_time, end_time):
        if current_time < start_time:
            return start_pos
        elif current_time > end_time:
            return end_pos
        else:
            t = (current_time - start_time) / (end_time - start_time)
            return (
                start_pos[0] + t * (end_pos[0] - start_pos[0]),
                start_pos[1] + t * (end_pos[1] - start_pos[1])
            )

    # 定义动态障碍物
    dynamic_obstacles = [
        ((2.25, 2.25), (2.25 + 0.06 * 11, 2.25 - 0.12 * 11), 13.0, 24.0),
        ((4.25, 2.45), (4.25 - 0.06 * 27, 2.45 + 0.06 * 27), 13.0, 40.0),
        ((4.75, 3.75), (4.75 - 0.1 * 19, 3.75), 27.0, 46.0)
    ]

    # 绘制静态障碍物和动态障碍物的起点和终点
    for obstacle in obstacles:
        if obstacle[:2] not in [obs[0] for obs in dynamic_obstacles]:
            # 静态障碍物
            solid_circle = Circle(obstacle[:2], obstacle[2], color="black", linewidth=4)
            plt.gca().add_patch(solid_circle)
            dashed_circle_safe = Circle(obstacle[:2], 0.37, color="orange", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)
            dashed_circle_safe = Circle(obstacle[:2], 0.4, color="black", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)
            dashed_circle_safe = Circle(obstacle[:2], 0.7, color="purple", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)
        else:
            # 动态障碍物的起点和终点
            start_pos, end_pos, _, _ = next(obs for obs in dynamic_obstacles if obs[0] == obstacle[:2])
            for pos in [start_pos, end_pos]:
                solid_circle = Circle(pos, obstacle[2], color="red", linewidth=4, alpha=0.5)
                plt.gca().add_patch(solid_circle)

    # 找到动态障碍物轨迹上距离机器人轨迹最近的点
    for start_pos, end_pos, start_time, end_time in dynamic_obstacles:
        min_distance = float('inf')
        closest_point = None
        closest_robot_point = None
        closest_time = None
        for t, x, y in zip(T, x_path, y_path):
            obs_pos = get_obstacle_position(t, start_pos, end_pos, start_time, end_time)
            distance = ((x - obs_pos[0])**2 + (y - obs_pos[1])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_point = obs_pos
                closest_robot_point = (x, y)
                closest_time = t

        # 在最近点处绘制红色圆
        if closest_point:
            solid_circle = Circle(closest_point, 0.1, color="red", linewidth=4)
            plt.gca().add_patch(solid_circle)
            dashed_circle_safe = Circle(closest_point, 0.37, color="orange", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)
            dashed_circle_safe = Circle(closest_point, 0.4, color="black", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)
            dashed_circle_safe = Circle(closest_point, 0.7, color="purple", fill=False, linestyle='--', linewidth=4)
            plt.gca().add_patch(dashed_circle_safe)

            # 绘制简化的小车模型在机器人轨迹上
            car_size = 0.2
            car_angle = np.arctan2(y_path[T.index(closest_time)] - y_path[T.index(closest_time)-1],
                                   x_path[T.index(closest_time)] - x_path[T.index(closest_time)-1])
            car_rect = plt.Rectangle((closest_robot_point[0] - car_size/2, closest_robot_point[1] - car_size/2),
                                     car_size, car_size, angle=np.degrees(car_angle),
                                     facecolor='blue', edgecolor='black')
            plt.gca().add_patch(car_rect)
            # 添加车轮
            wheel_size = 0.05
            for wheel_pos in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                wheel_x = closest_robot_point[0] + wheel_pos[0] * car_size/3 * np.cos(car_angle) - wheel_pos[1] * car_size/3 * np.sin(car_angle)
                wheel_y = closest_robot_point[1] + wheel_pos[0] * car_size/3 * np.sin(car_angle) + wheel_pos[1] * car_size/3 * np.cos(car_angle)
                wheel = plt.Circle((wheel_x, wheel_y), wheel_size, facecolor='black')
                plt.gca().add_patch(wheel)

        # 绘制动态障碍物的轨迹
        trajectory = [get_obstacle_position(t, start_pos, end_pos, start_time, end_time) for t in np.linspace(start_time, end_time, 100)]
        plt.plot(*zip(*trajectory), color='gray', linestyle='--', linewidth=2)

    # 设置坐标轴范围
    all_x = x_path + [obs[0] for obs in obstacles] + [pos[0] for obs in dynamic_obstacles for pos in obs[:2]]
    all_y = y_path + [obs[1] for obs in obstacles] + [pos[1] for obs in dynamic_obstacles for pos in obs[:2]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    
    plt.xlabel('x position [m]', fontsize=23)
    plt.ylabel('y position [m]', fontsize=23)
    plt.title('Trajectories', fontsize=24)
    plt.xticks(np.arange(int(x_min), int(x_max)+2, 1), fontsize=18)
    plt.yticks(np.arange(int(y_min), int(y_max)+1, 1), fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    save_plot(plt, 'trajectories.png')
    plt.close()

def generate_time_analysis_plot():
    mpc_data = load_json_data('time_analysis.json')
    saliency_data = load_json_data('saliency_detection_statistics.json')
    
    if mpc_data is None or saliency_data is None:
        return

    T = mpc_data['T'][:-1]
    mpc_cbf_times = mpc_data['mpc_cbf_execution_times'][:-1]
    mpc_avg_time = mpc_data['average_execution_time']
    
    saliency_avg_time = saliency_data['average_detection_time_ms'] / 1000  # Convert to seconds
    saliency_max_time = saliency_data['max_detection_time_ms'] / 1000
    saliency_min_time = saliency_data['min_detection_time_ms'] / 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # MPC-CBF Execution Time Plot
    ax1.plot(T, mpc_cbf_times, label='MPC-CBF Execution Time', color='orange', linewidth=4, alpha=0.7)
    ax1.axhline(y=mpc_avg_time, color='r', linestyle='--', label=f'MPC-CBF Avg: {mpc_avg_time:.3f}s')
    ax1.set_ylabel('Execution Time [s]', fontsize=20)
    ax1.set_title('MPC-CBF Loop', fontsize=20)
    ax1.legend(fontsize=18)
    ax1.grid(True, alpha=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=17)  # Increase tick label size

    # Saliency Detection Time Plot
    ax2.axhline(y=saliency_avg_time, color='g', linestyle='--', label=f'Saliency Avg: {saliency_avg_time:.3f}s')
    ax2.axhline(y=saliency_max_time, color='r', linestyle=':', label=f'Saliency Max: {saliency_max_time:.3f}s')
    ax2.axhline(y=saliency_min_time, color='b', linestyle=':', label=f'Saliency Min: {saliency_min_time:.3f}s')
    ax2.set_ylim(0, saliency_max_time * 1.1)  # Set y-axis limit to slightly above max time
    ax2.set_xlabel('time (s)', fontsize=20)
    ax2.set_ylabel('Detection Time [s]', fontsize=20)
    ax2.set_title('Saliency Detection Time', fontsize=20)
    ax2.legend(fontsize=18)
    ax2.grid(True, alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=17)  # Increase tick label size

    plt.tight_layout()
    save_plot(plt, 'comprehensive_time_analysis_plot.png')
    plt.close()

def generate_cost_plot():
    data = load_json_data('cost_values.json')
    if data is None:
        return

    cost = data["cost_values"]
    T = data["T"][:-1]

    assert len(cost) == len(T)

    plt.figure(figsize=(14, 7))
    plt.plot(T, cost, color='blue', label='cost', linewidth=4)

    plt.xlabel('time [s]', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.title('Cost', fontsize=22)

    # Set x-axis ticks and limits
    x_max = max(T)
    x_ticks = range(0, int(x_max) + 6, 10)
    plt.xticks(x_ticks, fontsize=18)
    plt.xlim(0, x_ticks[-1])

    # Set y-axis ticks and limits
    y_max = max(cost)
    y_ticks = np.arange(0, y_max + 1001, 2000.0)
    plt.yticks(y_ticks, fontsize=18)
    plt.ylim(0, y_ticks[-1])

    plt.legend(fontsize=18)
    plt.grid(True)

    plt.tight_layout()
    save_plot(plt, 'cost.png')
    plt.close()

def generate_distance_to_goal_plot(target_points):
    data = load_json_data('xy_path_sod.json')
    
    if not data:
        return

    plt.figure(figsize=(14, 7))

    max_time = 0
    max_distance = 0
    all_target_times = set()

    x_path = data["x_path"]
    y_path = data["y_path"]
    theta_path = data["theta_path"]
    T = data["T"][:-1]

    distance = []
    target_index = 0
    target_times = []

    for x, y, theta, t in zip(x_path, y_path, theta_path, T):
        x_idk = np.array([x, y, theta])
        xs = np.array(target_points[target_index])
        
        current_distance = np.linalg.norm(x_idk - xs)
        dist_to_goal = np.sqrt((x - target_points[target_index][0])**2 + (y - target_points[target_index][1])**2)
        distance.append(dist_to_goal)
        
        if current_distance < 0.05 and target_index + 1 < len(target_points):
            target_index += 1
            target_times.append(t)
    
    # Add the final time point to target_times
    target_times.append(T[-1])
    
    plt.plot(T[:len(distance)], distance, color='gold', label='SOD-MPC-CBF', linewidth=4)
    
    max_time = max(T)
    max_distance = max(distance)

    # Add vertical lines for target points including the final point
    for t in target_times:
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=2)
        all_target_times.add(t)

    # Set axis limits and ticks
    plt.xlim(0, math.ceil(max_time))
    plt.ylim(0, math.ceil(max_distance))
    plt.xticks(range(0, math.ceil(max_time) + 1, 10), fontsize=18)
    plt.yticks(range(0, math.ceil(max_distance) + 1), fontsize=18)

    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('distance [m]', fontsize=20)
    plt.title('Distance to Goal', fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.tight_layout()
    save_plot(plt, 'distance_to_goal.png')
    plt.close()

def generate_control_inputs_plot():
    data = load_json_data('input.json')
    if data is None:
        return

    v_path = data["v"]
    w_path = data["w"]
    T = data["T"][:-1]

    assert len(v_path) == len(T) and len(w_path) == len(T), "Length of v_path, w_path, and T do not match."

    plt.figure(figsize=(14, 7))
    plt.plot(T, v_path, color='#800080', label='Linear velocity', linewidth=2)  
    plt.plot(T, w_path, color='#D2691E', label='Angular velocity', linewidth=2) 
     
    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('control inputs', fontsize=20)
    
    x_max = max(T)
    x_ticks = range(0, int(x_max) + 6, 10)
    plt.xticks(x_ticks, fontsize=18) 
    plt.xlim(0, x_ticks[-1])
    plt.yticks(fontsize=18)
    plt.axhline(y=0.2, color='#800080', linestyle='--', linewidth=4)  
    plt.axhline(y=-0.2, color='#800080', linestyle='--', linewidth=4)  
    plt.axhline(y=0.4, color='#D2691E', linestyle='--', linewidth=4)  
    plt.axhline(y=-0.4, color='#D2691E', linestyle='--', linewidth=4)  

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_plot(plt, 'control_inputs.png')
    plt.close()

def generate_obstacle_distance_plot(obstacles):
    data = load_json_data('xy_path_sod.json')
    
    if data is None:
        return
    
    plt.figure(figsize=(14, 7))
    
    x_path = data["x_path"]
    y_path = data["y_path"]
    T = data["T"][:-1]
    
    obs_dist = []
    for x, y in zip(x_path, y_path):
        distances = [np.sqrt((x - obs[0])**2 + (y - obs[1])**2) - obs[2] - 0.15 for obs in obstacles]
        obs_dist.append(min(distances))
    
    # 确保T和obs_dist长度相同
    min_length = min(len(T), len(obs_dist))
    T = T[:min_length]
    obs_dist = obs_dist[:min_length]
    
    plt.plot(T, obs_dist, color='gold', label='SOD-MPC-CBF', linewidth=4)
    
    # Calculate the range for y-axis
    min_dist = min(obs_dist)
    max_dist = max(obs_dist)
    
    # Set axis limits
    plt.xlim(0, math.ceil(max(T)))
    plt.ylim(max(0, min_dist - 0.1), max_dist + 0.1)
    
    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('distance [m]', fontsize=20)
    plt.title('Distance to Nearest Obstacles', fontsize=24)
    
    # Set x-axis ticks with larger intervals
    x_ticks = range(0, math.ceil(max(T)) + 1, 10)
    plt.xticks(x_ticks, fontsize=18)
    
    # Set y-axis ticks with 0.1 intervals
    y_min = math.floor(max(0, min_dist) * 10) / 10  # Round down to nearest 0.1
    y_max = math.ceil(max_dist * 10) / 10  # Round up to nearest 0.1
    y_ticks = np.arange(y_min, y_max + 0.1, 0.1)
    plt.yticks(y_ticks, fontsize=18)
    
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_plot(plt, 'obstacle_distances.png')
    plt.close()

# 主函数
def main():
    
    target_points = [(5, 5, np.pi/2),]  #目标点
    obstacles = [
    (0.8, 0.8, 0.1),
    (1.5, 1.5, 0.1),
    (2.25, 2.25, 0.1),
    (0.75, 2.25, 0.1),
    (2.2, 0.87, 0.1),
    (4.75, 3.75, 0.1),
    (1.5, 0.18, 0.1),
    (0.75, 3.0, 0.1),
    (4.25, 2.45, 0.1),
    (2.95, 1.5, 0.1)]
    
    generate_trajectory_plot(target_points, obstacles)
    generate_time_analysis_plot()
    generate_cost_plot()
    generate_distance_to_goal_plot(target_points)
    generate_control_inputs_plot()
    generate_obstacle_distance_plot(obstacles)

if __name__ == "__main__":
    main()
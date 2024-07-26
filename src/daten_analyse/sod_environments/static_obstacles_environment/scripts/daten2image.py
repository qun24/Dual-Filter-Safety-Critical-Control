import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from cycler import cycler
import numpy as np
import os

# 共享配置
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/static_obstacles_environment/one_obstacle'
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/static_obstacles_environment/five_obstacles'
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/static_obstacles_environment/ten_obstacles'
FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/static_obstacles_environment/twenty_obstacles'



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
    
    # 绘制目标点和障碍物
    for i, point in enumerate(target_points, 1):
        plt.plot(point[0], point[1], marker='*', color='green', markersize=35)
        plt.annotate(f'$\\it{{p_{i}}}$', 
                    (point[0], point[1]), 
                    xytext=(10, 10),
                    textcoords='offset points', 
                    fontsize=20, 
                    color='black')
    
    for obstacle in obstacles:
        solid_circle = Circle(obstacle[:2], obstacle[2], color="black", linewidth=4)
        plt.gca().add_patch(solid_circle)
        dashed_circle_safe = Circle(obstacle[:2], 0.37, color="orange", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
        dashed_circle_safe = Circle(obstacle[:2], 0.4, color="black", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
        dashed_circle_safe = Circle(obstacle[:2], 0.7, color="purple", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
    
    
    all_x = data["x_path"] + [obs[0] for obs in obstacles]
    all_y = data["y_path"] + [obs[1] for obs in obstacles]
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1

    # 设置坐标轴范围
    x_min, x_max = min(x_path), max(x_path)
    y_min, y_max = min(y_path), max(y_path)
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
    # 定义目标点和障碍物
    # one_obstacle
    # target_points = [(4, 4, np.pi/2),]  #目标点
    # obstacles = [(2.5, 2.5, 0.1)]  # 障碍物(x, y, radius)  
    # # five_obstacle
    # target_points = [(4, 4, np.pi/2),]  #目标点
    # obstacles = [(1.75, 1.8, 0.1),(2.4, 1.25, 0.1),(2.4, 2.35, 0.1),(3.05, 1.8, 0.1),(3.2, 2.6, 0.1)]  # 障碍物(x, y, radius)  

    #ten_obstacles
    # target_points = [(4, 4, np.pi/2),]  #目标点
    # obstacles = [
    # (0.75, 0.75, 0.1),
    # (1.5, 1.5, 0.1),
    # (2.25, 2.25, 0.1),
    # (3.0, 3.0, 0.1),
    # (1.69, 0.56, 0.1),
    # (0.56, 1.69, 0.1),
    # (2.44, 1.31, 0.1),
    # (1.31, 2.44, 0.1),
    # (3.2, 2.06, 0.1),
    # (2.06, 3.2, 0.1)]
    #twenty_obstacles
    target_points = [(5, 5, np.pi/2),]  #目标点
    obstacles = [
    (0.75, 0.75, 0.1),
    (1.8, 1.2, 0.1),
    (2.25, 2.25, 0.1),
    (1.05, 1.95, 0.1),
    (2.55, 0.45, 0.1),
    (3.3, 2.7, 0.1),
    (1.5, 0.05, 0.1),
    (3.75, 0.75, 0.1),
    (3.0, 4.5, 0.1),
    (2.95, 1.5, 0.1),
    (0.05, 1.5, 0.1),
    (1.5, 3.0, 0.1),
    (4.8, 1.2, 0.1),
    (0.75, 3.75, 0.1),
    (4.05, 1.95, 0.1),
    (4.8, 4.2, 0.1),
    (3.75, 3.75, 0.1),
    (4.5, 3.0, 0.1),
    (2.55, 3.45, 0.1),
    (1.8, 4.2, 0.1)]
    
    generate_trajectory_plot(target_points, obstacles)
    generate_time_analysis_plot()
    generate_cost_plot()
    generate_distance_to_goal_plot(target_points)
    generate_control_inputs_plot()
    generate_obstacle_distance_plot(obstacles)

if __name__ == "__main__":
    main()
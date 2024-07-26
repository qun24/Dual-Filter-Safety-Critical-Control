import json
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import os

# 共享配置
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/previous_environments/ten_obstacles/comparing'
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/previous_environments/five_obstacles/comparing'
# FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/previous_environments/one_obstacle/comparing'
FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/previous_environments/twenty_one_obstacles/comparing'


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
    data1 = load_json_data('xy_path_mpcall.json')
    data2 = load_json_data('xy_path_mpczone.json')
    data3 = load_json_data('xy_path_mpc_cbf.json')
    data4 = load_json_data('xy_path_base.json')
    data5 = load_json_data('xy_path_sod.json')
    
    if not all([data1, data2, data3, data4,data5]):
        return

    plt.figure(figsize=(14, 10))
    # # 修改 MPC-All 的数据处理 ten_obstacles
    # x_path1 = data1["x_path"][:114]
    # y_path1 = data1["y_path"][:114]
    x_path1 = data1["x_path"][:22]
    y_path1 = data1["y_path"][:22]
    # x_path1 = data1["x_path"]
    # y_path1 = data1["y_path"]
    
    # 绘制不同方法的轨迹，增加线宽
    plt.plot(x_path1, y_path1, color='blue', label='MPC-All', linewidth=4)
    # plt.plot(data1["x_path"], data1["y_path"], color='blue', label='MPC-All', linewidth=4)
    plt.plot(data2["x_path"], data2["y_path"], color='red', label='MPC-Zone', linewidth=4)
    plt.plot(data3["x_path"], data3["y_path"], color='green', label='MPC-CBF(TypeII)', linewidth=4)
    plt.plot(data4["x_path"], data4["y_path"], color='black', label='MPC-Base', linewidth=4)
    plt.plot(data5["x_path"], data5["y_path"], color='gold', label='SOD-MPC-CBF', linewidth=4)
    
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
    
    # 计算适当的显示范围
    all_x = x_path1 + data2["x_path"] + data3["x_path"] + data4["x_path"] + data5["x_path"] + [obs[0] for obs in obstacles]
    all_y = y_path1 + data2["y_path"] + data3["y_path"] + data4["y_path"] + data5["y_path"] + [obs[1] for obs in obstacles]
    
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.xlabel('x position [m]', fontsize=23)
    plt.ylabel('y position [m]', fontsize=23)
    plt.title('Trajectories', fontsize=24)
    plt.xticks(np.arange(int(x_min), int(x_max)+1, 1), fontsize=18)
    plt.yticks(np.arange(int(y_min), int(y_max)+1, 1), fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    save_plot(plt, 'trajectories.png')
    plt.close()

def generate_distance_to_goal_plot(target_points):
    data1 = load_json_data('xy_path_mpcall.json')
    data2 = load_json_data('xy_path_mpczone.json')
    data3 = load_json_data('xy_path_mpc_cbf.json')
    data4 = load_json_data('xy_path_base.json')
    data5 = load_json_data('xy_path_sod.json')
    
    if not all([data1, data2, data3, data4, data5]):
        return

    plt.figure(figsize=(14, 7))

    max_time = 0
    max_distance = 0
    all_target_times = set()

    for data, label, color in zip([data1, data2, data3, data4, data5],
                                  ['MPC-All', 'MPC-Zone', 'MPC-CBF(TypeII)', 'MPC-Base', 'SOD-MPC-CBF'],
                                  ['blue', 'red', 'green', 'black', 'gold']):
        x_path = data["x_path"]
        y_path = data["y_path"]
        theta_path = data["theta_path"]
        T = data["T"][:-1]

        # if label == 'MPC-All':#ten_obstacles
        #     x_path = x_path[:114]
        #     y_path = y_path[:114]
        #     theta_path = theta_path[:114]
        #     T = T[:114]

        if label == 'MPC-All':#21_obstacles
            x_path = x_path[:22]
            y_path = y_path[:22]
            theta_path = theta_path[:22]
            T = T[:22]

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
        
        plt.plot(T[:len(distance)], distance, color=color, label=label, linewidth=4)
        
        max_time = max(max_time, T[-1])
        max_distance = max(max_distance, max(distance))

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

def generate_obstacle_distance_plot(obstacles):
    data1 = load_json_data('xy_path_mpcall.json')
    data2 = load_json_data('xy_path_mpczone.json')
    data3 = load_json_data('xy_path_mpc_cbf.json')
    data4 = load_json_data('xy_path_base.json')
    data5 = load_json_data('xy_path_sod.json')
    
    if not all([data1, data2, data3, data4, data5]):
        return
    
    plt.figure(figsize=(14, 7))
    
    all_distances = []
    max_time = 0
    
    for data, label, color in zip([data1, data2, data3, data4, data5],
                                  ['MPC-All', 'MPC-Zone', 'MPC-CBF(TypeII)', 'MPC-Base', 'SOD-MPC-CBF'],
                                  ['blue', 'red', 'green', 'black', 'gold']):
        x_path = data["x_path"]
        y_path = data["y_path"]
        T = data["T"][:-1]
        
        if label == 'MPC-All':
            x_path = x_path[:22]
            y_path = y_path[:22]
            T = T[:22]
        
        obs_dist = []
        for x, y in zip(x_path, y_path):
            distances = [np.sqrt((x - obs[0])**2 + (y - obs[1])**2) - obs[2] for obs in obstacles]
            obs_dist.append(min(distances))
        
         # 确保T和obs_dist长度相同
        min_length = min(len(T), len(obs_dist))
        T = T[:min_length]
        obs_dist = obs_dist[:min_length]
        
        all_distances.extend(obs_dist)
        max_time = max(max_time, T[-1])
        
        plt.plot(T, obs_dist, color=color, label=label, linewidth=4)
    
    # Calculate the range for y-axis
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    
    # Set axis limits
    plt.xlim(0, math.ceil(max_time))
    plt.ylim(max(0, min_dist - 0.1), max_dist + 0.1)
    
    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('distance [m]', fontsize=20)
    plt.title('Distance to Nearest Obstacles', fontsize=24)
    
    # Set x-axis ticks with larger intervals
    x_ticks = range(0, math.ceil(max_time) + 1, 10)  # Increased interval to 10
    plt.xticks(x_ticks, fontsize=18)
    
    # Set y-axis ticks with 0.1 intervals
    y_min = math.floor(max(0, min_dist) * 10) / 10  # Round down to nearest 0.1
    y_max = math.ceil(max_dist * 10) / 10  # Round up to nearest 0.1
    y_ticks = np.arange(y_min, y_max + 0.1, 0.1)
    plt.yticks(y_ticks, fontsize=18)
    
    plt.legend(fontsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    save_plot(plt, 'obstacle_distance.png')
    plt.close()

def main():
    # 定义目标点和障碍物
    # one obstacle
    # target_points = [(3, 4, np.pi/2)]  # 目标点
    # obstacles = [(1.5, 2, 0.1)]  # 障碍物(x, y, radius)

    # five obstacles
    # target_points = [(3, 3, np.pi/2)]  # 目标点
    # obstacles = [(1.5,1.5,0.1),(0.75,0.75,0.1),(2.25,0.75,0.1),(0.75,2.25,0.1),(2.25,2.25,0.1)]  # 障碍物(x, y, radius)

    # ten obstacles
    # target_points = [(3, 3, -np.pi/2),(3, 0, -np.pi),(0, 0, 0)]  # 目标点
    # obstacles = [(1.5,1.5,0.1),(0.75,0.75,0.1),(2.25,0.75,0.1),(0.75,2.25,0.1),(2.25,2.25,0.1),(-0.75,-0.75,0.1),(1.5,0.05,0.1),(0.75,-1,0.1),(-0.75,2.25,0.1),(2.95,1.5,0.1)]  # 障碍物(x, y, radius)
    
    # twenty_one obstacles
    target_points = [(3, 3,-np.pi/2), (3, 0,-np.pi), (0, 3,-np.pi/2),(0,0,0)]
    obstacles = [(1.5, 1.5, 0.1),(0.75, 0.75, 0.1),(0.75, 2.25, 0.1),(2.25, 0.75, 0.1),(2.25, 2.25, 0.1),(-1.0, -1.0, 0.1),
(1.5, 0.05, 0.1),(0.75, -1.0, 0.1),(-1.0, 4.0, 0.1),(2.9, 1.5, 0.1),(0.1, 1.5, 0.1),(1.5, 3.0, 0.1),(4.0, 0.75, 0.1),(-1.0, 0.75, 0.1),
(4.0, 2.25, 0.1),(2.25, -1.0, 0.1),(4.0, 4.0, 0.1),(4.0, -1.0, 0.1),(2.25, 4.0, 0.1),(-1.0, 2.25, 0.1),(0.75, 4.0, 0.1),]
    generate_trajectory_plot(target_points, obstacles)
    generate_distance_to_goal_plot(target_points)
    generate_obstacle_distance_plot(obstacles)

if __name__ == "__main__":
    main()
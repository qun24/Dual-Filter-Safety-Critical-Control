import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from cycler import cycler
import numpy as np
import os

# 共享配置
FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/previous_environments/twenty_one_obstacles'

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
    
    plt.xlabel('x position [m]', fontsize=23)
    plt.ylabel('y position [m]', fontsize=23)
    plt.title('Robot Trajectory', fontsize=24)
    
    # 设置坐标轴范围
    x_min, x_max = min(x_path), max(x_path)
    y_min, y_max = min(y_path), max(y_path)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    
    plt.grid(True)
    plt.legend(fontsize=16)# 增加图例字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    save_plot(plt, 'robot Trajectory.png')
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
    y_ticks = np.arange(0, y_max + 1001, 1000.0)
    plt.yticks(y_ticks, fontsize=18)
    plt.ylim(0, y_ticks[-1])

    plt.legend(fontsize=18)
    plt.grid(True)

    plt.tight_layout()
    save_plot(plt, 'cost.png')
    plt.close()

def generate_disttogoal_plot(targets):
    data = load_json_data('xy_path_sod.json')
    if data is None:
        return
    
    x_path = data["x_path"]
    y_path = data["y_path"]
    theta_path = data["theta_path"]
    T = data["T"][:-1]
    
    assert len(x_path) == len(T) == len(y_path) == len(theta_path), "All data lengths must be equal"
    
    plt.figure(figsize=(14, 7))
    
    target_index = 0
    distances = []
    target_reached_times = []
    
    for x, y, theta, t in zip(x_path, y_path, theta_path, T):
        x_idk = np.array([x, y, theta])
        xs = np.array(targets[target_index])
        
        current_distance = np.linalg.norm(x_idk - xs)
        dist_to_goal = np.sqrt((x - targets[target_index][0])**2 + (y - targets[target_index][1])**2)
        distances.append(dist_to_goal)
        
        if current_distance < 0.05:
            target_reached_times.append((t, target_index))
            if target_index + 1 < len(targets):
                target_index += 1
    
    plt.plot(T, distances, linewidth=2, label='Distance to Current Goal')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
    for i, (t, index) in enumerate(target_reached_times):
        plt.axvline(x=t, color=colors[index], linestyle='--', label=f'Reached Goal {index+1}')
        plt.text(t, max(distances), f'Goal {index+1}', rotation=90, verticalalignment='bottom')
    
    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('Distance to Goal [m]', fontsize=20)
    plt.title('Distance to Goal', fontsize=24)
    
    # Set x-axis range and ticks
    x_min, x_max = int(min(T)), int(max(T)) + 1
    plt.xlim(x_min, x_max)
    plt.xticks(range(x_min, x_max + 1), fontsize=18)
    
    # Set y-axis range and ticks
    y_min, y_max = 0, int(max(distances)) + 1
    plt.ylim(y_min, y_max)
    plt.yticks(range(y_min, y_max + 1), fontsize=18)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    save_plot(plt, 'distance_to_goals.png')
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

def generate_obsdis_plot(obstacles):
    xy_data = load_json_data('xy_path_sod.json')
    if xy_data is None:
        return

    x_path = xy_data["x_path"]
    y_path = xy_data["y_path"]
    T = xy_data["T"][:-1]

    obs_distances = []
    min_distances = []
    for obs_x, obs_y, obs_radius in obstacles:
        distances = [np.sqrt((x - obs_x)**2 + (y - obs_y)**2) - obs_radius -0.15 for x, y in zip(x_path, y_path)]
        obs_distances.append(distances)
        min_distances.append(min(distances))

    # 计算所有障碍物的最小距离
    overall_min_distances = [min(distances) for distances in zip(*obs_distances)]

    plt.figure(figsize=(14, 7))
    
    # 使用不同的颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(obstacles) + 1))  # +1 for overall min distances
    plt.gca().set_prop_cycle(cycler('color', colors))

    for i, distances in enumerate(obs_distances):
        color = colors[i]
        plt.plot(T, distances, label=f'Obstacle {i+1}', linewidth=2, color=color)
        
        # 找出每条曲线的最小点
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        min_time = T[min_index]
        
        # 用红色标注最小点
        plt.plot(min_time, min_distance, 'o', color='red', markersize=8)
        plt.vlines(min_time, 0, min_distance, colors='red', linestyles='dashed', alpha=0.7)

    # 绘制计算的整体最小距离
    plt.plot(T, overall_min_distances, color=colors[-1], label='Overall Min Distance', linewidth=2)

    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('DTNO [m]', fontsize=20)
    plt.title('Distance to Obstacles Over Time', fontsize=24)
    
    # 设置x轴刻度
    plt.xticks(range(0, max(int(max(T)) + 1, 50), 5), fontsize=16)
    
    # 设置y轴刻度，确保包含所有最小值
    y_ticks = list(plt.yticks()[0])  # 获取当前的y轴刻度
    for min_dist in min_distances + [min(overall_min_distances)]:
        y_ticks.append(min_dist)  # 添加每个最小距离值
    y_ticks = sorted(list(set(y_ticks)))  # 去重并排序
    plt.yticks(y_ticks, fontsize=16)
    
    # 将图例放在图的右侧
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)  # 确保y轴从0开始

    plt.tight_layout()
    save_plot(plt, 'obstacle_distances.png')
    plt.close()

# 主函数
def main():
    # 定义目标点和障碍物
    # one_obstacle
    target_points = [(3, 4, np.pi/2),]  #目标点
    obstacles = [(1.5, 2, 0.1)]  # 障碍物(x, y, radius)  
    
    # generate_trajectory_plot(target_points, obstacles)
    generate_time_analysis_plot()
    generate_cost_plot()
    # generate_disttogoal_plot(target_points)
    generate_control_inputs_plot()
    # generate_obsdis_plot(obstacles)

if __name__ == "__main__":
    main()
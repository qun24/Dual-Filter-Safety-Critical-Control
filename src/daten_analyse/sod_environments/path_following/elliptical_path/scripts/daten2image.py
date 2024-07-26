import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from cycler import cycler
import numpy as np
import os

# 共享配置
FOLDER_PATH = '/home/qun/turtlebot3_realsensed435i/src/sod_avoidence/daten_analyse/sod_environments/path_following/elliptical_path'

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

def generate_elliptical_trajectory(a, b, num_points):
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    x = a * np.cos(t)
    y = b * np.sin(t)
    dx = -a * np.sin(t)
    dy = b * np.cos(t)
    theta = np.arctan2(dy, dx)
    
    start_index = np.argmin(y)
    x = np.roll(x, -start_index)
    y = np.roll(y, -start_index)
    theta = np.roll(theta, -start_index)
    
    theta = (theta - theta[0] + 2*np.pi) % (2*np.pi)
    trajectory = [(x[i], y[i], theta[i]) for i in range(len(x))]
    return trajectory

def generate_trajectory_plot(obstacles):
    data = load_json_data('xy_path_sod.json')
    if data is None:
        return
    
    x_path = data["x_path"]
    y_path = data["y_path"]
    T = data["T"][:-1]  # 从T中删除最后一个元素
    
    assert len(x_path) == len(T) == len(y_path), "所有数据长度必须相同"
    
    plt.figure(figsize=(14, 10))
    
    # 绘制椭圆轨迹（使用点）
    elliptical_trajectory = generate_elliptical_trajectory(4.0, 2.5, 100)
    x_ellipse, y_ellipse, _ = zip(*elliptical_trajectory)
    plt.scatter(x_ellipse, y_ellipse, color='green', s=20, label='Elliptical Trajectory', zorder=2)
    
    # 绘制机器人轨迹
    plt.plot(x_path, y_path, color='gold', label='Robot Trajectory', linewidth=4)
    
    # 绘制障碍物
    for obstacle in obstacles:
        solid_circle = Circle(obstacle[:2], obstacle[2], color="black", linewidth=4)
        plt.gca().add_patch(solid_circle)
        dashed_circle_safe = Circle(obstacle[:2], 0.37, color="orange", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
        dashed_circle_safe = Circle(obstacle[:2], 0.4, color="black", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
        dashed_circle_safe = Circle(obstacle[:2], 0.7, color="purple", fill=False, linestyle='--', linewidth=4)
        plt.gca().add_patch(dashed_circle_safe)
    
    # 设置坐标轴范围
    all_x = x_path + [obs[0] for obs in obstacles] + list(x_ellipse)
    all_y = y_path + [obs[1] for obs in obstacles] + list(y_ellipse)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    
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



# 主函数
def main():
    # 障碍物(x, y, radius)  
    obstacles = [
    (3.0, -1.7, 0.1),
    (3.0, 1.7, 0.1),
    (-3.0, 1.7, 0.1),
    (-3.0, -1.7, 0.1),
    (0.0, 3.2, 0.1),
    (0.0, -3.2, 0.1),
    (2.0, -1.0, 0.1),
    (2.0, 1.0, 0.1),
    (-2.0, 1.0, 0.1),
    (-2.0, -1.0, 0.1),
    (0.0, 2.0, 0.1),
    (0.0, -2.0, 0.1),
    (0.0, -1.0, 0.1),
    (0.0, 1.0, 0.1),
    (-1.0, 0.0, 0.1),
    (1.0, 0.0, 0.1)
]
    generate_trajectory_plot(obstacles)
    generate_time_analysis_plot()
    generate_cost_plot()
    generate_control_inputs_plot()

if __name__ == "__main__":
    main()
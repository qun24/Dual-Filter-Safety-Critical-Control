#!/usr/bin/env python3
import casadi as ca
import cvxpy as cp
import json
import numpy as np
import rospy
import time
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
import concurrent.futures

def newOdom(msg):
    global x_real, y_real, theta_real
    x_real = msg.pose.pose.position.x
    y_real = msg.pose.pose.position.y
    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta_real) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

def processed_obstacle_callback(data):
    global latest_obstacles
    latest_obstacles = json.loads(data.data)

def trajectory_callback(msg):
    global target_points
    target_points = [(pose.position.x, pose.position.y, euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]) for pose in msg.poses]

def is_obstacle_near_target(target, obstacles, threshold=0.7):
    target_x, target_y, _ = target
    for obs in obstacles:
        obs_x, obs_y, obs_r, _, _ = obs
        distance = np.sqrt((target_x - obs_x)**2 + (target_y - obs_y)**2)
        if distance < (obs_r + threshold):
            return True
    return False


def process_obstacle(last_obstacle, robot_state, control_params):
    x_real, y_real, theta_real, l = robot_state
    v_max, omega_max, u_real = control_params
    obs_x, obs_y, obs_r, obs_vx, obs_vy = last_obstacle
    # 预计算和固定参数
    cos_theta = np.cos(theta_real)
    sin_theta = np.sin(theta_real)
    delta_x = x_real - obs_x 
    delta_y = y_real - obs_y
    delta_x_offset = delta_x + l * cos_theta
    delta_y_offset = delta_y + l * sin_theta
    distance = np.sqrt(delta_x_offset ** 2 + delta_y_offset ** 2) - obs_r - rob_diam/2 - 0.2

    if distance <= 0.4:
        CBF_Condition = distance
        Q_cbf = np.array([[1000, 0], [0, 1]])
        c_cbf = np.zeros(2)

        stop = 0
        if distance <= 0.2:
            obs_rel_y_1 = (delta_x + obs_vx*1) * sin_theta - (delta_y + obs_vy*1) * cos_theta 
            obs_rel_y_2 = (delta_x - obs_vx*1) * sin_theta - (delta_y - obs_vy*1) * cos_theta  
            obs_mul = obs_rel_y_1 * obs_rel_y_2

            obs_rel_y = delta_x * sin_theta - delta_y * cos_theta
            obs_rel_vy = obs_vy * cos_theta - obs_vx * sin_theta

            if (0 < obs_rel_y < 0.2 and obs_rel_vy < -0.05) or (-0.2 < obs_rel_y < 0 and obs_rel_vy > 0.05) or obs_mul < 0:
                stop = 1

        dist_sq = delta_x_offset ** 2 + delta_y_offset ** 2
        e1 = -1.0 / np.sqrt(dist_sq) * (delta_x_offset * cos_theta + delta_y_offset * sin_theta)
        e2 = -1.0 / np.sqrt(dist_sq) * (delta_x_offset * sin_theta * -l + delta_y_offset * cos_theta * l)
        e3 = 1.0 / np.sqrt(dist_sq) * (delta_x_offset * obs_vx + delta_y_offset * obs_vy)## CBF for move obstacle

        CBF_Condition = distance - e3
        A = np.array([[e1, e2], [1, 0], [0, 1], [0, -1]])
        b_cbf = np.array([CBF_Condition, v_max, omega_max, omega_max]).reshape(-1, 1)

        u1 = cp.Variable()
        u2 = cp.Variable()
        if u_real[0] >= 0:
            objective = cp.Minimize(0.5 * cp.quad_form(cp.vstack([u1 - 0.2, u2]), Q_cbf) + c_cbf @ cp.vstack([u1, u2]))
        else:
            objective = cp.Minimize(0.5 * cp.quad_form(cp.vstack([u1 + 0.2, u2]), Q_cbf) + c_cbf @ cp.vstack([u1, u2]))

        constraints = [cp.matmul(A, cp.vstack([u1, u2])) <= b_cbf]
        problem = cp.Problem(objective, constraints)
        problem.solve()
            
        rate = distance / 0.4
        rate = max(0, min(distance / 0.4, 1))
        u_real[0] = u1.value * (1 - rate) + rate * u_real[0]
        u_real[1] = u2.value * (1 - rate) + rate * u_real[1]

        if stop == 1:
            u_real[0] = 0
            u_real[1] = 0
    return distance

# 全局变量
x_real = 0
y_real = 0
theta_real = 0
latest_obstacles = []

rospy.init_node("pi_controller")
sub = rospy.Subscriber("/odom", Odometry, newOdom)
sub2 = rospy.Subscriber("/processed_obstacles", String, processed_obstacle_callback)
sub3 = rospy.Subscriber("/trajectory", PoseArray, trajectory_callback)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

speed = Twist()

### MPC参数
T = 0.05  # 时间步长
N = 30    # 预测步数

rob_diam = 0.3 
v_max = 0.2
omega_max = np.pi / 8.0 
l = 0.02  # 相机距离小车中心的偏移量


x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
v = ca.SX.sym('v')
omega = ca.SX.sym('omega')

states = ca.vertcat(x, y, theta)
controls = ca.vertcat(v, omega)
n_states = states.size()[0]
n_controls = controls.size()[0]

rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

U = ca.SX.sym('U', n_controls, N)
X = ca.SX.sym('X', n_states, N+1)
P = ca.SX.sym('P', n_states + n_states)

X[:, 0] = P[:3]
for i in range(N):
    f_value = f(X[:, i], U[:, i])
    X[:, i+1] = X[:, i] + f_value * T

ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

Q = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 0.005]])  # x、y、theta权重
R = np.array([[0.1, 0.0], [0.0, 0.01]])  # 线速度v和角速度w的权重
Rd = np.diag([10.0, 5.0])  #线速度和角速度变化的施加惩罚


obj = 0
g = []
for i in range(N):
    state_error = X[:, i] - P[3:]
    obj += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([U[:, i].T, R, U[:, i]])
    if i < N-1:  # Add control change cost
        control_change = U[:, i+1] - U[:, i]
        obj += ca.mtimes([control_change.T, Rd, control_change])
cost_func = ca.Function('cost_func', [U, P], [obj], ['control_input', 'params'], ['cost'])
cost_values = []


nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}
opts_setting = {'ipopt.max_iter': 100 , 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

lbg = []
ubg = []

lbx = []
ubx = []

for _ in range(N):
    lbx.append(0)
    ubx.append(v_max)
    lbx.append(-omega_max)
    ubx.append(omega_max)

t0 = 0.0
u0 = np.array([0.0, 0.0] * N).reshape(-1, 2)

x_c = []
u_c = []
t_c = []
xx = []

index_t = []

start_time = time.time()
### target point
target_points = [] 
current_target_index = 0

x_path = []
y_path = []
theta_path = []
u0_real_list = []
u1_real_list = []
execution_times = []
t = []
car_circle = 0
start_time3 = time.time()
T_zong_list = []
min_dist_list = []
flag = False
Rate = rospy.Rate(10)

while not flag:
    if not target_points:
        rospy.loginfo("Waiting for trajectory...")
        Rate.sleep()
        continue
     # 检查并跳过被障碍物占据的路径点
    while current_target_index < len(target_points):
        if latest_obstacles and is_obstacle_near_target(target_points[current_target_index], latest_obstacles):
            current_target_index += 1
        else:
            break
 
    if current_target_index < len(target_points):
        target = np.array(target_points[current_target_index]).reshape(-1, 1)
        initial_state = np.array([x_real, y_real, theta_real]).reshape(-1, 1)
        ref_trajectory = np.concatenate((initial_state, target))
        init_control = ca.reshape(u0, -1, 1)

        start_time = time.time()
        res = solver(x0=init_control, p=ref_trajectory, lbx=lbx, ubx=ubx)
        index_t.append(time.time() - start_time)

        u_sol = ca.reshape(res['x'], n_controls, N)  # 获取MPC输入
        ff_value = ff(u_sol, ref_trajectory)
        u_real = u_sol[:, 0]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        
        if latest_obstacles:
            stop = 0
            distlist = []
            robot_state = (x_real, y_real, theta_real, l)
            control_params = (v_max, omega_max, u_real)

            for obstacle in latest_obstacles:
                distance = process_obstacle(obstacle, robot_state, control_params)
                distlist.append(distance)

            mindist = min(distlist)
            min_dist_list.append(mindist)
            
        ### input for turtlebot3_waffle
        speed.linear.x = u_real[0]
        speed.angular.z = u_real[1]
        ### save data
        x_path.append(x_real)
        y_path.append(y_real)
        theta_path.append(theta_real)
        u0_real_list.append(u_real.full()[0][0])
        u1_real_list.append(u_real.full()[1][0])
        current_cost = cost_func(u_sol, ref_trajectory)
        cost_values.append(float(current_cost))
        pub.publish(speed)

        if np.linalg.norm(initial_state[:2] - target[:2]) < 0.1:
            current_target_index += 1 
    else:
        ### finished and stopped
        speed.linear.x = 0
        speed.angular.z = 0
        pub.publish(speed)
        flag = True

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    t_zong = end_time - start_time3
    T_zong_list.append(t_zong)
    Rate.sleep()

average_execution_time = sum(execution_times) / len(execution_times)

### save data in json
data_to_save = {
    "xy_path_sod": {"x_path": x_path, "y_path": y_path, "theta_path": theta_path, "T": T_zong_list},
    "time_analysis": {
        "mpc_cbf_execution_times": execution_times,
        "average_execution_time": average_execution_time,
        "T": T_zong_list
    },
    "cost_values": {"cost_values": cost_values, "T": T_zong_list},
    "input": {"v": u0_real_list, "w": u1_real_list, "T": T_zong_list},
    "min_dist": {"min": min_dist_list, "T": T_zong_list}
}

for filename, data in data_to_save.items():
    with open(f"{filename}.json", "w") as file:
        json.dump(data, file)
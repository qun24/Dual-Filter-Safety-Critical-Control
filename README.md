# E-FastSOD-Zone-Safety-Critical-Control
# Obstacle Avoidance and Path Following for Mobile Robots using Salient Object Detection and Type II D-ZCBFs

This program demonstrates a novel algorithm that combines Salient Object Detection (SOD) and Type II D-ZCBFs obstacle avoidance to enable a vehicle equipped with a depth camera to avoid obstacles and reach a specified destination or follow a corresponding trajectory. The simulation environment is based on Ubuntu 20.04, ROS Noetic, and Gazebo 11.

## Algorithms
- Salient Object Detection:
  - Reference: Minimum Barrier Salient Object Detection at 80 FPS 
  - C++ code reference: https://github.com/coderSkyChen/MBS_Cplus_c-
- Type II D-ZCBFs Obstacle Avoidance 
  - Reference: Online Efficient Safety-Critical Control for Mobile Robots in Unknown Dynamic Multi-Obstacle Environments
  - Code reference: https://github.com/GuangyaoTian/TypeII-D-ZCBFs/tree/main

## Simulation Environment
The package is primarily used in a Gazebo simulation environment that includes the TurtleBot3 and RealSense camera ROS packages. The simulated robot is a TurtleBot3 Waffle Pi equipped with a RealSense D435i depth camera. The Rviz and Gazebo files have been configured to include depth camera data.

## Scenarios
The package is mainly used in the Gazebo simulation environment and is divided into three types of scenarios:

### 1. Static Environment
Launch the Gazebo simulation environment:
```
roslaunch type2_D_ZCBFs turtlebot3_static_cylinder_obs_world.launch world_env:=1
```
- `world_env:=1`: One obstacle environment
- `world_env:=2`: Five obstacle environment 
- `world_env:=3`: Ten obstacle environment
- `world_env:=4`: Twenty obstacle environment

Launch the salient object detection algorithm, point cloud filtering, obstacle circle construction and other algorithms for obtaining obstacle information:
```
roslaunch type2_D_ZCBFs turtlebot3_dangerous_obstacle_detection.launch
```

Finally, launch the vehicle obstacle avoidance algorithm:
```
python3 src/sod_avoidence/type2_D_ZCBFs/static_obstacles_avoidence/scripts/static_obs_typeII-d-zcbfs_avoidence.py
```

### 2. Dynamic Environment I (Simple circular obstacles, both static and dynamic in the simulation environment)
Launch the Gazebo simulation environment:
```
roslaunch type2_D_ZCBFs turtlebot3_mix_cylinder_obs_world.launch
```

Launch the salient object detection algorithm, point cloud filtering, obstacle circle construction and other algorithms for obtaining obstacle information:
```
roslaunch type2_D_ZCBFs turtlebot3_dangerous_obstacle_detection.launch
```

Launch the obstacle movement code:
```
python3 src/sod_avoidence/type2_D_ZCBFs/move_obstacles_avoidence/scripts/control_obs_move.py  
```

Launch the vehicle obstacle avoidance algorithm:
```
python3 src/sod_avoidence/type2_D_ZCBFs/move_obstacles_avoidence/scripts/move_obs_typeII-d-zcbfs_avoidence.py
```

### 3. Dynamic Environment II (Simulated warehouse environment with 10 moving Turtlebot3 robots)
Launch the Gazebo simulation environment:
```
roslaunch type2_D_ZCBFs turtlebot3_warehouse_world.launch
```

Launch the launch file for the salient object detection algorithm, point cloud filtering, obstacle circle construction and other algorithms for obtaining obstacle information:
```
roslaunch type2_D_ZCBFs turtlebot3_dangerous_obstacle_detection.launch
```

Launch the vehicle movement code:
```
python3 src/sod_avoidence/type2_D_ZCBFs/move_obstacles_avoidence/scripts/control_robot_move.py
```

Launch the vehicle obstacle avoidance algorithm:
```
python3 src/sod_avoidence/type2_D_ZCBFs/move_obstacles_avoidence/scripts/move_obs_typeII-d-zcbfs_avoidence.py  
```

### 4. Path Following (Sine function trajectory and elliptical trajectory) 
#### 4.1 Sine Function Trajectory Environment (including generating trajectories and displaying trajectories in Rviz)
Launch the Gazebo environment:
```
roslaunch type2_D_ZCBFs turtlebot3_sin_path_following.launch
```

Launch the launch file for the salient object detection algorithm, point cloud filtering, obstacle circle construction and other algorithms for obtaining obstacle information:
```
roslaunch type2_D_ZCBFs turtlebot3_dangerous_obstacle_detection.launch
```

Launch the vehicle trajectory tracking and obstacle avoidance algorithm:
```
python3 src/sod_avoidence/type2_D_ZCBFs/path_following_with_obstacles/scripts/sin_path_following_typeII-d-zcbfs_avoidence.py
```

#### 4.2 Elliptical Trajectory Environment
Launch the Gazebo environment:
```
roslaunch type2_D_ZCBFs turtlebot3_elliptical_path_following.launch
```

Launch the launch file for the salient object detection algorithm, point cloud filtering, obstacle circle construction and other algorithms for obtaining obstacle information:
```
roslaunch type2_D_ZCBFs turtlebot3_dangerous_obstacle_detection.launch
```

Launch the vehicle trajectory tracking and obstacle avoidance algorithm:  
```
python3 src/sod_avoidence/type2_D_ZCBFs/path_following_with_obstacles/scripts/elliptical_path_following_typeII-d-zcbfs_avoidence.py
```

The trajectory can be changed in `python3 src/sod_avoidence/type2_D_ZCBFs/path_following_with_obstacles/scripts/path_generator.py`.
All comparison data, charts, and videos for each scenario are available in the daten_analyse folder.

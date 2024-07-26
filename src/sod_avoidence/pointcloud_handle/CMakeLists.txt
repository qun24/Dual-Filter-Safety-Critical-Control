cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_handle)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
)

## System dependencies are found with CMake's conventions
find_package(PCL 1.7 REQUIRED)
find_package(CGAL REQUIRED)
# 设置默认的构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(${CGAL_USE_FILE})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pointcloud_handle
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)
include(${CGAL_USE_FILE})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

## Declare a C++ executable
add_executable(obs_pcl_filter src/obs_pcl_filter.cpp)
target_link_libraries(obs_pcl_filter
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)



add_executable(obs_pcl_circle src/obs_pcl_circle.cpp)
target_link_libraries(obs_pcl_circle
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(obs_circle_rviz src/obs_circle_rviz.cpp)
target_link_libraries(obs_circle_rviz
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(obs_sphere_rviz src/obs_sphere_rviz.cpp)
target_link_libraries(obs_sphere_rviz
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

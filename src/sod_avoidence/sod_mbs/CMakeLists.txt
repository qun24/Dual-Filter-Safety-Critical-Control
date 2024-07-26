cmake_minimum_required(VERSION 3.0.2)
project(sod_mbs)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
  pcl_ros
  pcl_conversions
  message_filters
)

find_package(OpenCV REQUIRED)

# Find RealSense2
find_package(realsense2 REQUIRED)

# Find JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs cv_bridge image_transport sensor_msgs pcl_ros pcl_conversions
  LIBRARIES SODMBSLib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)

add_library(SODMBSLib
  src/MBS.cpp
  src/MBS.hpp
)

target_link_libraries(SODMBSLib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(analyse_dob_use_depth src/analyse_dob_use_depth.cpp)
target_link_libraries(analyse_dob_use_depth
  SODMBSLib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${realsense2_LIBRARY}
  ${JSONCPP_LIBRARIES}
)
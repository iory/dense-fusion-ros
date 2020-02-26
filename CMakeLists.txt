cmake_minimum_required(VERSION 2.8.3)
project(dense_fusion_ros)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(CATKIN_DEPENDS)

# download and install sample data
add_custom_target(${PROJECT_NAME}_install_sample_data
  ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_sample_data.py)

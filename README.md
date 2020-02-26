# Dense Fusion ROS Package

This is ROS Wrapper for Dense Fusion.

![](docs/image/title.gif)

## Prerequisite

Please Install ROS.


Install dense_fusion pyton library. You need to install pytorch.

```
pip install dense-fusion
```

## Install

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
cd ~/catkin_ws/src
git clone https://github.com/iory/dense-fusion-ros.git
cd ~/catkin_ws
rosdep install -y -r --from-paths src --ignore-src src
catkin build
source ~/catkin_ws/devel/setup.bash
```

## Sample

```
roslaunch dense_fusion_ros sample_dense_fusion.launch gpu:=0
```

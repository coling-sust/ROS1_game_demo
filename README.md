# ROS1_game_demo
一个基于ros1（wsl2平台，Ubuntu20.04）+yolo11s+PaddleOCR的仿真项目
启动方式：终端1运行 roslaunch game_demo car_gazebo.launch 启动gazebo地图仿真
终端2运行 roslaunch game_demo 06_include.launch 启动rviz导航图和导航插件
终端3运行 rosrun game_demo nav_copy02.py 启动小车进行导航（注意chmod +x添加可执行权限）

！！！注意，本项目所用依赖可能会跟你自己的库有冲突，建议重开一个虚拟环境，一定注意甄别（来自本人的血泪教训）

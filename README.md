# about ROS1+yolo11+PaddleOCR's simulation demo 
一个基于ros1（wsl2平台，Ubuntu20.04）+yolo11s+PaddleOCR的仿真项目，主要功能是实现小车的自主导航与避障，同时进行目标检测与文字识别  
(注意因为使用的WSL2虚拟平台，有GPU直通，所以我的脚本代码中使用了tensorRT部署加速推理，若是虚拟机可能需要去除.engine格式权重换成.pt或者onnx兼容格式--这两个格式我在yolo_weight文件夹中都有提供)  
启动方式:  
终端1运行 roslaunch game_demo car_gazebo.launch 启动gazebo地图仿真  
终端2运行 roslaunch game_demo 06_include.launch 启动rviz导航图和导航插件  
终端3运行 rosrun game_demo nav_copy02.py 启动小车进行导航（注意chmod +x添加可执行权限）  

  本项目使用的本地规划算法为TEB，在param的yaml文件中有详细参数定义，地图为使用blender建成（gazebo中无法修改，只能添加）  
  python版本3.8.10，cuda版本12.4，torch2.4.1，paddleOCR版本2.6.1.0  
  ！！！注意，本项目所用依赖可能会跟你自己的库有冲突，建议重开一个虚拟环境，一定注意甄别（来自本人的血泪教训）

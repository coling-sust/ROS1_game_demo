#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraViewer:
    def __init__(self):
        rospy.init_node('camera_viewer', anonymous=True)

        # 用于 ROS 图像转 OpenCV 图像
        self.bridge = CvBridge()
        self.latest_image = None

        # 订阅摄像头话题
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        # 创建 OpenCV 窗口
        cv2.namedWindow("Camera Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Window", 840, 630)

    def image_callback(self, msg):
        """接收 ROS 图像"""
        self.latest_image = msg

    def run(self):
        rate = rospy.Rate(30)  # 每秒 30 帧
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                try:
                    # ROS 图像转 OpenCV BGR
                    frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    frame_resized = cv2.resize(frame, (800, 600)) 
                    cv2.imshow("Camera Window", frame_resized)
                    cv2.waitKey(1)  # 必须加，保证窗口刷新
                except Exception as e:
                    rospy.logerr(f"Error converting ROS image: {str(e)}")
            rate.sleep()

        # 退出时释放窗口
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = CameraViewer()
    try:
        viewer.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

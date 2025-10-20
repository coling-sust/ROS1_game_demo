#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS + YOLO + OCR 智能导航系统
------------------------------------------------
该节点通过 YOLO（基于 TensorRT 加速）进行实时目标检测，
并结合 PaddleOCR 实现车牌识别功能。

功能概述：
1. 自动导航至预设航点；
2. 到达检测点时调用 YOLO 检测；
3. 若检测到车牌，则调用 OCR 模型识别车牌号码；
4. 实时显示检测画面与识别结果；
5. 将识别到的车牌号发布到 ROS 话题。
------------------------------------------------
"""

import rospy
import actionlib
import tf
import os
import time
import cv2
import numpy as np
import logging
import paddle
from PIL import Image, ImageDraw, ImageFont
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from paddleocr import PaddleOCR

class NavWithYOLOandOCR:
    """ROS + YOLO + OCR 导航类"""

    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("nav_with_yolo_ocr")

        # ================= YOLO 初始化 =================
        engine_path = "/home/xyxy/game_ws/src/game_demo/scripts/yolo_weight/yolo11s_game.engine"
        rospy.loginfo(f"Loading YOLO TensorRT engine from {engine_path} ...")
        # 加载 TensorRT 引擎加速版 YOLO 模型
        self.model = YOLO(engine_path, task="detect")
        self.infer_device = 0  # GPU 设备号

        # ================= PaddleOCR 初始化 =================
        rospy.loginfo("正在初始化 PaddleOCR with TensorRT ...")
        # 关闭 PaddleOCR 的冗余日志输出
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        # 初始化 OCR 模型（支持中文）
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 文字方向分类（竖排/横排）
            lang="ch",           # 中文识别
            use_gpu=True,        # 使用 GPU 加速
            show_log=False
        )

        # ================= 字体加载 =================
        # 尝试加载多种系统字体，支持中文显示（避免显示乱码）
        try:
            font_paths = [
                os.path.expanduser("~/.fonts/wqy-zenhei.ttc"),
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/PingFang.ttc",  
            ]
            self.font_path = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font_path = font_path
                    rospy.loginfo(f"Loaded Chinese font: {font_path}")
                    break
            if self.font_path is None:
                rospy.logwarn("No Chinese font found, using default PIL font")
                self.font = ImageFont.load_default()
        except Exception as e:
            rospy.logerr(f"Error loading font: {e}")
            self.font = ImageFont.load_default()

        # ================= ROS 通信组件 =================
        # move_base 动作客户端，用于导航控制
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base server.")

        # 摄像头图像接收
        self.bridge = CvBridge()
        self.latest_image = None
        rospy.Subscriber("/camera/image_raw", RosImage, self.image_callback)

        # ocr识别结果发布话题
        self.plate_pub = rospy.Publisher("/license_plate_number", String, queue_size=10)
        self.signs_pub = rospy.Publisher("/license_sign_txt", String, queue_size=10)
        self.txts_pub = rospy.Publisher("/license_txts_txt", String, queue_size=10)
        # ================= 航点配置 =================
        # 每个航点格式：(x, y, yaw, 是否执行YOLO检测, 是否执行OCR)
        self.waypoints = [
            (5.715, 6.042, 1.57, True, True),  # 检测点: 仅YOLO检测
            (7.667, 7.562, -3.14, True, True), # 检测点: 仅YOLO检测
            (5.691, 5.617, -1.57, True, True),   # 检测点: YOLO检测+OCR识别
            (4.559, 6.131, 1.18, True, True)   
        ]

        # ================= 显示窗口初始化 =================
        cv2.namedWindow("camera window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera window", 640, 480)

        # ================= 检测结果保存路径 =================
        self.save_dir = "/home/xyxy/game_ws/src/game_demo/scripts/yolo_photos"
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo(f"Detection results will be saved to: {self.save_dir}")

    # -------------------------------------------------------------------------
    def cv2_add_chinese_text(self, img, text, position, font_size=32,
                             text_color=(255, 255, 255), bg_color=None):
        """在OpenCV图像上绘制中文文字"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试加载字体
        if hasattr(self, 'font_path') and self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except Exception as e:
                rospy.logwarn(f"Failed to load font: {e}, using default font")
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # 计算文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 背景矩形（可选）
        if bg_color is not None:
            draw.rectangle(
                [position[0], position[1],
                 position[0] + text_width, position[1] + text_height],
                fill=bg_color
            )

        # 绘制文字
        draw.text(position, text, font=font, fill=text_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        """接收摄像头图像回调"""
        self.latest_image = msg

    # -------------------------------------------------------------------------
    def move_to_goal(self, x, y, yaw):
        """发送导航目标点"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]

        # 发送导航目标
        self.client.send_goal(goal)
        rate = rospy.Rate(10)
        rospy.loginfo(f"Moving to waypoint: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f})")

        # 循环检测任务状态
        while not rospy.is_shutdown():
            state = self.client.get_state()
            # 3=SUCCEEDED, 4=ABORTED 等状态
            if state in [3, 4, 5, 9]:
                break

            # 实时显示摄像头画面
            if self.latest_image is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    cv2.imshow("camera window", cv_image)
                    cv2.waitKey(1)
                except Exception as e:
                    rospy.logerr(f"Error displaying camera image: {str(e)}")
            rate.sleep()

    # -------------------------------------------------------------------------
    def recognize_roi(self, roi_image):
        """OCR识别区域"""
        try:
            result = self.ocr.ocr(roi_image, cls=True)
            if result and result[0]:
                texts = []
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.6:  # 置信度过滤
                        texts.append(text)
                if texts:
                    return ''.join(texts)
        except Exception as e:
            rospy.logerr(f"OCR recognition error: {str(e)}")
        return None

    # -------------------------------------------------------------------------
    def draw_detections_with_ocr(self, image, results, enable_ocr=False):
        """绘制YOLO检测结果 + OCR识别"""
        # 类别标签映射
        label_map = {1: "sign", 2: "txt", 4: "Plate"}
        color_map = {1: (0, 165, 255), 2: (0, 255, 255), 4: (255, 0, 0)}

        annotated_image = image.copy()
        detected_plates = []
        detected_signs = []
        detected_txts = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = label_map.get(cls, f"Class_{cls}")
            color = color_map.get(cls, (255, 255, 255))

            plate_number = None
            sign_number = None
            txt_number = None
            # 若启用OCR，且类别为车牌（class=4）
            if enable_ocr and cls == 4:
                h, w = image.shape[:2]
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(w, x2)
                y2_crop = min(h, y2)
                plate_roi = image[y1_crop:y2_crop, x1_crop:x2_crop]

                # OCR识别车牌区域
                if plate_roi.size > 0:
                    plate_number = self.recognize_roi(plate_roi)
                    if plate_number:
                        detected_plates.append(plate_number)
                        # rospy.loginfo(f"Detected license plate: {plate_number}")
                        self.plate_pub.publish(String(data=plate_number))
            # 若类别为sign（class=1）
            elif enable_ocr and cls == 1:
                h, w = image.shape[:2]
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(w, x2)
                y2_crop = min(h, y2)
                sign_roi = image[y1_crop:y2_crop, x1_crop:x2_crop]

                # OCR识别标识牌区域
                if sign_roi.size > 0:
                    sign_number = self.recognize_roi(sign_roi)
                    if sign_number:
                        detected_signs.append(sign_number)
                        self.signs_pub.publish(String(data=sign_number))
            #若类别为txt（class=2）
            elif enable_ocr and cls == 2:
                h, w = image.shape[:2]
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(w, x2)
                y2_crop = min(h, y2)
                txt_roi = image[y1_crop:y2_crop, x1_crop:x2_crop]

                # OCR识别文本区域
                if txt_roi.size > 0:
                    txt_number = self.recognize_roi(txt_roi)
                    if txt_number:
                        detected_txts.append(txt_number)
                        self.txts_pub.publish(String(data=txt_number))

            # 绘制检测框与标签
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            text = f"{label} {conf:.2f}" + (f" | {plate_number}" if plate_number else "")
            annotated_image = self.cv2_add_chinese_text(
                annotated_image, text, (x1, max(0, y1 - 40)),
                font_size=28, text_color=(255, 255, 255), bg_color=color
            )

        return annotated_image, detected_plates, detected_signs, detected_txts

    # -------------------------------------------------------------------------
    def run_yolo_with_ocr(self, index, enable_ocr=False):
        """在当前航点执行YOLO检测（可选OCR识别）"""
        if self.latest_image is None:
            rospy.logwarn("No camera image received, skipping detection.")
            return

        detect_duration = 8.0  # 每个检测点持续检测时长
        start_time = time.time()
        frame_count = 0
        latest_annotated = None
        all_detected_plates = set()
        all_detected_signs = set()
        all_detected_txts = set()

        rospy.loginfo(f"=== Starting {'YOLO + OCR' if enable_ocr else 'YOLO'} detection at waypoint {index+1} ===")

        while time.time() - start_time < detect_duration:
            if self.latest_image is None:
                continue

            try:
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            except Exception as e:
                rospy.logerr(f"Error converting ROS image: {str(e)}")
                continue

            # YOLO 推理
            results = self.model.predict(
                source=cv_image,
                conf=0.5,
                imgsz=640,
                device=self.infer_device,
                verbose=False
            )

            # 绘制检测框+OCR结果
            latest_annotated, detected_plates, detected_signs, detected_txts = self.draw_detections_with_ocr(cv_image, results, enable_ocr)
            all_detected_plates.update(detected_plates)
            all_detected_signs.update(detected_signs)
            all_detected_txts.update(detected_txts)

            # 计算 FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            # 在图像下方显示 FPS
            h, w, _ = latest_annotated.shape
            cv2.putText(latest_annotated, f"FPS: {fps:.1f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)


            h, w, _ = latest_annotated.shape   # 图像高和宽
            line_height = 50  # 每行文字间距
            num_lines = len(all_detected_plates) + len(all_detected_signs) + len(all_detected_txts)
            # 初始 y 坐标：从图像高度 2/3 开始显示
            y_offset = int(h * 2 / 3)
            # 如果文字太多，可确保不超出底部
            y_offset = max(line_height, min(y_offset, h - line_height * num_lines))

            # 显示车牌
            if enable_ocr and all_detected_plates:
                for plate in all_detected_plates:
                    latest_annotated = self.cv2_add_chinese_text(
                        latest_annotated, f"车牌: {plate}", (10, y_offset),
                        font_size=36, text_color=(0, 255, 0), bg_color=(0, 0, 0)
                    )
                    y_offset += line_height

            # 显示标识牌
            if enable_ocr and all_detected_signs:
                for sign in all_detected_signs:
                    latest_annotated = self.cv2_add_chinese_text(
                        latest_annotated, f"标识牌: {sign}", (10, y_offset),
                        font_size=36, text_color=(255, 165, 0), bg_color=(0, 0, 0)
                    )
                    y_offset += line_height

            # 显示文本
            if enable_ocr and all_detected_txts:
                for txt in all_detected_txts:
                    latest_annotated = self.cv2_add_chinese_text(
                        latest_annotated, f"文本: {txt}", (10, y_offset),
                        font_size=36, text_color=(0, 0, 255), bg_color=(0, 0, 0)
                    )
                    y_offset += line_height


            # 实时显示
            cv2.imshow("camera window", latest_annotated)
            cv2.waitKey(1)

        # 保存检测结果图像
        if latest_annotated is not None:
            save_path = os.path.join(self.save_dir, f"waypoint_{index+1}_detection.jpg")
            cv2.imwrite(save_path, latest_annotated)
            rospy.loginfo(f"Detection result saved: {save_path}")

        if enable_ocr and all_detected_plates:
            rospy.loginfo(f"All detected plates at waypoint {index+1}: {list(all_detected_plates)}")

        rospy.loginfo(f"=== Detection at waypoint {index+1} completed ===")

    # -------------------------------------------------------------------------
    def run(self):
        """主循环：依次前往各航点并执行检测任务"""
        rospy.loginfo("Starting autonomous navigation with YOLO + OCR detection...")
        for idx, (x, y, yaw, do_yolo, do_ocr) in enumerate(self.waypoints):
            self.move_to_goal(x, y, yaw)  # 导航到目标点
            if do_yolo:
                rospy.sleep(0.5)
                self.run_yolo_with_ocr(idx, enable_ocr=do_ocr)
            rospy.sleep(1.0)
        rospy.loginfo("All waypoints completed! Autonomous navigation finished.")
        cv2.destroyAllWindows()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 修复 numpy 1.24 版本移除 np.int 的兼容问题
    if not hasattr(np, "int"):
        np.int = int
    try:
        nav_yolo_ocr = NavWithYOLOandOCR()
        nav_yolo_ocr.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

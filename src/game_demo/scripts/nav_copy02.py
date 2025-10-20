#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS + YOLO + OCR 智能导航系统（改进版）
------------------------------------------------
新增功能：
1. OCR 结果稳定性过滤（避免跳变）
2. 优化显示逻辑（只显示稳定的结果）
3. 自动去重和置信度统计
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
from collections import Counter, deque
from PIL import Image, ImageDraw, ImageFont
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ================= 稳定性过滤器 =================
class OCRStabilityFilter:
    """OCR识别结果稳定性过滤器"""
    
    def __init__(self, buffer_size=8, min_confidence=0.6):
        """
        参数:
            buffer_size: 缓冲区大小（保存最近N帧的识别结果）
            min_confidence: 最小置信度（某个结果出现的比例阈值）
        """
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        
        # 为不同类别维护独立的缓冲区
        self.plate_buffer = deque(maxlen=buffer_size)
        self.sign_buffer = deque(maxlen=buffer_size)
        self.txt_buffer = deque(maxlen=buffer_size)
        
    def add_result(self, result_type, text):
        """添加识别结果到缓冲区"""
        if text is None or text == "":
            return
            
        if result_type == "plate":
            self.plate_buffer.append(text)
        elif result_type == "sign":
            self.sign_buffer.append(text)
        elif result_type == "txt":
            self.txt_buffer.append(text)
    
    def get_stable_results(self, result_type):
        """获取稳定的识别结果（返回所有符合条件的结果）"""
        # 选择对应的缓冲区
        if result_type == "plate":
            buffer = self.plate_buffer
        elif result_type == "sign":
            buffer = self.sign_buffer
        elif result_type == "txt":
            buffer = self.txt_buffer
        else:
            return []
        
        # 缓冲区数据不足
        if len(buffer) < max(3, self.buffer_size * 0.4):  # 至少需要3帧或40%的数据
            return []
        
        # 统计各结果出现次数
        counter = Counter(buffer)
        if not counter:
            return []
        
        # 获取所有置信度足够高的结果
        stable_results = []
        for text, count in counter.items():
            confidence = count / len(buffer)
            if confidence >= self.min_confidence:
                stable_results.append((text, confidence))
        
        # 按置信度排序（降序）
        stable_results.sort(key=lambda x: x[1], reverse=True)
        return stable_results
    
    def clear_buffer(self, result_type=None):
        """清空缓冲区"""
        if result_type == "plate" or result_type is None:
            self.plate_buffer.clear()
        if result_type == "sign" or result_type is None:
            self.sign_buffer.clear()
        if result_type == "txt" or result_type is None:
            self.txt_buffer.clear()


# ================= 主导航类 =================
class NavWithYOLOandOCR:
    """ROS + YOLO + OCR 导航类"""

    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("nav_with_yolo_ocr")

        # ================= YOLO 初始化 =================
        engine_path = "/home/xyxy/game_ws/src/game_demo/scripts/yolo_weight/yolo11s_game.engine"
        rospy.loginfo(f"Loading YOLO TensorRT engine from {engine_path} ...")
        self.model = YOLO(engine_path, task="detect")
        self.infer_device = 0

        # ================= PaddleOCR 初始化 =================
        rospy.loginfo("正在初始化 PaddleOCR with TensorRT ...")
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=True,
            show_log=False
        )

        # ================= 稳定性过滤器初始化 =================
        self.stability_filter = OCRStabilityFilter(
            buffer_size=8,       # 缓冲8帧
            min_confidence=0.65  # 至少65%的帧识别为同一结果才显示
        )

        # ================= 字体加载 =================
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
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base server.")

        self.bridge = CvBridge()
        self.latest_image = None
        rospy.Subscriber("/camera/image_raw", RosImage, self.image_callback)

        # OCR识别结果发布话题
        self.plate_pub = rospy.Publisher("/license_plate_number", String, queue_size=10)
        self.signs_pub = rospy.Publisher("/license_sign_txt", String, queue_size=10)
        self.txts_pub = rospy.Publisher("/license_txts_txt", String, queue_size=10)
        
        # ================= 航点配置 =================
        self.waypoints = [
            (7.81, 5.11, 2.35, False, False),
            (5.815, 5.8, 1.57, True, False), 
            (7.267, 7.1, 0, True, True), 
            (5.291, 5.617, -1.57, True, False),
            (4.93, 4.67, -1.57, True, True),
            (4.559, 6.131, -2.35, True, False), 
            (0, 0, 0, False, False)  
        ]

        # ================= 显示窗口初始化 =================
        cv2.namedWindow("camera window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera window", 960, 720)

        # ================= 检测结果保存路径 =================
        self.save_dir = "/home/xyxy/game_ws/src/game_demo/scripts/yolo_photos"
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo(f"Detection results will be saved to: {self.save_dir}")

    # -------------------------------------------------------------------------
    def cv2_add_chinese_text(self, img, text, position, font_size=26,
                             text_color=(255, 255, 255), bg_color=None):
        """在OpenCV图像上绘制中文文字"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        if hasattr(self, 'font_path') and self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except Exception as e:
                rospy.logwarn(f"Failed to load font: {e}, using default font")
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 绘制半透明背景（可选）
        if bg_color is not None:
            overlay = img_pil.copy()
            draw_overlay = ImageDraw.Draw(overlay)
            padding = 5
            draw_overlay.rectangle(
                [position[0] - padding, position[1] - padding,
                 position[0] + text_width + padding, position[1] + text_height + padding],
                fill=bg_color
            )
            img_pil = Image.blend(img_pil, overlay, 0.4)  # 40%透明度
            draw = ImageDraw.Draw(img_pil)

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

        self.client.send_goal(goal)
        rate = rospy.Rate(10)
        rospy.loginfo(f"Moving to waypoint: (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f})")

        while not rospy.is_shutdown():
            state = self.client.get_state()
            if state in [3, 4, 5, 9]:
                break

            if self.latest_image is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                    resize_fram = cv2.resize(cv_image, (960, 720), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("camera window", resize_fram)
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
        """绘制YOLO检测结果 + OCR识别（带稳定性过滤）"""
        label_map = {
            0: "child",      
            1: "sign",
            2: "txt",
            3: "car",
            4: "Plate",
            5: "person",
        }
        
        color_map = {
            0: (0, 255, 0),      # 绿色
            1: (0, 165, 255),    # 橙色
            2: (0, 255, 255),    # 黄色
            3: (255, 255, 0),    # 青色
            4: (255, 0, 0),      # 蓝色
            5: (0, 255, 127),    # 春绿色
        }

        annotated_image = image.copy()
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = label_map.get(cls, f"Class_{cls}")
            color = color_map.get(cls, (128, 128, 255))

            ocr_text = None
            
            # OCR识别逻辑
            if enable_ocr and cls in [1, 2, 4]:  # sign, txt, plate
                h, w = image.shape[:2]
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(w, x2)
                y2_crop = min(h, y2)
                roi = image[y1_crop:y2_crop, x1_crop:x2_crop]

                if roi.size > 0:
                    ocr_text = self.recognize_roi(roi)
                    if ocr_text:
                        ocr_text = ocr_text.strip()
                        # 添加到稳定性过滤器
                        if cls == 4:
                            self.stability_filter.add_result("plate", ocr_text)
                        elif cls == 1:
                            self.stability_filter.add_result("sign", ocr_text)
                        elif cls == 2:
                            self.stability_filter.add_result("txt", ocr_text)

            # 绘制检测框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签（不显示OCR原始结果，避免跳变）
            text = f"{label} {conf:.2f}"
            annotated_image = self.cv2_add_chinese_text(
                annotated_image, text, (x1, max(0, y1 - 30)),
                font_size=16, text_color=(255, 0, 255)
            )

        return annotated_image

    # -------------------------------------------------------------------------
    def run_yolo_with_ocr(self, index, enable_ocr=False):
        """在当前航点执行YOLO检测（可选OCR识别）"""
        if self.latest_image is None:
            rospy.logwarn("No camera image received, skipping detection.")
            return

        detect_duration = 8.0
        start_time = time.time()
        frame_count = 0
        latest_annotated = None

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

            # 绘制检测框+OCR结果收集
            latest_annotated = self.draw_detections_with_ocr(cv_image, results, enable_ocr)

            # 计算 FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            h, w, _ = latest_annotated.shape
            cv2.putText(latest_annotated, f"FPS: {fps:.1f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            # ================= 显示稳定的OCR结果 =================
            if enable_ocr:
                y_offset = int(h * 0.65)  # 从65%高度开始显示
                line_height = 50
                
                # 获取稳定的车牌结果
                stable_plates = self.stability_filter.get_stable_results("plate")
                if stable_plates:
                    for plate_text, confidence in stable_plates:
                        # 去除第一个字符（如果存在）
                        display_plate = plate_text[1:] if len(plate_text) > 1 else plate_text
                        display_text = f"车牌: {display_plate} ({confidence*100:.0f}%)"
                        latest_annotated = self.cv2_add_chinese_text(
                            latest_annotated, display_text, (10, y_offset),
                            font_size=32, text_color=(0, 255, 0), bg_color=(0, 0, 0)
                        )
                        y_offset += line_height
                        # 发布到ROS话题（只发布最稳定的，也去除第一个字符）
                        if confidence > 0.6:
                            self.plate_pub.publish(String(data=display_plate))
                
                # 获取稳定的标识牌结果
                stable_signs = self.stability_filter.get_stable_results("sign")
                if stable_signs:
                    for sign_text, confidence in stable_signs:
                        display_text = f"标识牌: {sign_text} ({confidence*100:.0f}%)"
                        latest_annotated = self.cv2_add_chinese_text(
                            latest_annotated, display_text, (10, y_offset),
                            font_size=32, text_color=(255, 165, 0), bg_color=(0, 0, 0)
                        )
                        y_offset += line_height
                        if confidence > 0.5:
                            self.signs_pub.publish(String(data=sign_text))
                
                # 获取稳定的文本结果
                stable_txts = self.stability_filter.get_stable_results("txt")
                if stable_txts:
                    for txt_text, confidence in stable_txts:
                        display_text = f"文本: {txt_text} ({confidence*100:.0f}%)"
                        latest_annotated = self.cv2_add_chinese_text(
                            latest_annotated, display_text, (10, y_offset),
                            font_size=32, text_color=(255, 255, 255), bg_color=(0, 0, 0)
                        )
                        y_offset += line_height
                        if confidence > 0.6:
                            self.txts_pub.publish(String(data=txt_text))

            # 实时显示
            resize_fram = cv2.resize(latest_annotated, (960, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("camera window", resize_fram)
            cv2.waitKey(1)

        # 保存检测结果图像
        if latest_annotated is not None:
            save_path = os.path.join(self.save_dir, f"waypoint_{index+1}_detection.jpg")
            cv2.imwrite(save_path, latest_annotated)
            rospy.loginfo(f"Detection result saved: {save_path}")

        # 打印最终稳定结果
        if enable_ocr:
            final_plates = self.stability_filter.get_stable_results("plate")
            final_signs = self.stability_filter.get_stable_results("sign")
            final_txts = self.stability_filter.get_stable_results("txt")
            
            if final_plates:
                rospy.loginfo(f"Stable plates at waypoint {index+1}: {[p[0] for p in final_plates]}")
            if final_signs:
                rospy.loginfo(f"Stable signs at waypoint {index+1}: {[s[0] for s in final_signs]}")
            if final_txts:
                rospy.loginfo(f"Stable texts at waypoint {index+1}: {[t[0] for t in final_txts]}")

        rospy.loginfo(f"=== Detection at waypoint {index+1} completed ===")

    # -------------------------------------------------------------------------
    def run(self):
        """主循环：依次前往各航点并执行检测任务"""
        rospy.loginfo("Starting autonomous navigation with YOLO + OCR detection...")
        for idx, (x, y, yaw, do_yolo, do_ocr) in enumerate(self.waypoints):
            # 每个航点开始前清空缓冲区
            self.stability_filter.clear_buffer()
            
            self.move_to_goal(x, y, yaw)
            if do_yolo:
                rospy.sleep(0.5)
                self.run_yolo_with_ocr(idx, enable_ocr=do_ocr)
            rospy.sleep(1.0)
        rospy.loginfo("All waypoints completed! Autonomous navigation finished.")
        cv2.destroyAllWindows()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    if not hasattr(np, "int"):
        np.int = int
    try:
        nav_yolo_ocr = NavWithYOLOandOCR()
        nav_yolo_ocr.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
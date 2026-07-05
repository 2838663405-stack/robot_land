#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pickle

class InsightFaceRecognizer(Node):
    def __init__(self):
        super().__init__('insightface_recognizer')
        
        # --- 加载队员特征数据库 ---
        try:
            with open('face_embeddings.pkl', 'rb') as f:
                self.face_db = pickle.load(f)
            self.get_logger().info(f"成功加载特征库，共 {len(self.face_db)} 人。")
            for person in self.face_db:
                self.get_logger().info(f"  - {person['name']}")
        except FileNotFoundError:
            self.get_logger().error("找不到 face_embeddings.pkl 文件！请先运行 generate_embeddings.py")
            rclpy.shutdown()
            return

        # --- 初始化InsightFace ---
        # providers=['CPUExecutionProvider'] 强制使用CPU，避免GPU驱动问题
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.bridge = CvBridge()

        # --- 订阅摄像头话题 ---
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        # --- 防抖与冷却逻辑（保留原貌）---
        self.last_name = None
        self.confirm_count = 0
        self.cooldown_until = 0
        self.CONFIRM_FRAMES = 5
        self.COOLDOWN_SEC = 20

        self.get_logger().info("InsightFace识别节点已启动，正在等待图像...")

    def image_callback(self, msg):
        import time
        current_time = time.time()

        # 1. 将ROS图像消息转为OpenCV格式
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 2. 用InsightFace检测人脸并提取特征
        #    faces 是一个列表，包含每个检测到的人脸信息，如bbox, embedding等[citation:4]
        faces = self.app.get(frame)
        
        recognized_name = None
        # 3. 对检测到的第一个人脸进行识别
        if len(faces) > 0 and current_time >= self.cooldown_until:
            face = faces[0]  # 取画面中最显著的人脸
            query_embedding = face.embedding
            
            # 4. 与数据库中的特征进行比对（计算余弦相似度）
            max_sim = -1
            for person in self.face_db:
                db_embedding = person['embedding']
                # 余弦相似度 = (A·B) / (||A|| * ||B||) ，值越接近1越相似[citation:6]
                sim = np.dot(query_embedding, db_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
                if sim > max_sim:
                    max_sim = sim
                    recognized_name = person['name']
            
            # 设置一个相似度阈值，比如大于0.4就认为是匹配[citation:6]
            if max_sim < 0.4:
                recognized_name = None

            # 5. 防抖与触发逻辑
            if recognized_name:
                if recognized_name == self.last_name:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 1
                    self.last_name = recognized_name

                if self.confirm_count >= self.CONFIRM_FRAMES:
                    self.get_logger().info(f"🎉 InsightFace识别到 {recognized_name}！ (相似度: {max_sim:.2f})")
                    # 这里可以调用你之前的say_hello()函数
                    self.cooldown_until = current_time + self.COOLDOWN_SEC
                    self.confirm_count = 0
            else:
                # 如果没认出，重置防抖
                self.confirm_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = InsightFaceRecognizer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

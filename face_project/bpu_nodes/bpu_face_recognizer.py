#!/usr/bin/env python3
"""
BPU 人脸识别 ROS2 节点
使用 hrt_model_exec 在 BPU 上进行人脸检测和识别
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import pickle
import time
from bpu_infer import BPUInfer

class BPUFaceRecognizer(Node):
    def __init__(self):
        super().__init__('bpu_face_recognizer')
        
        # ---------- 加载特征数据库 ----------
        self.db_path = os.path.expanduser('~/face_project/bpu_nodes/data/face_embeddings.pkl')
        self.face_db = []
        self._load_database()
        
        # ---------- 初始化 BPU 推理器 ----------
        self.get_logger().info('正在加载 BPU 模型...')
        try:
            # 人脸检测模型
            self.det_model = BPUInfer('/app/models/det_10g.bin')
            # 人脸识别模型
            self.rec_model = BPUInfer('/app/models/w600k_r50.bin')
            self.get_logger().info('✅ BPU 模型加载成功')
        except Exception as e:
            self.get_logger().error(f'❌ BPU 模型加载失败: {e}')
            rclpy.shutdown()
            return
        
        self.bridge = CvBridge()
        
        # ---------- 订阅摄像头 ----------
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # ---------- 防抖与冷却 ----------
        self.last_name = None
        self.confirm_count = 0
        self.cooldown_until = 0
        self.CONFIRM_FRAMES = 5
        self.COOLDOWN_SEC = 20
        
        self.get_logger().info('✅ BPU 人脸识别节点已启动')
        self.get_logger().info(f'📁 特征数据库: {len(self.face_db)} 人')
        for person in self.face_db:
            self.get_logger().info(f'  - {person["name"]}')
    
    def _load_database(self):
        """加载特征数据库"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.face_db = pickle.load(f)
                self.get_logger().info(f'✅ 成功加载特征库，共 {len(self.face_db)} 人')
            except Exception as e:
                self.get_logger().error(f'❌ 加载特征库失败: {e}')
                self.face_db = []
        else:
            self.get_logger().warning(f'⚠️ 特征库不存在: {self.db_path}')
            self.get_logger().warning('请先运行 generate_embeddings.py 生成特征库')
    
    def preprocess_for_detection(self, frame):
        """预处理人脸检测输入"""
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        return img
    
    def detect_faces(self, frame):
        """
        检测人脸
        Returns:
            list: 人脸框列表 [(x1, y1, x2, y2), ...]
        """
        # 保存预处理后的图片
        temp_img = '/tmp/det_input.jpg'
        cv2.imwrite(temp_img, frame)
        
        try:
            # 调用 BPU 推理（det_10g 输出需要解析）
            # 目前先用简单方式：直接用识别模型检测
            # TODO: 解析 det_10g 的输出
            pass
        except Exception as e:
            self.get_logger().debug(f'检测失败: {e}')
        
        # 简化版：假设已经检测到人脸，直接返回整张图
        # 实际应该解析 det_10g 的输出
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]
    
    def extract_feature(self, face_crop):
        """
        提取人脸特征
        Args:
            face_crop: 裁剪后的人脸图像 (BGR)
        Returns:
            numpy array: 特征向量
        """
        # 保存临时图片
        temp_img = '/tmp/face_crop.jpg'
        cv2.imwrite(temp_img, face_crop)
        
        try:
            feature = self.rec_model.infer_from_file(temp_img)
            return feature
        except Exception as e:
            self.get_logger().error(f'特征提取失败: {e}')
            return None
    
    def recognize_face(self, feature):
        """
        识别特征向量
        Returns:
            str: 姓名，None 表示未识别
        """
        if feature is None or len(self.face_db) == 0:
            return None
        
        max_sim = -1
        recognized_name = None
        
        for person in self.face_db:
            db_feature = person['embedding']
            # 余弦相似度
            sim = np.dot(feature, db_feature) / (
                np.linalg.norm(feature) * np.linalg.norm(db_feature) + 1e-8
            )
            if sim > max_sim:
                max_sim = sim
                recognized_name = person['name']
        
        # 相似度阈值（0.4 可以调整）
        if max_sim < 0.4:
            return None
        
        return recognized_name
    
    def image_callback(self, msg):
        current_time = time.time()
        
        # 转换为 OpenCV 图像
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 冷却检查
        if current_time < self.cooldown_until:
            return
        
        # 提取人脸特征（简化版：直接使用整张图片）
        try:
            # 裁剪人脸区域（这里先用整张图，实际应该用 det_10g 检测）
            h, w = frame.shape[:2]
            face_crop = cv2.resize(frame, (112, 112))
            
            # 提取特征
            feature = self.extract_feature(face_crop)
            if feature is None:
                return
            
            # 识别
            name = self.recognize_face(feature)
            
            # 防抖逻辑
            if name:
                if name == self.last_name:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 1
                    self.last_name = name
                
                if self.confirm_count >= self.CONFIRM_FRAMES:
                    self.get_logger().info(f'🎉 识别到队员: {name}')
                    self.cooldown_until = current_time + self.COOLDOWN_SEC
                    self.confirm_count = 0
                    
                    # TODO: 在这里调用打招呼函数
                    # self.say_hello(name)
            else:
                self.confirm_count = 0
                
        except Exception as e:
            self.get_logger().error(f'处理失败: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = BPUFaceRecognizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

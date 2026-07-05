#!/usr/bin/env python3
"""
BPU 人脸识别 ROS2 节点（带容错防抖）
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
import subprocess
import tempfile
import shutil
from audio_player import AudioPlayer

class BPUFaceRecognizer(Node):
    def __init__(self):
        super().__init__('bpu_face_recognizer')
        
        # ---------- 加载特征数据库 ----------
        self.db_path = os.path.expanduser('~/face_project/bpu_nodes/data/face_embeddings.pkl')
        self.face_db = []
        self._load_database()
        
        # ---------- 初始化音频播放器 ----------
        self.audio = AudioPlayer(volume=200)
        
        # ---------- 初始化 BPU 识别模型 ----------
        self.get_logger().info('正在加载 BPU 识别模型...')
        self.rec_model_path = '/app/models/w600k_r50.bin'
        
        if not os.path.exists(self.rec_model_path):
            self.get_logger().error(f'❌ 识别模型不存在: {self.rec_model_path}')
            rclpy.shutdown()
            return
        
        self.get_logger().info('✅ BPU 识别模型加载成功')
        
        # ---------- 初始化 OpenCV 人脸检测器 ----------
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            self.get_logger().error('❌ OpenCV 人脸检测器加载失败')
            rclpy.shutdown()
            return
        self.get_logger().info('✅ OpenCV 人脸检测器加载成功')
        
        self.bridge = CvBridge()
        self.temp_dir = tempfile.mkdtemp(prefix='bpu_face_')
        
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
        self.fail_count = 0          # 连续失败计数
        self.cooldown_until = 0
        self.CONFIRM_FRAMES = 3      # 需要连续成功帧数
        self.MAX_FAILS = 3           # 允许连续失败帧数（容错）
        self.COOLDOWN_SEC = 20
        self.SIMILARITY_THRESHOLD = 0.5
        
        self.silent_count = 0
        self.SILENT_LOG_INTERVAL = 50
        
        self.get_logger().info('✅ BPU 人脸识别节点已启动（容错模式）')
        self.get_logger().info(f'📁 特征数据库: {len(self.face_db)} 人')
        self.get_logger().info(f'🎯 相似度阈值: {self.SIMILARITY_THRESHOLD}')
        self.get_logger().info(f'🔄 连续确认: {self.CONFIRM_FRAMES} 帧, 容错: {self.MAX_FAILS} 帧')
        for person in self.face_db:
            self.get_logger().info(f'  - {person["name"]}')
    
    def _load_database(self):
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
    
    def _run_hrt_model(self, model_path, input_file, output_dir):
        cmd = [
            'hrt_model_exec', 'infer',
            '--model_file', model_path,
            '--input_file', input_file,
            '--core_id', '0',
            '--enable_dump', 'true',
            '--dump_path', output_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    
    def _get_output_file(self, output_dir):
        for f in os.listdir(output_dir):
            if f.startswith('model_infer_output_') and f.endswith('.bin'):
                return os.path.join(output_dir, f)
        return None
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
        return boxes
    
    def extract_feature(self, face_crop):
        if face_crop.shape[0] != 112 or face_crop.shape[1] != 112:
            face_crop = cv2.resize(face_crop, (112, 112))
        
        temp_img = os.path.join(self.temp_dir, 'rec_input.jpg')
        cv2.imwrite(temp_img, face_crop)
        
        output_dir = os.path.join(self.temp_dir, f'rec_out_{int(time.time())}')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            result = self._run_hrt_model(self.rec_model_path, temp_img, output_dir)
            if result.returncode != 0:
                return None
            output_file = self._get_output_file(output_dir)
            if output_file is None:
                return None
            feature = np.fromfile(output_file, dtype=np.float32)
            return feature
        except Exception as e:
            return None
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            if os.path.exists(temp_img):
                os.remove(temp_img)
    
    def recognize_face(self, feature):
        if feature is None or len(self.face_db) == 0:
            return None, 0.0
        
        max_sim = -1.0
        recognized_name = None
        
        for person in self.face_db:
            db_feature = person['embedding']
            sim = np.dot(feature, db_feature) / (
                np.linalg.norm(feature) * np.linalg.norm(db_feature) + 1e-8
            )
            if sim > max_sim:
                max_sim = sim
                recognized_name = person['name']
        
        if max_sim < self.SIMILARITY_THRESHOLD:
            return None, max_sim
        
        return recognized_name, max_sim
    
    def image_callback(self, msg):
        current_time = time.time()
        
        if current_time < self.cooldown_until:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        try:
            # 1. 人脸检测
            boxes = self.detect_faces(frame)
            
            if len(boxes) == 0:
                self.silent_count += 1
                if self.silent_count % self.SILENT_LOG_INTERVAL == 0:
                    self.get_logger().info('📊 等待人脸...')
                
                # 🔥 关键改动：增加容错，不立即清零
                self.fail_count += 1
                if self.fail_count > self.MAX_FAILS:
                    # 连续失败超过阈值，才重置
                    self.confirm_count = 0
                    self.last_name = None
                return
            
            # 检测到人脸，重置失败计数
            self.fail_count = 0
            self.silent_count = 0
            
            # 2. 取第一个人脸
            x1, y1, x2, y2 = boxes[0]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return
            
            # 3. 提取特征
            feature = self.extract_feature(face_crop)
            if feature is None:
                return
            
            # 4. 识别
            name, max_sim = self.recognize_face(feature)
            
            # 5. 防抖逻辑
            if name:
                if name == self.last_name:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 1
                    self.last_name = name
                
                self.get_logger().info(f'📊 识别中: {name} ({self.confirm_count}/{self.CONFIRM_FRAMES})')
                
                if self.confirm_count >= self.CONFIRM_FRAMES:
                    self.get_logger().info(f'🎉 识别到队员: {name} (相似度: {max_sim:.3f})')
                    self.audio.say_hello(name)
                    self.cooldown_until = current_time + self.COOLDOWN_SEC
                    self.confirm_count = 0
                    self.last_name = None
            else:
                # 未识别到同一个人，重置但保留最后名字用于对比
                self.confirm_count = 0
                
        except Exception as e:
            self.get_logger().error(f'处理失败: {e}')
    
    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def main(args=None):
    rclpy.init(args=args)
    node = BPUFaceRecognizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

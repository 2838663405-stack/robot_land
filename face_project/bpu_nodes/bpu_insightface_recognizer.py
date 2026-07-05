#!/usr/bin/env python3
"""
BPU 人脸识别 ROS2 节点 - 极致优化版
使用运动检测 + 定时检测 大幅降低空闲CPU占用
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

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print('⚠️ InsightFace 未安装')

class BPUInsightFaceRecognizer(Node):
    def __init__(self):
        super().__init__('bpu_insightface_recognizer')
        
        if not INSIGHTFACE_AVAILABLE:
            self.get_logger().error('❌ InsightFace 未安装')
            rclpy.shutdown()
            return
        
        # ---------- 加载特征数据库 ----------
        self.db_path = os.path.expanduser('~/face_project/bpu_nodes/data/face_embeddings.pkl')
        self.face_db = []
        self._load_database()
        
        # ---------- 初始化音频播放器 ----------
        self.audio = AudioPlayer(volume=200)
        
        # ---------- 初始化 OpenCV 快速检测器 ----------
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.get_logger().info('✅ OpenCV 快速检测器加载成功')
        
        # ---------- 初始化 InsightFace ----------
        self.get_logger().info('正在初始化 InsightFace...')
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            self.get_logger().info('✅ InsightFace 初始化成功')
        except Exception as e:
            self.get_logger().error(f'❌ InsightFace 初始化失败: {e}')
            rclpy.shutdown()
            return
        
        # ---------- BPU 识别模型 ----------
        self.rec_model_path = '/app/models/w600k_r50.bin'
        if not os.path.exists(self.rec_model_path):
            self.get_logger().error(f'❌ 识别模型不存在: {self.rec_model_path}')
            rclpy.shutdown()
            return
        self.get_logger().info('✅ BPU 识别模型加载成功')
        
        self.bridge = CvBridge()
        
        # ---------- 运动检测 ----------
        self.prev_frame = None
        self.motion_threshold = 5000  # 运动阈值
        self.last_detection_time = 0
        self.DETECT_INTERVAL = 0.5    # 检测间隔 (秒)
        
        # ---------- 帧率控制 ----------
        self.frame_count = 0
        self.PROCESS_INTERVAL = 2
        
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
        self.CONFIRM_FRAMES = 1
        self.COOLDOWN_SEC = 15
        self.SIMILARITY_THRESHOLD = 0.4
        
        self.get_logger().info('✅ BPU+InsightFace 人脸识别节点已启动 (极致优化版)')
        self.get_logger().info(f'📁 特征数据库: {len(self.face_db)} 人')
        self.get_logger().info(f'⚡ 运动检测阈值: {self.motion_threshold}')
        self.get_logger().info(f'⏱️ 检测间隔: {self.DETECT_INTERVAL}s')
    
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
    
    def _run_bpu_inference(self, image_path):
        temp_dir = tempfile.mkdtemp(prefix='bpu_rec_')
        
        cmd = [
            'hrt_model_exec', 'infer',
            '--model_file', self.rec_model_path,
            '--input_file', image_path,
            '--core_id', '0',
            '--dump_intermediate', '1',
            '--dump_path', temp_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        feature = None
        if result.returncode == 0:
            for f in os.listdir(temp_dir):
                if f.startswith('layer-3-cpu-graph_output-output-0-683') and f.endswith('.data.bin'):
                    file_path = os.path.join(temp_dir, f)
                    if os.path.getsize(file_path) == 2048:
                        feature = np.fromfile(file_path, dtype=np.float32)
                        break
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        return feature
    
    def extract_feature_bpu(self, face_img):
        if face_img is None or face_img.size == 0:
            return None
        face_img = cv2.resize(face_img, (112, 112))
        temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_img.name, face_img)
        feature = self._run_bpu_inference(temp_img.name)
        os.unlink(temp_img.name)
        return feature
    
    def compare_features(self, feature1, feature2):
        if feature1 is None or feature2 is None:
            return 0.0
        return np.dot(feature1, feature2) / (
            np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-8
        )
    
    def detect_motion(self, frame):
        """检测画面是否有运动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        
        # 计算帧差
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # 更新背景
        self.prev_frame = gray
        
        # 计算运动区域
        motion_score = np.sum(thresh) / 255
        return motion_score > self.motion_threshold
    
    def image_callback(self, msg):
        current_time = time.time()
        
        if current_time < self.cooldown_until:
            return
        
        # 帧率控制
        self.frame_count += 1
        if self.frame_count % self.PROCESS_INTERVAL != 0:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # ===== 第1级：运动检测 (极低CPU) =====
        has_motion = self.detect_motion(frame)
        
        # 如果没有运动，跳过所有检测
        if not has_motion:
            # CPU 占用 < 1%
            return
        
        # 如果运动刚发生，等待一小段时间再检测 (避免频繁检测)
        if current_time - self.last_detection_time < self.DETECT_INTERVAL:
            return
        
        self.last_detection_time = current_time
        
        try:
            # ===== 第2级：OpenCV 快速预筛选 =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(80, 80)
            )
            
            if len(faces) == 0:
                return
            
            # ===== 第3级：InsightFace 精确检测 =====
            faces_insight = self.app.get(frame)
            
            recognized_name = None
            max_sim = 0.0
            
            if len(faces_insight) > 0:
                face = faces_insight[0]
                bbox = face.bbox.astype(np.int32)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                margin = 30
                h, w = frame.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    feature = self.extract_feature_bpu(face_crop)
                    
                    if feature is not None:
                        max_sim = -1.0
                        recognized_name = None
                        for person in self.face_db:
                            sim = self.compare_features(feature, person['embedding'])
                            if sim > max_sim:
                                max_sim = sim
                                recognized_name = person['name']
                        
                        if max_sim < self.SIMILARITY_THRESHOLD:
                            recognized_name = None
            
            # 防抖
            if recognized_name:
                if recognized_name == self.last_name:
                    self.confirm_count += 1
                else:
                    self.confirm_count = 1
                    self.last_name = recognized_name
                
                if self.confirm_count >= self.CONFIRM_FRAMES:
                    self.get_logger().info(f'🎉 识别到队员: {recognized_name}！')
                    self.audio.say_hello(recognized_name)
                    self.cooldown_until = current_time + self.COOLDOWN_SEC
                    self.confirm_count = 0
            else:
                self.confirm_count = 0
                
        except Exception as e:
            self.get_logger().error(f'处理失败: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = BPUInsightFaceRecognizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
BPU 推理封装类 - 使用 hrt_model_exec 进行推理
"""
import subprocess
import numpy as np
import os
import tempfile
import shutil
import time

class BPUInfer:
    def __init__(self, model_path, core_id=0):
        """
        初始化 BPU 推理器
        Args:
            model_path: 模型文件路径 (.bin)
            core_id: BPU 核心 ID (0 表示任意核心)
        """
        self.model_path = model_path
        self.core_id = core_id
        self.temp_dir = tempfile.mkdtemp(prefix='bpu_')
        self.input_size = None
        self.output_size = None
        
        # 获取模型信息
        self._get_model_info()
    
    def _get_model_info(self):
        """获取模型输入输出信息"""
        cmd = [
            'hrt_model_exec', 'model_info',
            '--model_file', self.model_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        # 简单解析，实际可以用更复杂的方式
        return True
    
    def infer_from_file(self, image_path):
        """
        从图片文件进行推理
        Args:
            image_path: 图片路径 (jpg/png)
        Returns:
            numpy array: 特征向量
        """
        output_dir = os.path.join(self.temp_dir, f'out_{int(time.time())}')
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            'hrt_model_exec', 'infer',
            '--model_file', self.model_path,
            '--input_file', image_path,
            '--core_id', str(self.core_id),
            '--enable_dump', 'true',
            '--dump_path', output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f'推理失败: {result.stderr}')
        
        # 查找输出文件
        output_file = None
        for f in os.listdir(output_dir):
            if f.startswith('model_infer_output_') and f.endswith('.bin'):
                output_file = os.path.join(output_dir, f)
                break
        
        if output_file is None:
            raise RuntimeError('未找到输出文件')
        
        # 读取特征向量
        feature = np.fromfile(output_file, dtype=np.float32)
        
        # 清理临时文件
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return feature
    
    def infer_from_array(self, image_array):
        """
        从 numpy 数组进行推理
        Args:
            image_array: numpy array (H, W, 3) BGR 格式
        Returns:
            numpy array: 特征向量
        """
        # 保存为临时文件
        temp_img = os.path.join(self.temp_dir, f'temp_{int(time.time())}.jpg')
        import cv2
        cv2.imwrite(temp_img, image_array)
        
        try:
            feature = self.infer_from_file(temp_img)
        finally:
            os.remove(temp_img)
        
        return feature
    
    def __del__(self):
        """清理临时目录"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

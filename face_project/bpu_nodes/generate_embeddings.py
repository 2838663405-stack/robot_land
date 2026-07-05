#!/usr/bin/env python3
"""
生成人脸特征数据库
使用 BPU 从照片中提取特征并保存
"""
import os
import cv2
import numpy as np
import pickle
import glob
from bpu_infer import BPUInfer

# 配置
DATA_DIR = os.path.expanduser('~/face_project/face_data')  # 存放各人照片的目录
OUTPUT_PATH = os.path.expanduser('~/face_project/bpu_nodes/data/face_embeddings.pkl')

def generate_embeddings():
    print('🔍 初始化 BPU 推理器...')
    recognizer = BPUInfer('/app/models/w600k_r50.bin')
    
    face_db = []
    
    # 遍历每个人员的文件夹
    for person_dir in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_dir)
        if not os.path.isdir(person_path):
            continue
        
        print(f'\n📁 处理人员: {person_dir}')
        
        # 获取该人员所有图片
        images = glob.glob(os.path.join(person_path, '*.jpg')) + \
                 glob.glob(os.path.join(person_path, '*.jpeg')) + \
                 glob.glob(os.path.join(person_path, '*.png'))
        
        if not images:
            print(f'  ⚠️ 没有找到图片，跳过')
            continue
        
        embeddings = []
        for img_path in images[:20]:  # 最多取20张
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Resize 到 112x112
                img = cv2.resize(img, (112, 112))
                
                # 提取特征
                feature = recognizer.infer_from_array(img)
                if feature is not None:
                    embeddings.append(feature)
                    print(f'  ✅ {os.path.basename(img_path)}')
            except Exception as e:
                print(f'  ❌ {os.path.basename(img_path)}: {e}')
        
        if embeddings:
            # 取平均作为该人员的特征
            avg_embedding = np.mean(embeddings, axis=0)
            # 归一化
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            face_db.append({
                'name': person_dir,
                'embedding': avg_embedding,
                'count': len(embeddings)
            })
            print(f'  📊 平均特征: {len(embeddings)} 张图片')
        else:
            print(f'  ❌ 无有效图片')
    
    # 保存数据库
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(face_db, f)
    
    print(f'\n✅ 特征数据库已保存到: {OUTPUT_PATH}')
    print(f'📊 共 {len(face_db)} 人')
    for person in face_db:
        print(f'  - {person["name"]}: {person["count"]} 张图片')

if __name__ == '__main__':
    generate_embeddings()

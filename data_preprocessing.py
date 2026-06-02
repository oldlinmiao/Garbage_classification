# -*- coding: utf-8 -*-
import os
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

SEED = 42
BASE_DIR = '垃圾图片数据集/数据集_120类'
WORK_DIR = 'dataset_split'
CLASSES = ['可回收物', '厨余垃圾', '其他垃圾', '有害垃圾']

random.seed(SEED)
np.random.seed(SEED)

def clean_and_split_data():
    all_paths = []
    all_labels = []
    
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f'找不到路径：{BASE_DIR}')
    
    for sub_folder in os.listdir(BASE_DIR):
        sub_path = os.path.join(BASE_DIR, sub_folder)
        if not os.path.isdir(sub_path):
            continue
        
        parts = sub_folder.split('_')
        if len(parts) < 2:
            continue
        
        big_class = parts[0]
        if big_class not in CLASSES:
            continue
        
        label_idx = CLASSES.index(big_class)
        
        for img_name in os.listdir(sub_path):
            img_path = os.path.join(sub_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                if w < 64 or h < 64:
                    continue
                all_paths.append(img_path)
                all_labels.append(label_idx)
            except Exception:
                continue
    
    total = len(all_paths)
    if total == 0:
        raise ValueError('没有读取到任何有效图片')
    
    print(f'清洗后有效图片总数：{total} 张')
    for idx, cls in enumerate(CLASSES):
        count = all_labels.count(idx)
        print(f'  {cls}：{count} 张')
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.30,
        random_state=SEED,
        stratify=all_labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )
    
    splits = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }
    
    for split_name, (paths, labels) in splits.items():
        split_dir = os.path.join(WORK_DIR, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        
        for cls in CLASSES:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
        
        for path, label in zip(paths, labels):
            cls_name = CLASSES[label]
            dst = os.path.join(split_dir, cls_name, os.path.basename(path))
            shutil.copy2(path, dst)
    
    print(f'\n训练集：{len(train_paths)} 张')
    print(f'验证集：{len(val_paths)} 张')
    print(f'测试集：{len(test_paths)} 张')
    print('划分完成，已保存至', WORK_DIR)

if __name__ == '__main__':
    clean_and_split_data()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yolo.py — VOC→YOLO 数据转换 + YOLOv8 模型训练与推理模块

将 train_yolo.ipynb 中的全部功能改写为可复用的 Python 函数，
供 DL 课程设计其他文件 import 调用。

包含的功能：
  1. VOC XML → YOLO txt 格式转换
  2. dataset 目录组织 + data.yaml 生成
  3. YOLOv8 模型训练
  4. 训练结果可视化
  5. 模型推理测试

使用方式：
  import train_yolo
  train_yolo.convert_voc_to_yolo()              # 数据集准备
  train_yolo.train_yolo()                       # 训练
  train_yolo.show_training_results()            # 查看训练曲线
  train_yolo.run_inference('path/to/image.jpg') # 推理
"""

import os
import shutil
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================
# 全局配置（可按需通过 set_config 修改）
# ============================================================

VOC_ROOT = "./VOCdevkit/VOC2012"          # VOC2012 根目录
OUTPUT_DIR = "./YOLO_dataset"             # 转换后输出目录
TRAIN_RATIO = 0.8                         # 训练集比例
YOLO_MODEL = "yolov8n.pt"                # 预训练权重
EPOCHS = 30                               # 训练轮数
IMAGE_SIZE = 640                          # 输入尺寸
DEVICE = 0                                # 0=GPU, 'cpu'=CPU
BATCH_SIZE = 32                           # 批大小
WORKERS = 4                               # 数据加载线程数
PATIENCE = 15                             # 早停耐心值
PROJECT = 'runs/train'                    # 训练项目目录
NAME = 'voc_yolo_notebook'                # 训练名称
EXIST_OK = True                           # 是否覆盖同名目录

# VOC 的 20 个类别（顺序不能错）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}


# ============================================================
# 配置管理
# ============================================================

def set_config(voc_root=None, output_dir=None, train_ratio=None,
               yolo_model=None, epochs=None, image_size=None,
               device=None, batch_size=None, workers=None,
               patience=None, project=None, name=None, exist_ok=None):
    """自定义全局配置参数（在调用训练/转换前设置）。"""
    global VOC_ROOT, OUTPUT_DIR, TRAIN_RATIO, YOLO_MODEL
    global EPOCHS, IMAGE_SIZE, DEVICE, BATCH_SIZE
    global WORKERS, PATIENCE, PROJECT, NAME, EXIST_OK

    if voc_root is not None:
        VOC_ROOT = voc_root
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    if train_ratio is not None:
        TRAIN_RATIO = train_ratio
    if yolo_model is not None:
        YOLO_MODEL = yolo_model
    if epochs is not None:
        EPOCHS = epochs
    if image_size is not None:
        IMAGE_SIZE = image_size
    if device is not None:
        DEVICE = device
    if batch_size is not None:
        BATCH_SIZE = batch_size
    if workers is not None:
        WORKERS = workers
    if patience is not None:
        PATIENCE = patience
    if project is not None:
        PROJECT = project
    if name is not None:
        NAME = name
    if exist_ok is not None:
        EXIST_OK = exist_ok


def get_config():
    """返回当前配置的字典。"""
    return {
        'VOC_ROOT': VOC_ROOT,
        'OUTPUT_DIR': OUTPUT_DIR,
        'TRAIN_RATIO': TRAIN_RATIO,
        'YOLO_MODEL': YOLO_MODEL,
        'EPOCHS': EPOCHS,
        'IMAGE_SIZE': IMAGE_SIZE,
        'DEVICE': DEVICE,
        'BATCH_SIZE': BATCH_SIZE,
        'WORKERS': WORKERS,
        'PATIENCE': PATIENCE,
        'PROJECT': PROJECT,
        'NAME': NAME,
        'EXIST_OK': EXIST_OK,
    }


def print_config():
    """打印当前配置。"""
    print("=" * 50)
    print("当前配置:")
    print("=" * 50)
    cfg = get_config()
    for k, v in cfg.items():
        print(f"  {k:>20s}: {v}")
    print("=" * 50)


# ============================================================
# 数据转换
# ============================================================

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """解析单个 VOC XML 文件，返回 YOLO 格式的标注行列表。

    参数:
        xml_path: XML 标注文件路径
        img_width: 图像宽度（像素）
        img_height: 图像高度（像素）

    返回:
        list[str]: 每行格式为 "class_id x_center y_center width height"
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # 归一化
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 防止溢出 0-1
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return yolo_lines


def convert_voc_to_yolo(voc_root=None, output_dir=None, train_ratio=None,
                        shuffle_seed=42, verbose=True):
    """将 VOC 数据集转换为 YOLO 格式。

    流程:
        1. 创建 images/ 和 labels/ 下的 train/ val/ 子目录
        2. 按比例随机划分训练集/验证集
        3. 复制图像并转换 XML 标注为 YOLO txt
        4. 生成 data.yaml

    返回:
        Path: YOLO 数据集的根目录
    """
    if voc_root is None:
        voc_root = VOC_ROOT
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if train_ratio is None:
        train_ratio = TRAIN_RATIO

    voc_root = Path(voc_root)
    yolo_root = Path(output_dir)
    jpeg_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"

    # 创建输出目录
    for split in ['train', 'val']:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 获取所有有效图像
    image_files = [
        p.stem for p in jpeg_dir.glob("*.jpg")
        if (ann_dir / f"{p.stem}.xml").exists()
    ]
    if not image_files:
        raise FileNotFoundError(
            f"未找到图片或 XML 文件，请检查 VOC_ROOT: {voc_root}"
        )

    random.seed(shuffle_seed)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    if verbose:
        print(f"总图片数: {len(image_files)}, "
              f"训练集: {len(train_files)}, 验证集: {len(val_files)}")

    # 复制并转换
    for split, files in [('train', train_files), ('val', val_files)]:
        iterator = tqdm(files, desc=f"处理 {split} 集") if verbose else files
        for stem in iterator:
            # 拷贝图片
            src_img = jpeg_dir / f"{stem}.jpg"
            dst_img = yolo_root / "images" / split / f"{stem}.jpg"
            shutil.copy(src_img, dst_img)

            # 解析 XML 获取图像尺寸
            xml_path = ann_dir / f"{stem}.xml"
            tree = ET.parse(xml_path)
            size = tree.getroot().find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            # 转换并写入标签
            yolo_lines = convert_xml_to_yolo(xml_path, img_w, img_h)
            label_path = yolo_root / "labels" / split / f"{stem}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))

    # 生成 data.yaml
    data_yaml = {
        'path': str(yolo_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(VOC_CLASSES),
        'names': VOC_CLASSES
    }
    yaml_path = yolo_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    if verbose:
        print(f"✅ 数据集准备完成！YAML 配置文件保存在：{yaml_path}")

    return yolo_root, yaml_path


# ============================================================
# 模型训练
# ============================================================

def train_yolo(yaml_path=None, yolo_model=None, epochs=None,
               image_size=None, batch_size=None, device=None,
               workers=None, patience=None, project=None,
               name=None, exist_ok=None, verbose=True):
    """训练 YOLOv8 模型。

    参数:
        yaml_path: data.yaml 路径（None 则自动使用 OUTPUT_DIR/data.yaml）
        yolo_model: 预训练权重路径
        epochs: 训练轮数
        image_size: 输入尺寸
        batch_size: 批大小
        device: 设备（0=GPU, 'cpu'=CPU）
        workers: 数据加载线程数
        patience: 早停耐心值
        project: 保存项目目录
        name: 训练名称
        exist_ok: 是否覆盖同名目录
        verbose: 是否打印详细信息

    返回:
        tuple: (model, results) — 训练好的模型和训练结果对象
    """
    if yaml_path is None:
        yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml 不存在: {yaml_path}。请先运行 convert_voc_to_yolo()"
        )

    # 使用全局默认值
    if yolo_model is None:
        yolo_model = YOLO_MODEL
    if epochs is None:
        epochs = EPOCHS
    if image_size is None:
        image_size = IMAGE_SIZE
    if batch_size is None:
        batch_size = BATCH_SIZE
    if device is None:
        device = DEVICE
    if workers is None:
        workers = WORKERS
    if patience is None:
        patience = PATIENCE
    if project is None:
        project = PROJECT
    if name is None:
        name = NAME
    if exist_ok is None:
        exist_ok = EXIST_OK

    device_label = 'GPU' if device != 'cpu' else 'CPU'
    if verbose:
        print(f"开始训练，使用设备: {device_label}")
        print(f"  模型: {yolo_model}")
        print(f"  数据: {yaml_path}")
        print(f"  Epochs: {epochs}, Batch: {batch_size}, ImgSz: {image_size}")

    model = YOLO(yolo_model)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        workers=workers,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        patience=patience,
    )

    if verbose:
        print("🎉 训练完成！")
        print(f"📁 模型保存在: {results.save_dir}")

    return model, results


# ============================================================
# 训练结果可视化
# ============================================================

def show_training_results(results_dir=None):
    """显示训练过程中保存的 results.png（损失与指标曲线）。

    参数:
        results_dir: 训练输出目录
                     （None 则自动使用 PROJECT/NAME）
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if results_dir is None:
        results_dir = Path(PROJECT) / "runs" / "train" / NAME
    else:
        results_dir = Path(results_dir)

    result_img_path = results_dir / "results.png"

    if result_img_path.exists():
        img = Image.open(result_img_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("训练过程可视化 (损失与指标曲线)")
        plt.show()
    else:
        print(f"还未生成 results.png: {result_img_path}")
        print("请确保训练已完成至少 1 个 epoch。")

    return result_img_path.exists()


# ============================================================
# 推理测试
# ============================================================

def run_inference(image_path=None, model_path=None, show_result=True):
    """使用训练好的模型对单张图像进行推理。

    参数:
        image_path: 输入图像路径（None 则自动取验证集第一张）
        model_path: 模型权重路径
                    （None 则自动使用 BEST_WEIGHTS）
        show_result: 是否显示检测结果

    返回:
        tuple: (results, model) — 推理结果和模型对象
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2

    # 默认模型路径
    if model_path is None:
        model_path = Path(PROJECT) / "runs" / "train" / NAME / "weights" / "best.pt"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 默认图像路径
    if image_path is None:
        val_img_dir = Path(OUTPUT_DIR) / "images" / "val"
        test_images = list(val_img_dir.glob("*.jpg"))
        if not test_images:
            raise FileNotFoundError(f"未找到验证集图片: {val_img_dir}")
        image_path = str(test_images[0])

    print(f"测试图片: {image_path}")
    print(f"模型: {model_path}")

    model = YOLO(str(model_path))
    results = model(image_path, save=True, show_labels=True, show_conf=True)

    if show_result:
        for r in results:
            im_array = r.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(im_rgb)
            plt.axis('off')
            plt.show()

    return results, model


def run_batch_inference(image_dir=None, model_path=None, max_images=4):
    """批量推理并展示结果。

    参数:
        image_dir: 图像目录（None 则使用验证集目录）
        model_path: 模型路径
        max_images: 最多展示的图像数量

    返回:
        list: 推理结果列表
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2

    if model_path is None:
        model_path = Path(PROJECT) / "runs" / "train" / NAME / "weights" / "best.pt"

    if image_dir is None:
        image_dir = Path(OUTPUT_DIR) / "images" / "val"

    image_dir = Path(image_dir)
    test_images = list(image_dir.glob("*.jpg"))[:max_images]

    if not test_images:
        raise FileNotFoundError(f"未找到图像: {image_dir}")

    model = YOLO(str(model_path))
    all_results = []

    for img_path in test_images:
        results = model(img_path, verbose=False)[0]
        all_results.append(results)

    # 显示
    fig, axes = plt.subplots(1, len(test_images),
                             figsize=(5 * len(test_images), 5))
    if len(test_images) == 1:
        axes = [axes]

    for ax, img_path, r in zip(axes, test_images, all_results):
        im_array = r.plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
        ax.axis('off')
        n_det = len(r.boxes) if r.boxes is not None else 0
        ax.set_title(f'{img_path.name}\n{n_det} detections', fontsize=9)

    plt.tight_layout()
    plt.show()

    return all_results


# ============================================================
# 一键运行
# ============================================================

def run_all(skip_convert=False, skip_train=False):
    """运行完整流程：数据转换 → 训练 → 可视化 → 推理测试。

    参数:
        skip_convert: 跳过数据转换（数据集已存在时设为 True）
        skip_train: 跳过训练（模型已存在时设为 True）
    """
    print("=" * 60)
    print("  YOLOv8 VOC 训练完整流程")
    print("=" * 60)
    print_config()

    # Step 1: 数据转换
    if not skip_convert:
        print("\n>>> Step 1: VOC → YOLO 数据转换")
        yolo_root, yaml_path = convert_voc_to_yolo()
    else:
        print("\n>>> Step 1: 跳过（数据集已存在）")
        yaml_path = Path(OUTPUT_DIR) / "data.yaml"

    # Step 2: 训练
    if not skip_train:
        print("\n>>> Step 2: 模型训练")
        model, results = train_yolo(yaml_path=yaml_path)
    else:
        print("\n>>> Step 2: 跳过（模型已存在）")
        model = None

    # Step 3: 显示训练曲线
    print("\n>>> Step 3: 训练曲线可视化")
    show_training_results()

    # Step 4: 推理测试
    print("\n>>> Step 4: 推理测试")
    run_inference()

    print("\n🎉 全部流程完成！")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    run_all()

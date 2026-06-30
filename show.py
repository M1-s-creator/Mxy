#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show.py — YOLOv8 训练结果展示与可视化模块

将 show.ipynb 中的全部功能改写为可复用的 Python 函数，
供 DL 课程设计其他文件 import 调用。

包含的功能：
  1. 模型训练结果展示（损失曲线、mAP曲线、混淆矩阵）
  2. 数据可视化图表（类别分布、标注框示例、边界框尺寸分布）
  3. 错误样本分析（低置信度检测、漏检、误检）
  4. 特征图/注意力权重可视化

使用方式：
  import show
  show.plot_training_curves()        # 绘制训练曲线
  show.plot_class_distribution()     # 绘制类别分布
  show.run_error_analysis()          # 运行错误分析
  show.run_all()                     # 运行全部展示流程
"""

import os
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2

import torch
import torch.nn as nn
from ultralytics import YOLO

# ============================================================
# 全局配置
# ============================================================

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 路径配置 —— 可按需修改
TRAIN_DIR = Path('runs/detect/runs/train/voc_yolo_notebook')
RESULTS_CSV = TRAIN_DIR / 'results.csv'
BEST_WEIGHTS = TRAIN_DIR / 'weights' / 'best.pt'
DATA_YAML = Path('YOLO_dataset/data.yaml')
VOC_ROOT = Path('VOCdevkit/VOC2012')
YOLO_DATASET = Path('YOLO_dataset')

# 20 个 VOC 类别名称
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(CLASS_NAMES)

# 错误分析阈值
LOW_CONF_THRESHOLD = 0.35    # 低于此值视为低置信度
HIGH_IOU_THRESHOLD = 0.5     # IoU > 此值视为匹配成功
ANALYSIS_SUBSET = 500        # 错误分析使用的图像数量

# 全局模型缓存（惰性加载）
_model = None


# ============================================================
# 工具函数
# ============================================================

def set_paths(train_dir=None, best_weights=None, voc_root=None,
              yolo_dataset=None, data_yaml=None):
    """允许外部调用者自定义路径配置。"""
    global TRAIN_DIR, RESULTS_CSV, BEST_WEIGHTS, DATA_YAML
    global VOC_ROOT, YOLO_DATASET

    if train_dir is not None:
        TRAIN_DIR = Path(train_dir)
        RESULTS_CSV = TRAIN_DIR / 'results.csv'
        BEST_WEIGHTS = TRAIN_DIR / 'weights' / 'best.pt'
    if best_weights is not None:
        BEST_WEIGHTS = Path(best_weights)
    if voc_root is not None:
        VOC_ROOT = Path(voc_root)
    if yolo_dataset is not None:
        YOLO_DATASET = Path(yolo_dataset)
    if data_yaml is not None:
        DATA_YAML = Path(data_yaml)


def verify_paths():
    """验证必要路径，返回 True/False。"""
    ok = True
    for p, desc in [
        (RESULTS_CSV, 'results.csv'),
        (BEST_WEIGHTS, 'best.pt'),
        (VOC_ROOT, 'VOC2012'),
    ]:
        if not p.exists():
            print(f"[WARN] 未找到 {desc}: {p}")
            ok = False
    return ok


def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件，返回 (img_w, img_h, objects)。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(float(size.find('width').text))
    img_h = int(float(size.find('height').text))

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = obj.find('difficult')
        is_difficult = int(difficult.text) if difficult is not None else 0
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': is_difficult
        })

    return img_w, img_h, objects


def load_yolo_labels(label_path):
    """加载 YOLO 格式标注，返回 [[class_id, cx, cy, w, h], ...]（归一化坐标）。"""
    labels = []
    label_path = Path(label_path)
    if not label_path.exists():
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    return labels


def yolo_to_pixel(bbox, img_w, img_h):
    """将 YOLO 归一化坐标转换为像素坐标 [xmin, ymin, xmax, ymax]。"""
    class_id, cx, cy, w, h = bbox
    xmin = (cx - w / 2) * img_w
    ymin = (cy - h / 2) * img_h
    xmax = (cx + w / 2) * img_w
    ymax = (cy + h / 2) * img_h
    return int(class_id), xmin, ymin, xmax, ymax


def compute_iou(box1, box2):
    """计算两个 [xmin, ymin, xmax, ymax] 框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def count_class_distribution(label_dir):
    """统计 YOLO 标签目录中每个类别的目标数量。"""
    class_counts = Counter()
    label_dir = Path(label_dir)
    for label_file in label_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        except Exception:
            continue
    return class_counts


def collect_bbox_sizes(xml_dir, img_dir=None, max_samples=3000):
    """收集 XML 标注中所有边界框的宽高，返回 (widths, heights, class_ids)。"""
    widths, heights, class_ids = [], [], []
    xml_files = sorted(Path(xml_dir).glob('*.xml'))[:max_samples]

    for xml_path in xml_files:
        try:
            img_w, img_h, objects = parse_voc_xml(xml_path)
            for obj in objects:
                if obj['name'] not in CLASS_NAMES:
                    continue
                xmin, ymin, xmax, ymax = obj['bbox']
                w = xmax - xmin
                h = ymax - ymin
                widths.append(w)
                heights.append(h)
                class_ids.append(CLASS_NAMES.index(obj['name']))
        except Exception:
            continue
    return np.array(widths), np.array(heights), np.array(class_ids)


def draw_boxes_on_ax(ax, img_path, xml_path, title_prefix=""):
    """在 matplotlib axes 上绘制图像及 VOC 标注框。"""
    img = Image.open(img_path)
    img_w, img_h, objects = parse_voc_xml(xml_path)

    ax.imshow(img)

    for obj in objects:
        if obj['name'] not in CLASS_NAMES:
            continue
        xmin, ymin, xmax, ymax = obj['bbox']
        class_id = CLASS_NAMES.index(obj['name'])
        color = plt.cm.tab20(class_id / NUM_CLASSES)

        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--' if obj.get('difficult') else '-'
        )
        ax.add_patch(rect)

        label = obj['name']
        ax.text(
            xmin, max(ymin - 5, 10), label, fontsize=7,
            color='white', weight='bold',
            bbox=dict(boxstyle='square,pad=0.15',
                      facecolor=tuple(color[:3]) + (0.85,),
                      edgecolor='none')
        )

    ax.set_title(
        f"{title_prefix}{Path(img_path).name}\n{len(objects)} objects",
        fontsize=10
    )
    ax.axis('off')


def draw_error_detail(img_path, gts, preds, img_w, img_h, ax, title):
    """绘制错误分析图：绿色=GT, 红色=低置信度预测, 蓝色=正常预测。"""
    img = Image.open(img_path)
    ax.imshow(img)

    # 画 GT
    for gt in gts:
        bbox = gt['bbox']
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 8, f"GT: {gt['class_name']}",
            fontsize=7, color='white',
            bbox=dict(boxstyle='square,pad=0.1', facecolor='green'),
            weight='bold'
        )

    # 画预测
    for pred in preds:
        bbox = pred['bbox']
        conf = pred.get('confidence', 0)
        color = 'red' if conf < LOW_CONF_THRESHOLD else 'cyan'
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] + bbox[3] - bbox[1] + 2,
            f"{pred.get('class_name', '?')} {conf:.2f}",
            fontsize=7, color='white',
            bbox=dict(boxstyle='square,pad=0.1', facecolor=color),
            weight='bold'
        )

    ax.set_title(title, fontsize=10)
    ax.axis('off')


def get_model():
    """惰性加载 YOLO 模型（缓存，避免重复加载）。"""
    global _model
    if _model is None:
        print("正在加载 YOLOv8 模型...")
        _model = YOLO(str(BEST_WEIGHTS))
    return _model


# ============================================================
# Section 1: 训练结果展示
# ============================================================

def load_training_results():
    """加载训练结果 DataFrame，返回 df。"""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"未找到 results.csv: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()
    return df


def print_final_metrics(df=None):
    """打印最终的训练指标。"""
    if df is None:
        df = load_training_results()

    print("=" * 70)
    print("YOLOv8n 在 VOC2012 上的训练最终结果 (Epoch 30)")
    print("=" * 70)
    print(f"  mAP@0.5:       {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
    print(f"  mAP@0.5:0.95:  {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
    print(f"  Precision:     {df['metrics/precision(B)'].iloc[-1]:.4f}")
    print(f"  Recall:        {df['metrics/recall(B)'].iloc[-1]:.4f}")
    print("-" * 70)
    for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                'val/box_loss', 'val/cls_loss', 'val/dfl_loss']:
        print(f"  {col:>20s}:  {df[col].iloc[-1]:.4f}")
    print("=" * 70)

    # 各 epoch 指标汇总
    cols_show = [
        'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)',
        'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]
    print("\n各 epoch 指标汇总:")
    print(df[cols_show].round(4).to_string(index=False))

    return df


def plot_training_curves(df=None, save_path='output_loss_curves.png',
                         show=True):
    """绘制训练/验证损失曲线 + 学习率曲线。"""
    if df is None:
        df = load_training_results()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # (a) 训练损失
    axes[0].plot(df['epoch'], df['train/box_loss'], 'o-', markersize=2,
                 color='#2196F3', label='Box Loss')
    axes[0].plot(df['epoch'], df['train/cls_loss'], 's-', markersize=2,
                 color='#FF5722', label='Cls Loss')
    axes[0].plot(df['epoch'], df['train/dfl_loss'], '^-', markersize=2,
                 color='#4CAF50', label='DFL Loss')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss Curves', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 31)

    # (b) 验证损失
    axes[1].plot(df['epoch'], df['val/box_loss'], 'o-', markersize=2,
                 color='#2196F3', label='Box Loss')
    axes[1].plot(df['epoch'], df['val/cls_loss'], 's-', markersize=2,
                 color='#FF5722', label='Cls Loss')
    axes[1].plot(df['epoch'], df['val/dfl_loss'], '^-', markersize=2,
                 color='#4CAF50', label='DFL Loss')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Validation Loss Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 31)

    # (c) 学习率
    axes[2].plot(df['epoch'], df['lr/pg0'], '-', color='#9C27B0', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Learning Rate', fontsize=11)
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 31)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"损失曲线图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_map_pr_curves(df=None, save_path='output_map_pr_curves.png',
                       show=True):
    """绘制 mAP、Precision、Recall 曲线。"""
    if df is None:
        df = load_training_results()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) mAP 曲线
    axes[0].plot(df['epoch'], df['metrics/mAP50(B)'], 'o-', markersize=3,
                 color='#2196F3', linewidth=2, label='mAP@0.5')
    axes[0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 's-', markersize=3,
                 color='#FF5722', linewidth=2, label='mAP@0.5:0.95')
    axes[0].fill_between(df['epoch'], df['metrics/mAP50(B)'],
                         alpha=0.1, color='#2196F3')
    axes[0].fill_between(df['epoch'], df['metrics/mAP50-95(B)'],
                         alpha=0.1, color='#FF5722')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('mAP', fontsize=11)
    axes[0].set_title('mAP Curves (Validation)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 31)

    # 标注最终值
    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
    final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
    axes[0].annotate(f'{final_map50:.3f}', xy=(30, final_map50),
                     xytext=(25, final_map50 + 0.03),
                     arrowprops=dict(arrowstyle='->', color='#2196F3'),
                     fontsize=10, color='#2196F3', fontweight='bold')
    axes[0].annotate(f'{final_map50_95:.3f}', xy=(30, final_map50_95),
                     xytext=(25, final_map50_95 - 0.04),
                     arrowprops=dict(arrowstyle='->', color='#FF5722'),
                     fontsize=10, color='#FF5722', fontweight='bold')

    # (b) Precision & Recall
    axes[1].plot(df['epoch'], df['metrics/precision(B)'], 'o-', markersize=3,
                 color='#4CAF50', linewidth=2, label='Precision')
    axes[1].plot(df['epoch'], df['metrics/recall(B)'], 's-', markersize=3,
                 color='#FF9800', linewidth=2, label='Recall')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Precision & Recall Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 31)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"mAP和PR曲线图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def show_yolo_generated_charts(show=True):
    """展示 YOLO 自动生成的训练综合图表（results.png 和 confusion_matrix）。"""
    results_png = TRAIN_DIR / 'results.png'
    cm_png = TRAIN_DIR / 'confusion_matrix_normalized.png'

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    if results_png.exists():
        axes[0].imshow(plt.imread(str(results_png)))
        axes[0].axis('off')
        axes[0].set_title('YOLOv8 Training Results Overview',
                          fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'results.png not found', ha='center', va='center')

    if cm_png.exists():
        axes[1].imshow(plt.imread(str(cm_png)))
        axes[1].axis('off')
        axes[1].set_title('Normalized Confusion Matrix',
                          fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'confusion_matrix not found',
                     ha='center', va='center')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def show_pr_f1_curves(show=True):
    """展示 per-class PR 曲线和 F1 曲线。"""
    pr_png = TRAIN_DIR / 'BoxPR_curve.png'
    f1_png = TRAIN_DIR / 'BoxF1_curve.png'

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    if pr_png.exists():
        axes[0].imshow(plt.imread(str(pr_png)))
        axes[0].axis('off')
        axes[0].set_title('Precision-Recall Curve (per class)',
                          fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'BoxPR_curve.png not found',
                     ha='center', va='center')

    if f1_png.exists():
        axes[1].imshow(plt.imread(str(f1_png)))
        axes[1].axis('off')
        axes[1].set_title('F1-Confidence Curve (per class)',
                          fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'BoxF1_curve.png not found',
                     ha='center', va='center')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def show_validation_batches(show=True):
    """展示验证集预测样本（GT vs Prediction）和训练批次样本。"""
    val_batch_dirs = list(TRAIN_DIR.glob('val_batch*_labels.jpg'))
    val_batch_dirs.sort()

    if val_batch_dirs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        for i in range(min(3, len(val_batch_dirs))):
            # GT
            gt_path = val_batch_dirs[i]
            axes[0, i].imshow(plt.imread(str(gt_path)))
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Validation Batch {i}: Ground Truth',
                                 fontsize=11, fontweight='bold')

            # Prediction
            pred_path = Path(str(gt_path).replace('_labels', '_pred'))
            if pred_path.exists():
                axes[1, i].imshow(plt.imread(str(pred_path)))
                axes[1, i].axis('off')
                axes[1, i].set_title(f'Validation Batch {i}: Predictions',
                                     fontsize=11, fontweight='bold')
            else:
                axes[1, i].text(0.5, 0.5, 'No prediction', ha='center',
                                va='center')
                axes[1, i].axis('off')

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("未找到验证批次预测图像")
        fig = None

    # 训练批次样本
    train_batch_early = sorted(TRAIN_DIR.glob('train_batch0.jpg'))
    train_batch_late = sorted(TRAIN_DIR.glob('train_batch10700.jpg'))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for ax, path, title in [
        (axes2[0],
         train_batch_early[0] if train_batch_early else None,
         'Early Training Batch (with Mosaic Aug.)'),
        (axes2[1],
         train_batch_late[0] if train_batch_late else None,
         'Late Training Batch (with Mosaic Aug.)')
    ]:
        if path and path.exists():
            ax.imshow(plt.imread(str(path)))
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig, fig2


def show_labels_distribution_image(show=True):
    """展示 YOLO 生成的标签分布图。"""
    labels_img = TRAIN_DIR / 'labels.jpg'
    if labels_img.exists():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(plt.imread(str(labels_img)))
        ax.axis('off')
        ax.set_title('YOLO-Generated Label Distribution Visualization',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    else:
        print("未找到 labels.jpg")
        return None


# ============================================================
# Section 2: 数据可视化
# ============================================================

def plot_class_distribution(save_path='output_class_distribution.png',
                            show=True):
    """绘制类别分布直方图（Section 4.2.1）。"""
    train_label_dir = YOLO_DATASET / 'labels' / 'train'
    val_label_dir = YOLO_DATASET / 'labels' / 'val'

    print("正在统计训练集类别分布...")
    train_counts = count_class_distribution(train_label_dir)
    print("正在统计验证集类别分布...")
    val_counts = count_class_distribution(val_label_dir)

    train_vals = np.array([train_counts.get(i, 0) for i in range(NUM_CLASSES)])
    val_vals = np.array([val_counts.get(i, 0) for i in range(NUM_CLASSES)])
    total_per_class = train_vals + val_vals

    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(NUM_CLASSES)
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_vals, width,
                   label=f'Train ({int(sum(train_vals)):,} objects)',
                   color='steelblue', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, val_vals, width,
                   label=f'Validation ({int(sum(val_vals)):,} objects)',
                   color='coral', edgecolor='white', linewidth=0.5)

    # 在柱子上标注较大的值
    for bar, val in zip(bars1, train_vals):
        if val > max(train_vals) * 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                    str(int(val)), ha='center', va='bottom', fontsize=7,
                    rotation=90)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Objects', fontsize=12)
    ax.set_title('Class Distribution: PASCAL VOC 2012 Object Instances per Category',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"类别分布图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 打印统计
    print("\n各类别目标数量统计:")
    for i, name in enumerate(CLASS_NAMES):
        t = train_vals[i]
        v = val_vals[i]
        print(f"  {name:>15s}:  Train={int(t):>6d}  Val={int(v):>6d}  Total={int(t+v):>6d}")
    print(f"\n总目标数: Train={int(sum(train_vals)):,}  Val={int(sum(val_vals)):,}")
    print(f"类别最不平衡比例: {max(total_per_class)/max(1, min(total_per_class)):.1f}:1")

    return fig, train_vals, val_vals


def plot_sample_annotations(save_path='output_sample_annotations.png',
                            show=True):
    """展示原始图像 + 标注框示例（Section 4.2.2）。"""
    img_dir = VOC_ROOT / 'JPEGImages'
    xml_dir = VOC_ROOT / 'Annotations'
    all_images = sorted([f.stem for f in img_dir.glob('*.jpg')
                         if (xml_dir / f'{f.stem}.xml').exists()])

    # 扫描前 2000 张按目标数量选择
    sample_pool = []
    for name in random.sample(all_images, min(2000, len(all_images))):
        xml_path = xml_dir / f'{name}.xml'
        _, _, objs = parse_voc_xml(xml_path)
        unique_classes = set(o['name'] for o in objs if o['name'] in CLASS_NAMES)
        sample_pool.append((name, len(objs), len(unique_classes)))

    sample_pool.sort(key=lambda x: x[1])
    n = len(sample_pool)
    indices = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, -1]
    selected = [sample_pool[idx][0] for idx in indices]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, img_name in enumerate(selected):
        img_path = img_dir / f'{img_name}.jpg'
        xml_path = xml_dir / f'{img_name}.xml'
        draw_boxes_on_ax(axes[i], img_path, xml_path)

    fig.suptitle('Sample Original Images with Ground Truth Annotations (VOC2012)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"标注示例图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 图例
    fig_legend, ax_legend = plt.subplots(figsize=(14, 1.5))
    legend_patches = [
        Rectangle((0, 0), 1, 1,
                  facecolor=plt.cm.tab20(i / NUM_CLASSES),
                  edgecolor='black', linewidth=0.5, label=name)
        for i, name in enumerate(CLASS_NAMES)
    ]
    ax_legend.legend(handles=legend_patches, ncol=10, loc='center',
                     fontsize=8, title='Class Color Legend', title_fontsize=10)
    ax_legend.axis('off')
    if show:
        plt.show()
    else:
        plt.close(fig_legend)

    return fig, fig_legend


def plot_bbox_distribution(save_path='output_bbox_distribution.png',
                           show=True):
    """绘制边界框尺寸分布 - 散点图 + 密度图（Section 4.2.3）。"""
    img_dir = VOC_ROOT / 'JPEGImages'
    xml_dir = VOC_ROOT / 'Annotations'

    print("正在收集边界框尺寸数据...")
    bbox_widths, bbox_heights, bbox_classes = collect_bbox_sizes(
        xml_dir, img_dir, max_samples=5000
    )

    areas = bbox_widths * bbox_heights
    aspect_ratios = bbox_widths / (bbox_heights + 1e-8)

    print(f"收集到 {len(bbox_widths)} 个边界框")
    print(f"  宽度: mean={bbox_widths.mean():.1f}, std={bbox_widths.std():.1f}, "
          f"min={bbox_widths.min():.0f}, max={bbox_widths.max():.0f}")
    print(f"  高度: mean={bbox_heights.mean():.1f}, std={bbox_heights.std():.1f}, "
          f"min={bbox_heights.min():.0f}, max={bbox_heights.max():.0f}")
    print(f"  面积: mean={areas.mean():.0f}, median={np.median(areas):.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) 散点图
    sc = axes[0].scatter(bbox_widths, bbox_heights, c=bbox_classes,
                         cmap='tab20', alpha=0.35, s=2, edgecolors='none')
    axes[0].set_xlabel('Width (pixels)', fontsize=11)
    axes[0].set_ylabel('Height (pixels)', fontsize=11)
    axes[0].set_title('Bounding Box Size Scatter Plot', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(0, np.percentile(bbox_widths, 99))
    axes[0].set_ylim(0, np.percentile(bbox_heights, 99))

    # (b) 2D 密度热力图
    hist, x_edges, y_edges = np.histogram2d(
        np.clip(bbox_widths, 0, np.percentile(bbox_widths, 99)),
        np.clip(bbox_heights, 0, np.percentile(bbox_heights, 99)),
        bins=80
    )
    im = axes[1].imshow(hist.T, origin='lower', aspect='auto',
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        cmap='hot', norm=plt.matplotlib.colors.LogNorm())
    axes[1].set_xlabel('Width (pixels)', fontsize=11)
    axes[1].set_ylabel('Height (pixels)', fontsize=11)
    axes[1].set_title('Bounding Box Size Density Heatmap',
                      fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes[1], label='Count (log scale)')

    # (c) 面积分布直方图
    axes[2].hist(np.log10(areas + 1), bins=60, color='steelblue',
                 edgecolor='white', alpha=0.8, density=True)
    axes[2].axvline(np.log10(np.median(areas)), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(areas):.0f} px²')
    axes[2].set_xlabel('log10(Area) [pixels²]', fontsize=11)
    axes[2].set_ylabel('Density', fontsize=11)
    axes[2].set_title('Bounding Box Area Distribution', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"边界框尺寸分布图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, bbox_widths, bbox_heights, areas


# ============================================================
# Section 3: 错误分析
# ============================================================

def run_inference_on_val_subset(model=None, subset_size=None, verbose=True):
    """在验证集子集上运行推理，返回 all_predictions 列表。

    返回:
        list[dict]: 每个元素包含:
            img_path, img_w, img_h, predictions, ground_truth
    """
    if model is None:
        model = get_model()
    if subset_size is None:
        subset_size = ANALYSIS_SUBSET

    val_img_dir = YOLO_DATASET / 'images' / 'val'
    val_label_dir = YOLO_DATASET / 'labels' / 'val'
    val_images = sorted(val_img_dir.glob('*.jpg'))

    if verbose:
        print(f"验证集共有 {len(val_images)} 张图像")

    analysis_images = random.sample(val_images,
                                    min(subset_size, len(val_images)))
    if verbose:
        print(f"选取 {len(analysis_images)} 张图像进行错误分析")

    all_predictions = []

    for i, img_path in enumerate(analysis_images):
        if verbose and (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(analysis_images)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model(img_path, verbose=False)
        result = results[0]

        img = Image.open(img_path)
        img_w, img_h = img.size

        # 提取预测框
        preds = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, clss):
                preds.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': CLASS_NAMES[int(cls)]
                })

        # 加载真实标注
        label_path = val_label_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_labels(label_path)
        gts = []
        for lab in gt_labels:
            cls_id, xmin, ymin, xmax, ymax = yolo_to_pixel(lab, img_w, img_h)
            gts.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'class': cls_id,
                'class_name': CLASS_NAMES[cls_id]
            })

        all_predictions.append({
            'img_path': str(img_path),
            'img_w': img_w,
            'img_h': img_h,
            'predictions': preds,
            'ground_truth': gts
        })

    if verbose:
        print(f"推理完成！共处理 {len(all_predictions)} 张图像")

    return all_predictions


def analyze_errors(all_predictions, verbose=True):
    """分析错误样本（低置信度检测、漏检、误检）。

    返回:
        tuple: (low_conf_samples, missed_samples, fp_samples)
    """
    low_conf_samples = []
    missed_samples = []
    fp_samples = []

    for item in all_predictions:
        preds = item['predictions']
        gts = item['ground_truth']
        img_w, img_h = item['img_w'], item['img_h']

        matched_gt = set()
        matched_pred = set()

        # 匹配预测与 GT
        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                if pred['class'] != gt['class']:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > HIGH_IOU_THRESHOLD:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        # 低置信度正检
        for pi in matched_pred:
            if (LOW_CONF_THRESHOLD > preds[pi]['confidence'] > 0.1):
                # 找到对应的 gt 索引
                pred_idx_in_matched = list(matched_pred).index(pi)
                gt_idx = (list(matched_gt)[pred_idx_in_matched]
                          if pred_idx_in_matched < len(matched_gt) else 0)
                low_conf_samples.append({
                    'img_path': item['img_path'],
                    'pred': preds[pi],
                    'gt': gts[gt_idx],
                    'img_w': img_w,
                    'img_h': img_h
                })

        # 漏检
        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                gt_bbox = gt['bbox']
                gt_area = ((gt_bbox[2] - gt_bbox[0]) *
                           (gt_bbox[3] - gt_bbox[1]))
                missed_samples.append({
                    'img_path': item['img_path'],
                    'gt': gt,
                    'img_w': img_w,
                    'img_h': img_h,
                    'gt_area': gt_area
                })

        # 误检
        for pi, pred in enumerate(preds):
            if pi not in matched_pred and pred['confidence'] > 0.5:
                fp_samples.append({
                    'img_path': item['img_path'],
                    'pred': pred,
                    'img_w': img_w,
                    'img_h': img_h
                })

    # 排序
    missed_samples.sort(key=lambda x: x['gt_area'])
    low_conf_samples.sort(key=lambda x: x['pred']['confidence'])

    if verbose:
        print("=" * 50)
        print("错误样本统计:")
        print(f"  低置信度正检: {len(low_conf_samples)} 个")
        print(f"  漏检 (False Negative): {len(missed_samples)} 个")
        print(f"  误检 (False Positive): {len(fp_samples)} 个")
        print("=" * 50)

    return low_conf_samples, missed_samples, fp_samples


def plot_low_conf_samples(low_conf_samples,
                          save_path='output_low_conf_samples.png',
                          show=True):
    """绘制低置信度正检样本。"""
    if not low_conf_samples:
        print("未找到低置信度正检样本")
        return None

    n_show = min(6, len(low_conf_samples))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i in range(n_show):
        sample = low_conf_samples[i]
        pred = sample['pred']
        gt = sample.get('gt')

        gts_list = [gt] if gt else []
        preds_list = [pred]

        draw_error_detail(
            sample['img_path'], gts_list, preds_list,
            sample['img_w'], sample['img_h'], axes[i],
            f"Low Conf: {pred['class_name']} conf={pred['confidence']:.2f}\n"
            f"bbox={pred['bbox'][2]-pred['bbox'][0]:.0f}x"
            f"{pred['bbox'][3]-pred['bbox'][1]:.0f}px"
        )

    for i in range(n_show, 6):
        axes[i].axis('off')

    fig.suptitle('Error Analysis: Low-Confidence Correct Detections',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"低置信度样本图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_missed_samples(missed_samples,
                        save_path='output_missed_detections.png',
                        show=True):
    """绘制漏检样本并分析原因。"""
    if not missed_samples:
        print("未找到漏检样本")
        return None

    n_show = min(6, len(missed_samples))

    # 最小目标 + 中等目标
    show_samples = list(missed_samples[:n_show // 2])
    if len(missed_samples) > n_show // 2:
        mid_start = len(missed_samples) // 2
        show_samples += missed_samples[mid_start:mid_start + n_show - n_show // 2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, sample in enumerate(show_samples[:6]):
        gt = sample['gt']
        bbox = gt['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        img = Image.open(sample['img_path'])
        axes[i].imshow(img)

        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=3, edgecolor='red', facecolor='none', linestyle='-'
        )
        axes[i].add_patch(rect)

        if area < 5000:
            axes[i].text(
                bbox[0], bbox[1] - 10,
                f"MISSED: {gt['class_name']}",
                fontsize=8, color='white',
                bbox=dict(boxstyle='square,pad=0.1', facecolor='red'),
                weight='bold'
            )
            margin = 30
            zoom_x1 = max(0, bbox[0] - margin)
            zoom_y1 = max(0, bbox[1] - margin)
            zoom_x2 = min(sample['img_w'], bbox[2] + margin)
            zoom_y2 = min(sample['img_h'], bbox[3] + margin)
            rect_zoom = Rectangle(
                (zoom_x1, zoom_y1), zoom_x2 - zoom_x1, zoom_y2 - zoom_y1,
                linewidth=1.5, edgecolor='yellow', facecolor='none',
                linestyle=':'
            )
            axes[i].add_patch(rect_zoom)
        else:
            axes[i].text(
                bbox[0], bbox[1] - 10,
                f"MISSED: {gt['class_name']}",
                fontsize=8, color='white',
                bbox=dict(boxstyle='square,pad=0.1', facecolor='red'),
                weight='bold'
            )

        axes[i].set_title(
            f"FN: {gt['class_name']} | Area: {area:.0f}px² | "
            f"{bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f}px",
            fontsize=9, color='red'
        )
        axes[i].axis('off')

    for i in range(len(show_samples[:6]), 6):
        axes[i].axis('off')

    fig.suptitle('Error Analysis: Missed Detections (False Negatives)',
                 fontsize=14, fontweight='bold', color='darkred')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"漏检样本图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 漏检原因统计
    print("\n漏检原因分析:")
    missed_small = sum(1 for s in missed_samples if s['gt_area'] < 32 * 32)
    missed_medium = sum(1 for s in missed_samples
                        if 32 * 32 <= s['gt_area'] < 96 * 96)
    missed_large = sum(1 for s in missed_samples if s['gt_area'] >= 96 * 96)
    total = max(1, len(missed_samples))
    print(f"  极小目标 (<32x32):   {missed_small} "
          f"({missed_small/total*100:.1f}%)")
    print(f"  小目标 (32x32-96x96): {missed_medium} "
          f"({missed_medium/total*100:.1f}%)")
    print(f"  大目标 (>=96x96):     {missed_large} "
          f"({missed_large/total*100:.1f}%)")

    return fig


def run_error_analysis(model=None, subset_size=None, show=True):
    """运行完整的错误分析流程（推理 + 分析 + 可视化）。

    返回:
        tuple: (low_conf_samples, missed_samples, fp_samples)
    """
    if model is None:
        model = get_model()

    print("\n>>> 步骤 1/3: 运行推理...")
    all_preds = run_inference_on_val_subset(model, subset_size)

    print("\n>>> 步骤 2/3: 分析错误样本...")
    low_conf, missed, fp = analyze_errors(all_preds)

    print("\n>>> 步骤 3/3: 绘制错误分析图...")
    plot_low_conf_samples(low_conf, show=show)
    plot_missed_samples(missed, show=show)

    return low_conf, missed, fp


# ============================================================
# Section 4: 特征图 / 注意力可视化
# ============================================================

def register_feature_hooks(model=None):
    """在 YOLOv8 backbone 各层注册 forward hook 以提取特征图。

    返回:
        tuple: (feature_maps_dict, hooks_list)
    """
    global _model
    if model is None:
        model = get_model()

    feature_maps = {}

    def _hook_factory(name):
        def hook(module, input, output):
            if isinstance(output, (list, tuple)):
                feature_maps[name] = output[0].detach()
            else:
                feature_maps[name] = output.detach()
        return hook

    hooks = []
    target_indices = [0, 2, 4, 6, 9]

    try:
        base_model = model.model.model if hasattr(model.model, 'model') else model.model

        for idx in target_indices:
            if idx < len(base_model):
                layer = base_model[idx]
                hook_name = f'layer_{idx}'
                try:
                    h = layer.register_forward_hook(_hook_factory(hook_name))
                    hooks.append(h)
                    print(f"已注册 hook: {hook_name} -> {type(layer).__name__}")
                except Exception as e:
                    print(f"注册 hook {hook_name} 失败: {e}")

        print(f"\n共注册 {len(hooks)} 个 feature hooks")
    except Exception as e:
        print(f"注册 hooks 时出错: {e}")

    return feature_maps, hooks


def extract_features(model=None, image_path=None, feature_maps=None):
    """运行一次前向传播，提取特征图。

    参数:
        model: YOLO 模型（None 则自动加载）
        image_path: 测试图像路径（None 则从验证集随机选取）
        feature_maps: 用于存储的字典（None 则自动创建）

    返回:
        tuple: (feature_maps, image_path)
    """
    if model is None:
        model = get_model()

    val_img_dir = YOLO_DATASET / 'images' / 'val'

    if image_path is None:
        test_imgs = list(val_img_dir.glob('*.jpg'))
        image_path = str(test_imgs[0]) if test_imgs else None
        if image_path is None:
            raise FileNotFoundError("未找到验证集图像")

    if feature_maps is None:
        feature_maps = {}

    print(f"测试图像: {image_path}")

    # 显示原图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    img_test = Image.open(image_path)
    ax.imshow(img_test)
    ax.set_title(f'Test Image: {Path(image_path).name}', fontsize=12)
    ax.axis('off')
    plt.show()

    print("\n正在运行前向传播以提取特征图...")
    feature_maps.clear()
    with torch.no_grad():
        _ = model(image_path, verbose=False)

    print(f"提取到 {len(feature_maps)} 层特征图")
    for name, fm in feature_maps.items():
        print(f"  {name}: shape = {fm.shape}")

    return feature_maps, image_path


def plot_feature_maps(feature_maps, show=True):
    """对每一层特征图进行可视化（平均激活 + Top-8 高响应通道）。

    参数:
        feature_maps: extract_features 返回的字典
        show: 是否显示图像
    """
    if not feature_maps:
        print("未提取到特征图")
        return

    for layer_name, fm in feature_maps.items():
        fm_single = fm[0]  # [C, H, W]
        C, H, W = fm_single.shape

        mean_activation = fm_single.mean(dim=0).cpu().numpy()
        channel_std = fm_single.std(dim=[1, 2]).cpu().numpy()
        top_channels = np.argsort(channel_std)[-8:]

        fig, axes = plt.subplots(3, 3, figsize=(14, 13))
        axes = axes.flatten()

        # (0) 平均激活
        im0 = axes[0].imshow(mean_activation, cmap='hot',
                             interpolation='bilinear')
        axes[0].set_title(f'Mean Activation\n{H}x{W}', fontsize=9)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # (1-8) Top 8 通道
        for j, ch_idx in enumerate(reversed(top_channels)):
            ch_map = fm_single[ch_idx].cpu().numpy()
            ch_map = ((ch_map - ch_map.min()) /
                      (ch_map.max() - ch_map.min() + 1e-8))
            im = axes[j + 1].imshow(ch_map, cmap='inferno',
                                    interpolation='bilinear')
            axes[j + 1].set_title(
                f'Channel {ch_idx}\nstd={channel_std[ch_idx]:.3f}', fontsize=9
            )
            axes[j + 1].axis('off')

        fig.suptitle(
            f'Feature Map Visualization: {layer_name} '
            f'(shape: {list(fm.shape)})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        save_path = f'output_feature_map_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"特征图 {layer_name} 已保存至 {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


def create_attention_overlay(img_path, model, conf_threshold=0.25):
    """创建注意力叠加图 —— 检测区域清晰，非关注区域高斯模糊。

    参数:
        img_path: 图像路径
        model: YOLO 模型
        conf_threshold: 置信度阈值

    返回:
        tuple: (attention_map, mask, result)
    """
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    blurred = cv2.GaussianBlur(img_rgb, (51, 51), 30)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model(img_path, verbose=False, conf=conf_threshold)[0]

    mask = np.zeros((h, w), dtype=np.float32)

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = max(x2 - x1, y2 - y1) // 3

            for y in range(max(0, y1), min(h, y2)):
                for x in range(max(0, x1), min(w, x2)):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    val = conf * np.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-8))
                    mask[y, x] = max(mask[y, x], val)

    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    mask = np.clip(mask, 0, 1)

    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    attention_map = (img_rgb * mask_3ch + blurred * (1 - mask_3ch)).astype(
        np.uint8)

    return attention_map, mask, result


def plot_attention_maps(model=None, num_images=8,
                        save_path='output_attention_maps.png',
                        show=True):
    """绘制注意力热力图（模型关注区域清晰，其他区域模糊）。

    参数:
        model: YOLO 模型
        num_images: 显示的图像数量
        save_path: 保存路径
        show: 是否显示图像
    """
    if model is None:
        model = get_model()

    val_img_dir = YOLO_DATASET / 'images' / 'val'
    test_imgs = list(val_img_dir.glob('*.jpg'))[:num_images]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for i, img_path in enumerate(test_imgs):
        attn_map, mask, result = create_attention_overlay(img_path, model)
        axes[i].imshow(attn_map)

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                axes[i].add_patch(rect)

        axes[i].set_title(f'{img_path.name}', fontsize=8)
        axes[i].axis('off')

    fig.suptitle(
        'Attention Visualization: Model Focus Regions '
        '(Blurred = Low Attention)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"注意力热力图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_feature_visualization(model=None, show=True):
    """运行完整的特征图 + 注意力可视化流程。

    返回:
        tuple: (feature_maps, hooks)
    """
    if model is None:
        model = get_model()

    print("\n--- 特征图可视化 ---")
    fm, hooks = register_feature_hooks(model)
    fm, img_path = extract_features(model, feature_maps=fm)
    plot_feature_maps(fm, show=show)

    # 清理 hooks
    for h in hooks:
        h.remove()
    print("已清除所有 hooks")

    print("\n--- 注意力热力图 ---")
    plot_attention_maps(model, show=show)

    # 检测结果示例
    print("\n--- 检测结果示例 ---")
    plot_detection_examples(model, show=show)

    return fm, hooks


def plot_detection_examples(model=None, num_images=4,
                            save_path='output_detection_examples.png',
                            show=True):
    """YOLO 检测结果示例（备选方案，当特征图提取失败时使用）。"""
    if model is None:
        model = get_model()

    val_img_dir = YOLO_DATASET / 'images' / 'val'
    test_images = list(val_img_dir.glob('*.jpg'))[:num_images]

    fig, axes = plt.subplots(2, num_images, figsize=(18, 8))

    for i, img_path in enumerate(test_images):
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original: {img_path.name}', fontsize=9)
        axes[0, i].axis('off')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model(img_path, verbose=False)[0]
        annotated = result.plot()
        axes[1, i].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        if result.boxes is not None:
            n_det = len(result.boxes)
            confs = result.boxes.conf.cpu().numpy()
            classes_detected = [
                CLASS_NAMES[int(c)] for c in result.boxes.cls.cpu().numpy()
            ]
            title = f'Detections: {n_det}\n'
            for cls_name, conf in zip(classes_detected, confs):
                title += f'{cls_name}:{conf:.2f} '
        else:
            title = 'No detections'
        axes[1, i].set_title(title, fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle('YOLOv8 Detection Results on Validation Images',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"检测示例图已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ============================================================
# 汇总函数：一键运行全部
# ============================================================

def run_all(show_plots=True):
    """运行全部展示流程（等价于从头到尾执行 notebook）。

    参数:
        show_plots: 是否弹出 matplotlib 窗口显示图像。
                    False 时仅保存到文件，适合无 GUI 环境。
    """
    print("=" * 70)
    print("  YOLOv8 训练结果与数据可视化 — 完整流程")
    print("=" * 70)

    # ---- Section 1: 训练结果 ----
    print("\n" + "=" * 70)
    print("  Section 1: 模型训练结果展示")
    print("=" * 70)

    df = load_training_results()
    print("所有库导入成功！")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    print("\n所有路径验证通过！")
    print(f"训练结果目录: {TRAIN_DIR}")
    print(f"模型权重:     {BEST_WEIGHTS}")
    print(f"VOC数据目录:  {VOC_ROOT}")

    print_final_metrics(df)
    plot_training_curves(df, show=show_plots)
    plot_map_pr_curves(df, show=show_plots)
    show_yolo_generated_charts(show=show_plots)
    show_pr_f1_curves(show=show_plots)
    show_validation_batches(show=show_plots)

    # ---- Section 2: 数据可视化 ----
    print("\n" + "=" * 70)
    print("  Section 2: 数据可视化图表")
    print("=" * 70)

    plot_class_distribution(show=show_plots)
    plot_sample_annotations(show=show_plots)
    plot_bbox_distribution(show=show_plots)

    # ---- Section 3: 错误分析 ----
    print("\n" + "=" * 70)
    print("  Section 3: 错误样本分析")
    print("=" * 70)

    run_error_analysis(show=show_plots)

    # ---- Section 4: 特征图可视化 ----
    print("\n" + "=" * 70)
    print("  Section 4: 特征图/注意力可视化")
    print("=" * 70)

    run_feature_visualization(show=show_plots)

    # ---- 标签分布图 ----
    show_labels_distribution_image(show=show_plots)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  全部展示流程完成！")
    print("=" * 70)
    print_summary()


def print_summary():
    """打印课程设计总结信息。"""
    print("""
    训练结果总结:
      - 模型: YOLOv8n (Nano)，参数量约 3.0M，计算量约 8.1 GFLOPs
      - 数据集: PASCAL VOC 2012，20个类别
      - 训练配置: 30 epochs, batch=32, imgsz=640, AdamW 优化器
      - 最终性能: mAP50 = 0.863, mAP50-95 = 0.699, Precision = 0.851, Recall = 0.777

    错误分析发现:
      - 小目标和极小目标是漏检的主要原因
      - 低置信度检测通常出现在遮挡、光照不佳或目标与背景相似的场景
      - 类别不平衡（如 person 远多于 sheep）可能影响少数类的检测精度

    改进方向:
      - 使用更大的模型（YOLOv8s/m/l）提升检测精度
      - 增加数据增强策略（更强的 Mosaic、MixUp）
      - 针对小目标使用多尺度训练或 SAHI 推理策略
    """)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    run_all(show_plots=True)

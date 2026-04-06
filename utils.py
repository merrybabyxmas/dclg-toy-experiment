import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_attn_map(attn_map, save_path):
    """
    attn_map: [HW] tensor
    """
    if len(attn_map.shape) == 1:
        hw = attn_map.shape[0]
        h = w = int(np.sqrt(hw))
        attn_map = attn_map.reshape(h, w)
    
    attn_map = attn_map.detach().cpu().float().numpy()
    
    # 가시성 강화: 0.0 ~ 1.0 사이로 정규화 (하지만 전체적인 대조를 위해 상위 1% 컷오프 적용 가능)
    plt.imshow(attn_map, cmap='magma')
    plt.colorbar()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_overlap(attn_A, attn_B, save_path):
    """
    attn_A, attn_B: [HW] tensor
    """
    if len(attn_A.shape) == 1:
        hw = attn_A.shape[0]
        h = w = int(np.sqrt(hw))
        attn_A = attn_A.reshape(h, w)
        attn_B = attn_B.reshape(h, w)
    
    # Hadamard product
    overlap = (attn_A * attn_B).detach().cpu().float().numpy()
    
    plt.imshow(overlap, cmap='jet')
    plt.colorbar()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_grid(images, labels, save_path):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    if num_images == 1:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(labels[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def log_loss_curve(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, color='red', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Chimera Loss')
    plt.title('Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

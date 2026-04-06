import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def apply_overlay(image_pil, attn_map, color_map='jet', alpha=0.5):
    """
    원본 이미지 위에 어텐션 맵을 투명하게 겹칩니다.
    image_pil: PIL Image (512x512)
    attn_map: [HW] or [H, W] tensor/array
    """
    if torch.is_tensor(attn_map):
        attn_map = attn_map.detach().cpu().float().numpy()
    
    if len(attn_map.shape) == 1:
        h = w = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape(h, w)
    
    # 1. 어텐션 맵 정규화 및 리사이즈 (512x512)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map_uint8 = (attn_map * 255).astype(np.uint8)
    attn_map_resized = cv2.resize(attn_map_uint8, (image_pil.size[0], image_pil.size[1]))
    
    # 2. 컬러맵 적용
    cmap = plt.get_cmap(color_map)
    rgba_attn = cmap(attn_map_resized / 255.0)
    rgb_attn = (rgba_attn[:, :, :3] * 255).astype(np.uint8)
    
    # 3. 블렌딩
    img_np = np.array(image_pil)
    overlayed = cv2.addWeighted(img_np, 1 - alpha, rgb_attn, alpha, 0)
    return Image.fromarray(overlayed)

def visualize_attn_map(attn_map, save_path, color_map='magma'):
    if len(attn_map.shape) == 1:
        h = w = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape(h, w)
    attn_map = attn_map.detach().cpu().float().numpy()
    plt.imshow(attn_map, cmap=color_map)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_overlap(attn_A, attn_B, save_path):
    if len(attn_A.shape) == 1:
        h = w = int(np.sqrt(attn_A.shape[0]))
        attn_A = attn_A.reshape(h, w)
        attn_B = attn_B.reshape(h, w)
    overlap = (attn_A * attn_B).detach().cpu().float().numpy()
    plt.imshow(overlap, cmap='jet')
    plt.colorbar()
    plt.axis('off')
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

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_attn_map(attn_map, save_path):
    """
    attn_map: [H, W] or [HW] tensor
    """
    if len(attn_map.shape) == 1:
        hw = attn_map.shape[0]
        h = w = int(np.sqrt(hw))
        attn_map = attn_map.reshape(h, w)
    
    attn_map = attn_map.detach().cpu().float().numpy()
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def visualize_overlap(attn_A, attn_B, save_path):
    """
    attn_A, attn_B: [H, W] or [HW] tensor
    """
    if len(attn_A.shape) == 1:
        hw = attn_A.shape[0]
        h = w = int(np.sqrt(hw))
        attn_A = attn_A.reshape(h, w)
        attn_B = attn_B.reshape(h, w)
    
    overlap = (attn_A * attn_B).detach().cpu().float().numpy()
    plt.imshow(overlap, cmap='hot')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def save_grid(images, labels, save_path):
    """
    images: list of PIL Images
    labels: list of strings
    """
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
    """
    losses: list of floats
    """
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Chimera Loss')
    plt.title('Chimera Loss Curve')
    plt.savefig(save_path)
    plt.close()

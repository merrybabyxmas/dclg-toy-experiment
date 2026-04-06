"""
Dense Diffusion DCLG — Text-only, no IP-Adapter.
Two scenarios: (1) facing each other, (2) wrestling overlap.
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from dense_pipeline import DenseDCLGPipeline


def apply_overlay(image_pil, attn_map, color_map='Reds', alpha=0.5):
    if len(attn_map.shape) == 1:
        h = w = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape(h, w)
    attn_map = attn_map / (attn_map.max() + 1e-8)
    attn_map = (attn_map * 255).astype(np.uint8)
    attn_map = cv2.resize(attn_map, (image_pil.size[0], image_pil.size[1]),
                          interpolation=cv2.INTER_CUBIC)
    cmap = plt.get_cmap(color_map)
    rgba = cmap(attn_map / 255.0)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img_np = np.array(image_pil)
    blended = cv2.addWeighted(img_np, 1 - alpha, rgb, alpha, 0)
    return Image.fromarray(blended)


def create_report(results, prompt, output_dir, report_name):
    num_rows = len(results)
    num_cols = 7
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Generated Image', 'Knight Overlay', 'Orc Overlay',
                  'Knight Energy', 'Orc Energy', 'Overlap/Co-act', 'Loss']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')

    for row, res in enumerate(results):
        label = res.get('label', '')
        img = res['image']
        map_A = res.get('map_A')
        map_B = res.get('map_B')

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(label, fontsize=8, fontweight='bold', color='red',
                                rotation=0, labelpad=65, va='center')

        if map_A is not None and map_B is not None:
            overlay_A = apply_overlay(img, map_A, 'Reds', 0.5)
            axes[row, 1].imshow(overlay_A)
            overlay_B = apply_overlay(img, map_B, 'Blues', 0.5)
            axes[row, 2].imshow(overlay_B)

            h = w = int(np.sqrt(map_A.shape[0]))
            mA = map_A.reshape(h, w) / (map_A.max() + 1e-8)
            mB = map_B.reshape(h, w) / (map_B.max() + 1e-8)

            im3 = axes[row, 3].imshow(mA, cmap='Reds', vmin=0, vmax=1)
            plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)
            im4 = axes[row, 4].imshow(mB, cmap='Blues', vmin=0, vmax=1)
            plt.colorbar(im4, ax=axes[row, 4], fraction=0.046)
            im5 = axes[row, 5].imshow(mA * mB, cmap='YlOrRd')
            plt.colorbar(im5, ax=axes[row, 5], fraction=0.046)

        losses = res.get('losses', [])
        if losses and any(l > 0 for l in losses):
            axes[row, 6].plot(losses, color='red', linewidth=1.5)
            axes[row, 6].set_xlabel('Step', fontsize=8)
            axes[row, 6].grid(True, alpha=0.3)
            axes[row, 6].set_xticks(range(0, len(losses), 5))

        for c in range(num_cols):
            if c != 6:
                axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    fig.suptitle(
        f'Dense Diffusion DCLG (Text-Only)\n'
        f'PROMPT: \'{prompt}\'',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{report_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Report saved: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'configs', 'default.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    config['dclg']['grad_clip'] = 5.0

    device = "cuda"

    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model']['base'], torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print("Setting up Dense Diffusion DCLG...")
    pipeline = DenseDCLGPipeline(pipe, config, device)

    seed = config['generation']['seed']
    output_dir = os.path.join(base_dir, 'outputs')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    neg_prompt = "blurry, deformed, extra limbs, merged, single character, monochrome, duplicate"

    # ============================================================
    # Scenario 1: Non-overlapping (facing each other)
    # ============================================================
    print("\n" + "="*60)
    print("  SCENARIO 1: Facing Each Other (Non-Overlapping)")
    print("="*60)

    prompt_facing = ("A knight in shining silver armor on the left and "
                     "an orc with green skin on the right, "
                     "facing each other, battle scene, cinematic lighting")
    words_A_facing = ["knight", "shining", "silver", "armor", "left"]
    words_B_facing = ["orc", "green", "skin", "right"]

    experiments_facing = [
        # Text masking only (Dense Diffusion) — no gradient guidance
        {'lambda': 0.0, 'overlap': 0.0, 'label': 'mask_only'},
        # Different seeds to show consistency
        {'lambda': 0.0, 'overlap': 0.0, 'label': 'seed123', 'seed': 123},
        {'lambda': 0.0, 'overlap': 0.0, 'label': 'seed456', 'seed': 456},
        {'lambda': 0.0, 'overlap': 0.0, 'label': 'seed789', 'seed': 789},
    ]

    results_facing = []
    for exp in experiments_facing:
        lam = exp['lambda']
        label = exp['label']
        s = exp.get('seed', seed)
        print(f"\n  {label}: lambda={lam}, seed={s}")

        image, losses, final_maps = pipeline.generate(
            prompt=prompt_facing, negative_prompt=neg_prompt,
            words_A=words_A_facing, words_B=words_B_facing,
            lambda_max=lam, seed=s, overlap_width=exp['overlap'],
        )

        img_path = os.path.join(img_dir, f'dense_facing_{label}.png')
        image.save(img_path)

        results_facing.append({
            'label': label, 'image': image, 'losses': losses,
            'map_A': final_maps.get('map_A'), 'map_B': final_maps.get('map_B'),
        })

        if losses:
            nonzero = [l for l in losses if l > 0]
            if nonzero:
                print(f"    Loss: {nonzero[0]:.4f} → {nonzero[-1]:.4f}")

    create_report(results_facing, prompt_facing, output_dir, 'DENSE_FACING_REPORT')

    # ============================================================
    # Scenario 2: Wrestling (Overlapping)
    # ============================================================
    print("\n" + "="*60)
    print("  SCENARIO 2: Wrestling (Overlapping)")
    print("="*60)

    prompt_wrestling = ("A knight in shining silver armor grappling an orc with green skin, "
                        "the knight grabs from the left side, the orc pushes from the right side, "
                        "dynamic wrestling pose, cinematic lighting, detailed")
    words_A_wrestling = ["knight", "shining", "silver", "armor", "grabs", "left"]
    words_B_wrestling = ["orc", "green", "skin", "pushes", "right"]

    experiments_wrestling = [
        # Overlap sweep — text masking only
        {'lambda': 0.0, 'overlap': 0.3, 'label': 'ov30'},
        {'lambda': 0.0, 'overlap': 0.5, 'label': 'ov50'},
        {'lambda': 0.0, 'overlap': 0.7, 'label': 'ov70'},
        # Different seeds at best overlap
        {'lambda': 0.0, 'overlap': 0.5, 'label': 'ov50_s123', 'seed': 123},
        {'lambda': 0.0, 'overlap': 0.5, 'label': 'ov50_s456', 'seed': 456},
        {'lambda': 0.0, 'overlap': 0.5, 'label': 'ov50_s789', 'seed': 789},
    ]

    results_wrestling = []
    for exp in experiments_wrestling:
        lam = exp['lambda']
        ov = exp['overlap']
        label = exp['label']
        s = exp.get('seed', seed)
        print(f"\n  {label}: lambda={lam}, overlap={ov}, seed={s}")

        image, losses, final_maps = pipeline.generate(
            prompt=prompt_wrestling, negative_prompt=neg_prompt,
            words_A=words_A_wrestling, words_B=words_B_wrestling,
            lambda_max=lam, seed=s, overlap_width=ov,
        )

        img_path = os.path.join(img_dir, f'dense_wrestle_{label}.png')
        image.save(img_path)

        results_wrestling.append({
            'label': label, 'image': image, 'losses': losses,
            'map_A': final_maps.get('map_A'), 'map_B': final_maps.get('map_B'),
        })

        if losses:
            nonzero = [l for l in losses if l > 0]
            if nonzero:
                print(f"    Loss: {nonzero[0]:.4f} → {nonzero[-1]:.4f}")

    create_report(results_wrestling, prompt_wrestling, output_dir, 'DENSE_WRESTLING_REPORT')
    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()

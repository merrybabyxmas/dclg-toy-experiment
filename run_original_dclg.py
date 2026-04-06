"""
Original DCLG: Cosine similarity loss on text attention maps.
No spatial masks, no IP-Adapter. Pure attention overlap suppression.
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
from hook_manager import HookManager
from dclg_pipeline import DCLGPipeline


def get_token_index(tokenizer, prompt, target_word):
    inputs = tokenizer(prompt)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
    for i, token in enumerate(tokens):
        clean = token.replace('</w>', '').lower()
        if clean == target_word.lower():
            return i
    return -1


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


def create_report(results, prompt, output_dir):
    num = len(results)
    fig, axes = plt.subplots(3, num, figsize=(num * 4, 12))
    if num == 1:
        axes = axes.reshape(-1, 1)

    row_titles = ['Generated Image', 'Attention Overlay', 'Overlap Map']

    for col, res in enumerate(results):
        lam = res['lambda']
        img = res['image']
        map_A = res.get('map_A')
        map_B = res.get('map_B')

        axes[0, col].imshow(img)
        axes[0, col].set_title(f'Lambda = {lam}', fontsize=12, fontweight='bold')

        if map_A is not None:
            h = w = int(np.sqrt(map_A.shape[0]))
            mA = np.power(map_A.reshape(h, w), 2)
            mB = np.power(map_B.reshape(h, w), 2)
            mA = mA / (mA.max() + 1e-8)
            mB = mB / (mB.max() + 1e-8)

            # Knight=Red, Orc=Blue overlay
            overlay_A = apply_overlay(img, mA.flatten(), 'Reds', 0.4)
            overlay_B = apply_overlay(overlay_A, mB.flatten(), 'Blues', 0.3)
            axes[1, col].imshow(overlay_B)

            # Overlap heatmap
            overlap = mA * mB
            im = axes[2, col].imshow(overlap, cmap='YlOrRd')
            plt.colorbar(im, ax=axes[2, col], fraction=0.046)
        else:
            axes[1, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              fontsize=14, transform=axes[1, col].transAxes)
            axes[2, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              fontsize=14, transform=axes[2, col].transAxes)

        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    for r, title in enumerate(row_titles):
        axes[r, 0].set_ylabel(title, fontsize=11, fontweight='bold', rotation=90, labelpad=10)

    fig.suptitle(
        f'DCLG: Chimera Suppression via Attention Overlap Loss\n'
        f'PROMPT: \'{prompt}\'\n'
        f'Loss = cosine_similarity(map_A², map_B²) + erasure_penalty',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ORIGINAL_DCLG_REPORT.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Report saved: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'configs', 'default.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda"

    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model']['base'], torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Use the original prompt from config
    prompt = config['prompt']
    neg_prompt = config['negative_prompt']

    tokens = pipe.tokenizer(prompt)
    token_strs = pipe.tokenizer.convert_ids_to_tokens(tokens['input_ids'])
    print(f"Tokens: {list(enumerate(token_strs))}")

    idx_knight = get_token_index(pipe.tokenizer, prompt, "knight")
    idx_orc = get_token_index(pipe.tokenizer, prompt, "orc")
    print(f"Knight idx: {idx_knight}, Orc idx: {idx_orc}")

    hook_manager = HookManager(target_min_hw=config['dclg']['target_min_hw'])
    hook_manager.register_hooks(pipe.unet)

    dclg_pipe = DCLGPipeline(pipe, hook_manager, config)

    output_dir = os.path.join(base_dir, 'outputs')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    seed = config['generation']['seed']
    # Same lambda sweep as the original REPORT.png
    lambda_sweep = [0.0, 50.0, 200.0, 1000.0]

    results = []

    for lam in lambda_sweep:
        print(f"\n{'='*50}")
        print(f"  Lambda = {lam}")
        print(f"{'='*50}")

        image, losses = dclg_pipe.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            idx_A=idx_knight,
            idx_B=idx_orc,
            lambda_max=lam,
            seed=seed,
        )

        img_path = os.path.join(img_dir, f'orig_dclg_lam{lam}.png')
        image.save(img_path)
        print(f"  Saved: {img_path}")

        # Capture final attention maps
        captured = hook_manager.get_captured_maps()
        map_A, map_B = None, None
        if captured:
            name = list(captured.keys())[-1]
            attn_map = captured[name]
            num_heads = attn_map.shape[0] // 2
            cond_attn = attn_map[num_heads:].mean(dim=0)
            map_A = cond_attn[:, idx_knight].detach().cpu().float().numpy()
            map_B = cond_attn[:, idx_orc].detach().cpu().float().numpy()

        results.append({
            'lambda': lam, 'image': image, 'losses': losses,
            'map_A': map_A, 'map_B': map_B,
        })

        if losses:
            nonzero = [l for l in losses if l > 0]
            if nonzero:
                print(f"  Loss: {nonzero[0]:.4f} → {nonzero[-1]:.4f}")

    print("\nGenerating report...")
    create_report(results, prompt, output_dir)
    print(f"Done. Results in {output_dir}/")


if __name__ == "__main__":
    main()

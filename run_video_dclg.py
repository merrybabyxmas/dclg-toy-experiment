"""
Video DCLG — AnimateDiff + Cosine Similarity Loss.
Test scenarios: 2 entities fighting with heavy intersection.
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
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from video_dclg_pipeline import VideoDCLGPipeline


def save_video_gif(frames, path, duration=100):
    """Save frames as GIF."""
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)


def save_video_grid(frames, path, cols=8):
    """Save frames as a grid image."""
    n = len(frames)
    rows = (n + cols - 1) // cols
    w, h = frames[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        grid.paste(frame, (c * w, r * h))
    grid.save(path)


def create_overlay(frame, map_A, map_B, alpha=0.4):
    """Overlay attention maps on a frame."""
    if map_A is None or map_B is None:
        return frame
    h_map = w_map = int(np.sqrt(map_A.shape[0]))
    mA = np.power(map_A.reshape(h_map, w_map), 2)
    mB = np.power(map_B.reshape(h_map, w_map), 2)
    mA = mA / (mA.max() + 1e-8)
    mB = mB / (mB.max() + 1e-8)

    w, h = frame.size
    mA_up = cv2.resize(mA, (w, h), interpolation=cv2.INTER_CUBIC)
    mB_up = cv2.resize(mB, (w, h), interpolation=cv2.INTER_CUBIC)

    img_np = np.array(frame)
    # Red for entity A, Blue for entity B
    red = np.zeros_like(img_np)
    red[:, :, 0] = (mA_up * 255).astype(np.uint8)
    blue = np.zeros_like(img_np)
    blue[:, :, 2] = (mB_up * 255).astype(np.uint8)

    blended = cv2.addWeighted(img_np, 1.0, red, alpha, 0)
    blended = cv2.addWeighted(blended, 1.0, blue, alpha, 0)
    return Image.fromarray(blended)


def create_report(results, output_dir, report_name):
    """Create a comparison report for all experiments."""
    num_exps = len(results)
    fig, axes = plt.subplots(num_exps, 4, figsize=(20, num_exps * 3.5))
    if num_exps == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Frame Grid (16 frames)', 'First Frame', 'Attention Overlay', 'Loss Curve']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight='bold')

    for row, res in enumerate(results):
        label = res['label']
        frames = res['frames']

        # Column 0: mini grid (4x4)
        mini_w, mini_h = frames[0].size
        scale = min(256 / mini_w, 256 / mini_h)
        sw, sh = int(mini_w * scale), int(mini_h * scale)
        mini_frames = [f.resize((sw, sh)) for f in frames[:16]]
        grid = Image.new('RGB', (sw * 4, sh * 4))
        for i, f in enumerate(mini_frames):
            r, c = divmod(i, 4)
            grid.paste(f, (c * sw, r * sh))
        axes[row, 0].imshow(grid)
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold',
                                rotation=0, labelpad=80, va='center')

        # Column 1: first frame
        axes[row, 1].imshow(frames[0])

        # Column 2: attention overlay on first frame
        map_A = res.get('map_A')
        map_B = res.get('map_B')
        overlay = create_overlay(frames[0], map_A, map_B)
        axes[row, 2].imshow(overlay)

        # Column 3: loss curve
        losses = res.get('losses', [])
        if losses and any(l > 0 for l in losses):
            axes[row, 3].plot(losses, color='red', linewidth=1.5)
            axes[row, 3].set_xlabel('Step', fontsize=8)
            axes[row, 3].grid(True, alpha=0.3)

        for c in range(4):
            if c != 3:
                axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    fig.suptitle(f'Video DCLG: AnimateDiff + Cosine Similarity Guidance',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{report_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Report saved: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'configs', 'default.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Override for video
    config['generation']['num_inference_steps'] = 25
    config['generation']['guidance_scale'] = 7.5
    config['generation']['image_size'] = 512
    config['dclg']['tau_threshold'] = 15
    config['dclg']['grad_clip'] = 1.0

    device = "cuda"
    pipeline = VideoDCLGPipeline(config, device)
    pipeline.load_pipeline()

    output_dir = os.path.join(base_dir, 'outputs', 'video')
    os.makedirs(output_dir, exist_ok=True)

    neg_prompt = "blurry, deformed, extra limbs, merged, monochrome, duplicate, static"
    num_frames = 16

    # ============================================================
    # Test scenarios with heavy intersection
    # ============================================================
    scenarios = [
        {
            'name': 'wrestling',
            'prompt': "A knight in red armor and an orc with green skin wrestling fiercely, dynamic pose, cinematic lighting",
            'entity_A': 'knight',
            'entity_B': 'orc',
        },
        {
            'name': 'swordfight',
            'prompt': "A knight in silver armor and a dark warrior with a black sword clashing blades face to face, sparks flying, epic battle, cinematic",
            'entity_A': 'knight',
            'entity_B': 'warrior',
        },
        {
            'name': 'grapple',
            'prompt': "A red dragon and a blue dragon grappling each other in mid-air, their bodies intertwined, fire and ice, epic fantasy",
            'entity_A': 'red',
            'entity_B': 'blue',
        },
    ]

    # Lambda values to test
    lambda_sweep = [0.0, 50.0, 200.0]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"  PROMPT: {scenario['prompt']}")
        print(f"{'='*60}")

        results = []
        for lam in lambda_sweep:
            label = f"{scenario['name']}_lam{lam}"
            print(f"\n  {label}")

            try:
                frames, losses, final_maps = pipeline.generate(
                    prompt=scenario['prompt'],
                    negative_prompt=neg_prompt,
                    entity_A_word=scenario['entity_A'],
                    entity_B_word=scenario['entity_B'],
                    lambda_max=lam,
                    seed=42,
                    num_frames=num_frames,
                )

                # Save GIF
                gif_path = os.path.join(output_dir, f'{label}.gif')
                save_video_gif(frames, gif_path)
                print(f"    GIF: {gif_path}")

                # Save frame grid
                grid_path = os.path.join(output_dir, f'{label}_grid.png')
                save_video_grid(frames, grid_path)
                print(f"    Grid: {grid_path}")

                # Save overlay (first frame)
                overlay = create_overlay(frames[0], final_maps.get('map_A'), final_maps.get('map_B'))
                overlay.save(os.path.join(output_dir, f'{label}_overlay.png'))

                results.append({
                    'label': label,
                    'frames': frames,
                    'losses': losses,
                    'map_A': final_maps.get('map_A'),
                    'map_B': final_maps.get('map_B'),
                })

                if losses:
                    nonzero = [l for l in losses if l > 0]
                    if nonzero:
                        print(f"    Loss: {nonzero[0]:.4f} → {nonzero[-1]:.4f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        if results:
            create_report(results, output_dir, f'VIDEO_DCLG_{scenario["name"].upper()}_REPORT')

    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()

"""
Video DCLG — AnimateDiff + Cosine Similarity Loss.
Test scenarios: 2 entities fighting with heavy intersection.
Iter 1: Add BBox trajectory 3-zone masks + full debug GIF suite.
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from video_dclg_pipeline import VideoDCLGPipeline, _create_3zone_masks


# ============================================================
# BBox trajectories: Knight (A) vs Orc (B), 8 frames
# Image resolution: 256x256
# Entities approach from left/right and collide at frames 6 & 7
# BBox format: [x1, y1, x2, y2] in pixel coords
# ============================================================
KNIGHT_BBOXES = [
    [10,  28, 110, 228],   # frame 0 — far left
    [18,  28, 118, 228],   # frame 1
    [26,  28, 126, 228],   # frame 2
    [34,  28, 134, 228],   # frame 3
    [42,  28, 142, 228],   # frame 4
    [52,  28, 152, 228],   # frame 5
    [60,  28, 160, 228],   # frame 6 — collision: overlaps [96..160]
    [65,  28, 165, 228],   # frame 7 — collision: overlaps [91..165]
]

ORC_BBOXES = [
    [146, 28, 246, 228],   # frame 0 — far right
    [138, 28, 238, 228],   # frame 1
    [130, 28, 230, 228],   # frame 2
    [122, 28, 222, 228],   # frame 3
    [114, 28, 214, 228],   # frame 4
    [104, 28, 204, 228],   # frame 5
    [96,  28, 196, 228],   # frame 6 — collision
    [91,  28, 191, 228],   # frame 7 — collision
]

ENTITY_TRAJECTORIES = {"Knight (A)": KNIGHT_BBOXES, "Orc (B)": ORC_BBOXES}
COLLISION_FRAMES = (6, 7)
NUM_FRAMES = 8
FRAME_RES  = (256, 256)
LATENT_RES = (32, 32)   # 256 // 8 = 32


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

        # Column 0: mini grid (adaptive layout)
        mini_w, mini_h = frames[0].size
        n_frames = len(frames)
        grid_cols = min(4, n_frames)
        grid_rows = (n_frames + grid_cols - 1) // grid_cols
        scale = min(256 / mini_w, 256 / mini_h)
        sw, sh = int(mini_w * scale), int(mini_h * scale)
        mini_frames = [f.resize((sw, sh)) for f in frames]
        grid = Image.new('RGB', (sw * grid_cols, sh * grid_rows))
        for i, f in enumerate(mini_frames):
            r, c = divmod(i, grid_cols)
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


def colorize_map(arr, cmap_name="jet"):
    """Colorize a 2D numpy array as a PIL image."""
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(arr)
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))


def verify_3zone_masks(excl_A, excl_B, shared):
    """Print debug info for 3-zone mask tensors."""
    print(f"\n  === 3-Zone Mask Verification ===")
    print(f"  excl_A shape : {excl_A.shape}, sum by frame : {excl_A.sum(dim=(1,2,3)).tolist()}")
    print(f"  excl_B shape : {excl_B.shape}, sum by frame : {excl_B.sum(dim=(1,2,3)).tolist()}")
    print(f"  shared shape : {shared.shape}, sum by frame : {shared.sum(dim=(1,2,3)).tolist()}")
    print(f"  Collision frames {COLLISION_FRAMES}: shared nonzero = "
          f"{[shared[f, 0].sum().item() for f in COLLISION_FRAMES]}")
    non_col = [f for f in range(NUM_FRAMES) if f not in COLLISION_FRAMES]
    print(f"  Non-collision frames: shared all-zero = "
          f"{all(shared[f, 0].sum().item() == 0 for f in non_col)}")


def save_debug_3zone_gif(excl_A, excl_B, shared, path):
    """Save 3-zone mask visualization GIF."""
    lat_H, lat_W = excl_A.shape[2], excl_A.shape[3]
    vis_H, vis_W = 128, 128  # upscale for visibility
    zone_frames = []
    for t in range(excl_A.shape[0]):
        vis = np.zeros((lat_H, lat_W, 3), dtype=np.uint8)
        vis[excl_A[t, 0].bool().numpy()] = [255, 100, 100]   # red: A exclusive
        vis[shared[t, 0].bool().numpy()]  = [200, 200, 100]   # yellow: shared
        vis[excl_B[t, 0].bool().numpy()] = [100, 100, 255]   # blue: B exclusive
        img = Image.fromarray(vis).resize((vis_W, vis_H), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), f"f{t}", fill=(255, 255, 255))
        if t in COLLISION_FRAMES:
            draw.text((2, 14), "COLL", fill=(255, 220, 0))
        zone_frames.append(img)
    zone_frames[0].save(path, save_all=True, append_images=zone_frames[1:],
                        duration=200, loop=0)
    print(f"  3-zone mask GIF: {path}")


def save_debug_trajectory_gif(result_frames, path):
    """Save BBox trajectory overlay GIF."""
    traj_frames = []
    for t, frame in enumerate(result_frames):
        img = frame.copy()
        draw = ImageDraw.Draw(img)
        for ename, bboxes in ENTITY_TRAJECTORIES.items():
            x1, y1, x2, y2 = bboxes[t]
            color = "red" if "A" in ename else "blue"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, y1 + 2), ename, fill=color)
        traj_frames.append(img)
    traj_frames[0].save(path, save_all=True, append_images=traj_frames[1:],
                        duration=100, loop=0)
    print(f"  Trajectory GIF: {path}")


def save_debug_attn_gif(attn_map_history, path):
    """Save attention map sweep GIF (A=red, B=blue, overlap=hot)."""
    if not attn_map_history:
        print(f"  Attn GIF: skipped (no history)")
        return
    attn_frames = []
    for step_d in attn_map_history:
        mA = step_d["A"]
        mB = step_d["B"]
        side = int(mA.shape[0] ** 0.5)
        mA2d = mA.reshape(side, side)
        mB2d = mB.reshape(side, side)
        ov = mA2d * mB2d
        row_np = np.concatenate([
            np.array(colorize_map(mA2d, "Reds")),
            np.array(colorize_map(mB2d, "Blues")),
            np.array(colorize_map(ov,   "hot")),
        ], axis=1)
        attn_frames.append(Image.fromarray(row_np))
    attn_frames[0].save(path, save_all=True, append_images=attn_frames[1:],
                        duration=150, loop=0)
    print(f"  Attn sweep GIF: {path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'configs', 'default.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Override for video — smaller resolution to fit in GPU memory with gradients
    config['generation']['num_inference_steps'] = 25
    config['generation']['guidance_scale'] = 7.5
    config['generation']['image_size'] = 256  # 256x256 to fit gradient tracking
    config['dclg']['tau_threshold'] = 15
    config['dclg']['grad_clip'] = 1.0
    config['dclg']['target_min_hw'] = 64  # Lower threshold for smaller resolution

    device = "cuda"

    # ============================================================
    # Step 1: Verify 3-zone masks before loading the heavy pipeline
    # ============================================================
    print("=" * 60)
    print("  STEP 1: Verifying _create_3zone_masks()")
    print("=" * 60)
    excl_A, excl_B, shared = _create_3zone_masks(
        KNIGHT_BBOXES, ORC_BBOXES, FRAME_RES, LATENT_RES,
        collision_frames=COLLISION_FRAMES
    )
    verify_3zone_masks(excl_A, excl_B, shared)

    # ============================================================
    # Step 2: Load pipeline
    # ============================================================
    print("\n" + "=" * 60)
    print("  STEP 2: Loading AnimateDiff pipeline")
    print("=" * 60)
    pipeline = VideoDCLGPipeline(config, device)
    pipeline.load_pipeline()

    output_dir = os.path.join(base_dir, 'outputs', 'video')
    gif_dir    = os.path.join(base_dir, 'outputs', 'gifs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gif_dir,    exist_ok=True)

    # Save 3-zone mask debug GIF immediately (no generation needed)
    save_debug_3zone_gif(
        excl_A, excl_B, shared,
        os.path.join(gif_dir, "debug_3zone_mask.gif")
    )

    neg_prompt = "blurry, deformed, extra limbs, merged, monochrome, duplicate, static"
    num_frames = NUM_FRAMES  # 8

    # ============================================================
    # Step 3: Dummy run — knight vs orc, lambda=0 (no guidance)
    # This confirms pipeline works; GIFs for all debug outputs
    # ============================================================
    print("\n" + "=" * 60)
    print("  STEP 3: Dummy generation — knight vs orc (lambda=0)")
    print("=" * 60)

    dummy_scenario = {
        'name':     'wrestling',
        'prompt':   "A knight in red armor and an orc with green skin wrestling fiercely, dynamic pose, cinematic lighting",
        'entity_A': 'knight',
        'entity_B': 'orc',
    }

    frames, losses, final_maps = pipeline.generate(
        prompt=dummy_scenario['prompt'],
        negative_prompt=neg_prompt,
        entity_A_word=dummy_scenario['entity_A'],
        entity_B_word=dummy_scenario['entity_B'],
        lambda_max=0.0,
        seed=42,
        num_frames=num_frames,
    )
    print(f"  Generated {len(frames)} frames. Pipeline OK.")

    # 1. Result GIF
    result_gif_path = os.path.join(gif_dir, "wrestling_lam0_result.gif")
    frames[0].save(result_gif_path, save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"  Result GIF: {result_gif_path}")

    # 2. Trajectory overlay GIF
    save_debug_trajectory_gif(
        frames, os.path.join(gif_dir, "debug_trajectory.gif")
    )

    # 3. Attention sweep GIF — collect attn maps from pipeline's hook on last step
    # Build history from final_maps (single-step, not a full sweep)
    attn_map_history = []
    if final_maps.get('map_A') is not None and final_maps.get('map_B') is not None:
        attn_map_history.append({
            "A": final_maps['map_A'],
            "B": final_maps['map_B'],
        })
    save_debug_attn_gif(
        attn_map_history, os.path.join(gif_dir, "debug_attn_sweep.gif")
    )

    # Also save grid + overlay for record
    grid_path = os.path.join(output_dir, "wrestling_lam0_dummy_grid.png")
    save_video_grid(frames, grid_path)
    overlay = create_overlay(frames[0], final_maps.get('map_A'), final_maps.get('map_B'))
    overlay.save(os.path.join(output_dir, "wrestling_lam0_dummy_overlay.png"))

    print(f"\n  All debug GIFs saved to: {gif_dir}/")
    print(f"  dummy_scenario complete — pipeline is healthy.\n")

    # ============================================================
    # Step 4 (optional): Full lambda sweep scenarios (existing code)
    # ============================================================
    run_full_sweep = False  # Set True to run all 5 scenarios x 4 lambdas
    if not run_full_sweep:
        print("  (Skipping full lambda sweep — set run_full_sweep=True to enable)")
        print(f"\nAll done. Dummy run successful.")
        return

    # ============================================================
    # Full sweep (original code kept intact below)
    # ============================================================

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
        {
            'name': 'boxing',
            'prompt': "A man in red boxing gloves and a man in blue boxing gloves punching each other in a boxing ring, intense fight, action shot",
            'entity_A': 'red',
            'entity_B': 'blue',
        },
        {
            'name': 'catdog',
            'prompt': "A cat and a dog fighting playfully on a couch, the cat scratches and the dog bites, cute animals, dynamic motion",
            'entity_A': 'cat',
            'entity_B': 'dog',
        },
    ]

    # Lambda sweep — wider range to find effective guidance strength
    lambda_sweep = [0.0, 10.0, 50.0, 100.0]

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

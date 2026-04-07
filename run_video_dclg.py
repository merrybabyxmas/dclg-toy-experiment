"""
Video DCLG — AnimateDiff + 3-Zone Masked Loss.
Iter 2: Masked spatial loss + lambda sweep + 10 debug GIFs.
"""
import io
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


# ============================================================
# Utility functions
# ============================================================

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

    col_titles = ['Frame Grid (8 frames)', 'First Frame', 'Attention Overlay', 'Loss Curve']
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

    fig.suptitle('Video DCLG: AnimateDiff + 3-Zone Masked Guidance',
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


# ============================================================
# Debug GIF helpers (Iter 1 — 4 GIFs)
# ============================================================

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


# ============================================================
# Debug GIF helpers (Iter 2 — 6 new GIFs)
# ============================================================

def _plot_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def save_debug_loss_curve_gif(loss_histories, labels, path):
    """Save loss curves as an animated GIF (one frame per step, building up).

    Args:
        loss_histories: list of loss value lists (one per lambda run)
        labels: list of label strings matching loss_histories
        path: output GIF path
    """
    if not loss_histories or not any(loss_histories):
        print(f"  Loss curve GIF: skipped (no data)")
        return

    max_steps = max(len(h) for h in loss_histories)
    gif_frames = []

    colors = ['gray', 'green', 'orange', 'red']
    step_stride = max(1, max_steps // 30)  # max ~30 frames for speed

    for step in range(step_stride, max_steps + step_stride, step_stride):
        step = min(step, max_steps)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for hist, label, color in zip(loss_histories, labels, colors[:len(labels)]):
            nonzero = [(i, v) for i, v in enumerate(hist[:step]) if v > 0]
            if nonzero:
                xs, ys = zip(*nonzero)
                ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Denoising Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'DCLG Masked Loss (step {step}/{max_steps})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        gif_frames.append(_plot_to_pil(fig))

    if gif_frames:
        gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                           duration=150, loop=0)
        print(f"  Loss curve GIF: {path}")


def save_debug_chimera_heatmap_gif(per_frame_A_list, per_frame_B_list, path):
    """Save chimera heatmap (m_A * m_B) per frame as animated GIF.

    Args:
        per_frame_A_list: list of numpy arrays [HW] for entity A per frame
        per_frame_B_list: list of numpy arrays [HW] for entity B per frame
        path: output GIF path
    """
    if not per_frame_A_list or not per_frame_B_list:
        print(f"  Chimera heatmap GIF: skipped (no data)")
        return

    gif_frames = []
    for f, (mA_flat, mB_flat) in enumerate(zip(per_frame_A_list, per_frame_B_list)):
        side = int(mA_flat.shape[0] ** 0.5)
        mA = mA_flat.reshape(side, side)
        mB = mB_flat.reshape(side, side)
        chimera = mA * mB

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(mA, cmap='Reds', vmin=0, vmax=mA.max() + 1e-8)
        axes[0].set_title(f'Entity A (f{f})', fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(mB, cmap='Blues', vmin=0, vmax=mB.max() + 1e-8)
        axes[1].set_title(f'Entity B (f{f})', fontsize=9)
        axes[1].axis('off')
        im = axes[2].imshow(chimera, cmap='hot', vmin=0)
        axes[2].set_title(f'Chimera A×B (f{f})', fontsize=9)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        gif_frames.append(_plot_to_pil(fig))

    if gif_frames:
        gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                           duration=300, loop=0)
        print(f"  Chimera heatmap GIF: {path}")


def save_debug_gradient_norm_gif(grad_norm_histories, labels, path):
    """Save gradient norm curves as an animated GIF (building up per step).

    Args:
        grad_norm_histories: list of grad norm lists (one per lambda run)
        labels: list of label strings
        path: output GIF path
    """
    if not grad_norm_histories or not any(grad_norm_histories):
        print(f"  Gradient norm GIF: skipped (no data)")
        return

    max_steps = max(len(h) for h in grad_norm_histories)
    gif_frames = []
    colors = ['gray', 'green', 'orange', 'red']
    step_stride = max(1, max_steps // 30)

    for step in range(step_stride, max_steps + step_stride, step_stride):
        step = min(step, max_steps)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for hist, label, color in zip(grad_norm_histories, labels, colors[:len(labels)]):
            nonzero = [(i, v) for i, v in enumerate(hist[:step]) if v > 0]
            if nonzero:
                xs, ys = zip(*nonzero)
                ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Denoising Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'DCLG Gradient Norm (step {step}/{max_steps})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        gif_frames.append(_plot_to_pil(fig))

    if gif_frames:
        gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                           duration=150, loop=0)
        print(f"  Gradient norm GIF: {path}")


def save_debug_entity_mask_gif(excl_A, excl_B, shared, path):
    """Visualize full entity masks (excl + shared) per frame as GIF.

    Shows each entity's total territory: blue-green for A, orange for B.
    """
    lat_H, lat_W = excl_A.shape[2], excl_A.shape[3]
    vis_H, vis_W = 160, 320
    gif_frames = []

    for t in range(excl_A.shape[0]):
        # Full mask for A = excl_A + shared
        full_A = np.clip(
            excl_A[t, 0].numpy() + shared[t, 0].numpy(), 0, 1
        )
        # Full mask for B = excl_B + shared
        full_B = np.clip(
            excl_B[t, 0].numpy() + shared[t, 0].numpy(), 0, 1
        )

        # Colorize A and B
        img_A = colorize_map(full_A, "Blues").resize((vis_W // 2, vis_H), Image.NEAREST)
        img_B = colorize_map(full_B, "Oranges").resize((vis_W // 2, vis_H), Image.NEAREST)

        # Combine side by side
        combined = Image.new('RGB', (vis_W, vis_H + 20), (20, 20, 20))
        combined.paste(img_A, (0, 20))
        combined.paste(img_B, (vis_W // 2, 20))
        draw = ImageDraw.Draw(combined)
        draw.text((4, 2), f"Entity A (excl+shared) — f{t}", fill=(100, 180, 255))
        draw.text((vis_W // 2 + 4, 2), f"Entity B (excl+shared) — f{t}", fill=(255, 160, 80))
        if t in COLLISION_FRAMES:
            draw.text((vis_W // 2 - 30, 2), "COLLISION", fill=(255, 220, 0))
        gif_frames.append(combined)

    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                       duration=250, loop=0)
    print(f"  Entity mask GIF: {path}")


def save_debug_attn_step_t_gif(attn_maps_at_t, frames, output_path,
                               entity_A_idx, entity_B_idx, num_heads=8):
    """특정 스텝 t에서의 프레임별 어텐션 맵을 프레임별로 overlay하여 GIF 저장."""
    if not attn_maps_at_t:
        print("    [Debug] No attention maps provided for step_t GIF.")
        return
    if entity_A_idx < 0 or entity_B_idx < 0:
        print(f"    [Debug] Invalid token indices A={entity_A_idx} B={entity_B_idx}; skipping.")
        return

    name = list(attn_maps_at_t.keys())[-1]
    attn_map = attn_maps_at_t[name]  # [2*F*heads, HW, 77]
    num_frames_val = len(frames)

    half = attn_map.shape[0] // 2
    cond_attn = attn_map[half:]  # [F*heads, HW, 77]
    hw = cond_attn.shape[1]
    total_f_heads = cond_attn.shape[0]
    actual_heads = total_f_heads // num_frames_val
    if actual_heads == 0:
        print("    [Debug] Cannot determine num_heads; skipping step_t GIF.")
        return

    try:
        cond_attn = cond_attn.view(num_frames_val, actual_heads, hw, -1)
    except RuntimeError as e:
        print(f"    [Debug] Reshape failed: {e}; skipping step_t GIF.")
        return

    overlay_frames = []
    for f in range(num_frames_val):
        map_A = cond_attn[f, :, :, entity_A_idx].mean(dim=0).cpu().float().numpy()
        map_B = cond_attn[f, :, :, entity_B_idx].mean(dim=0).cpu().float().numpy()
        overlay = create_overlay(frames[f], map_A, map_B, alpha=0.5)
        overlay_frames.append(overlay)

    save_video_gif(overlay_frames, output_path, duration=150)
    print(f"    [Debug GIF] Attn at step t saved: {output_path}")


def save_lambda_comparison_gif(scenario_name, lambda_frames_dict, path, duration=200):
    """Save side-by-side comparison of all lambda results as animated GIF.

    Args:
        scenario_name: name string for title
        lambda_frames_dict: dict {lambda_val: [PIL frames]} ordered by lambda
        path: output GIF path
        duration: ms per GIF frame
    """
    lambdas = sorted(lambda_frames_dict.keys())
    if not lambdas:
        print(f"  Lambda comparison GIF: skipped (no data)")
        return

    all_seqs = [lambda_frames_dict[lam] for lam in lambdas]
    num_video_frames = min(len(seq) for seq in all_seqs)
    frame_w, frame_h = all_seqs[0][0].size

    label_h = 24
    n_cols = len(lambdas)
    gap = 4
    combined_w = n_cols * frame_w + (n_cols - 1) * gap
    combined_h = frame_h + label_h

    gif_frames = []
    for t in range(num_video_frames):
        canvas = Image.new('RGB', (combined_w, combined_h), (15, 15, 15))
        draw = ImageDraw.Draw(canvas)
        for col_idx, (lam, seq) in enumerate(zip(lambdas, all_seqs)):
            x_off = col_idx * (frame_w + gap)
            canvas.paste(seq[t], (x_off, label_h))
            lam_str = f"λ={lam:.0f}" if lam > 0 else "λ=0 (base)"
            draw.text((x_off + 4, 4), lam_str, fill=(255, 255, 255))
        draw.text((combined_w - 60, 4), f"f{t}/{num_video_frames-1}",
                  fill=(200, 200, 200))
        gif_frames.append(canvas)

    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                       duration=duration, loop=0)
    print(f"  Lambda comparison GIF [{scenario_name}]: {path}")


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

    # Save entity full-mask GIF (excl + shared per entity)
    save_debug_entity_mask_gif(
        excl_A, excl_B, shared,
        os.path.join(gif_dir, "debug_entity_mask.gif")
    )

    neg_prompt = "blurry, deformed, extra limbs, merged, monochrome, duplicate, static"
    num_frames = NUM_FRAMES  # 8

    # ============================================================
    # Step 3: Dummy run — knight vs orc, lambda=0 (no guidance)
    # Confirms pipeline health + generates initial debug GIFs.
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

    frames, losses, final_maps, debug_info = pipeline.generate(
        prompt=dummy_scenario['prompt'],
        negative_prompt=neg_prompt,
        entity_A_word=dummy_scenario['entity_A'],
        entity_B_word=dummy_scenario['entity_B'],
        lambda_max=0.0,
        seed=42,
        num_frames=num_frames,
        bboxes_A=KNIGHT_BBOXES,
        bboxes_B=ORC_BBOXES,
        collision_frames=COLLISION_FRAMES,
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

    # 3. Attention sweep GIF — from final_maps
    attn_map_history = []
    if final_maps.get('map_A') is not None and final_maps.get('map_B') is not None:
        attn_map_history.append({
            "A": final_maps['map_A'],
            "B": final_maps['map_B'],
        })
    save_debug_attn_gif(
        attn_map_history, os.path.join(gif_dir, "debug_attn_sweep.gif")
    )

    # 4. Chimera heatmap GIF (per-frame m_A * m_B from final maps)
    save_debug_chimera_heatmap_gif(
        final_maps.get('per_frame_A', []),
        final_maps.get('per_frame_B', []),
        os.path.join(gif_dir, "debug_chimera_heatmap.gif")
    )

    # Also save grid + overlay for record
    grid_path = os.path.join(output_dir, "wrestling_lam0_dummy_grid.png")
    save_video_grid(frames, grid_path)
    overlay = create_overlay(frames[0], final_maps.get('map_A'), final_maps.get('map_B'))
    overlay.save(os.path.join(output_dir, "wrestling_lam0_dummy_overlay.png"))

    print(f"\n  Initial debug GIFs saved to: {gif_dir}/")
    print(f"  Dummy run complete — pipeline is healthy.\n")

    # ============================================================
    # Step 4: Full lambda sweep
    # ============================================================
    run_full_sweep = True   # Iter 2: enabled
    if not run_full_sweep:
        print("  (Skipping full lambda sweep)")
        print(f"\nAll done. Dummy run successful.")
        return

    # Test scenarios with heavy entity intersection
    # Iter 2 initial run: wrestling (primary) + grapple to validate masked loss
    # Full 5-scenario run can be done by adding back swordfight/boxing/catdog
    scenarios = [
        {
            'name': 'wrestling',
            'prompt': "A knight in red armor and an orc with green skin wrestling fiercely, dynamic pose, cinematic lighting",
            'entity_A': 'knight',
            'entity_B': 'orc',
        },
        {
            'name': 'grapple',
            'prompt': "A red dragon and a blue dragon grappling each other in mid-air, their bodies intertwined, fire and ice, epic fantasy",
            'entity_A': 'red',
            'entity_B': 'blue',
        },
        {
            'name': 'swordfight',
            'prompt': "A samurai in black robes and a ninja in white robes engaged in an intense swordfight, fast motion blur, dramatic lighting",
            'entity_A': 'samurai',
            'entity_B': 'ninja',
        },
        {
            'name': 'boxing',
            'prompt': "A boxer wearing red gloves and a boxer wearing blue gloves exchanging powerful punches in a boxing ring, action shot",
            'entity_A': 'red',
            'entity_B': 'blue',
        },
        {
            'name': 'catdog',
            'prompt': "A golden retriever dog and an orange tabby cat playfully chasing each other around a living room, cute and energetic",
            'entity_A': 'dog',
            'entity_B': 'cat',
        },
    ]

    # Iter 2: full lambda sweep [0, 10, 50, 100]
    lambda_sweep = [0.0, 10.0, 50.0, 100.0]

    # Accumulators for cross-scenario loss/grad debug GIFs
    all_loss_histories = []
    all_grad_histories = []
    all_lambda_labels  = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"  PROMPT: {scenario['prompt']}")
        print(f"{'='*60}")

        results = []
        scenario_loss_hists = []
        scenario_grad_hists = []
        scenario_lambda_labels = []
        lambda_frames_dict = {}   # for lambda comparison GIF

        for lam in lambda_sweep:
            label = f"{scenario['name']}_lam{lam:.0f}"
            print(f"\n  {label}")

            try:
                frames, losses, final_maps, debug_info = pipeline.generate(
                    prompt=scenario['prompt'],
                    negative_prompt=neg_prompt,
                    entity_A_word=scenario['entity_A'],
                    entity_B_word=scenario['entity_B'],
                    lambda_max=lam,
                    seed=42,
                    num_frames=num_frames,
                    bboxes_A=KNIGHT_BBOXES,
                    bboxes_B=ORC_BBOXES,
                    collision_frames=COLLISION_FRAMES,
                )

                # Save individual GIF
                gif_path = os.path.join(gif_dir, f'{label}.gif')
                save_video_gif(frames, gif_path)
                print(f"    GIF: {gif_path}")

                # Save debug_attn_step_5 GIF (only when guidance is active)
                if lam > 0:
                    attn_maps_at_t = debug_info.get('attn_maps_at_step_t')
                    idx_A = pipeline.get_token_index(scenario['prompt'], scenario['entity_A'])
                    idx_B = pipeline.get_token_index(scenario['prompt'], scenario['entity_B'])
                    attn_step_path = os.path.join(
                        gif_dir, f'debug_attn_step_5_{scenario["name"]}_lam{lam:.0f}.gif'
                    )
                    save_debug_attn_step_t_gif(
                        attn_maps_at_t, frames, attn_step_path, idx_A, idx_B
                    )

                # Save frame grid + overlay
                grid_path = os.path.join(output_dir, f'{label}_grid.png')
                save_video_grid(frames, grid_path)
                overlay = create_overlay(frames[0], final_maps.get('map_A'), final_maps.get('map_B'))
                overlay.save(os.path.join(output_dir, f'{label}_overlay.png'))

                # Accumulate for debug GIFs
                scenario_loss_hists.append(losses)
                scenario_grad_hists.append(debug_info.get('grad_norms', []))
                scenario_lambda_labels.append(label)
                lambda_frames_dict[lam] = frames

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

        # --- Per-scenario debug GIFs ---

        # 5. Loss curve GIF for this scenario
        if scenario_loss_hists:
            save_debug_loss_curve_gif(
                scenario_loss_hists,
                scenario_lambda_labels,
                os.path.join(gif_dir, f"debug_loss_curve_{scenario['name']}.gif")
            )

        # 6. Gradient norm GIF for this scenario
        if scenario_grad_hists:
            save_debug_gradient_norm_gif(
                scenario_grad_hists,
                scenario_lambda_labels,
                os.path.join(gif_dir, f"debug_gradient_norm_{scenario['name']}.gif")
            )

        # 7. Lambda comparison GIF
        if lambda_frames_dict:
            save_lambda_comparison_gif(
                scenario['name'],
                lambda_frames_dict,
                os.path.join(gif_dir, f"lambda_comparison_{scenario['name']}.gif")
            )

        # Comparison report
        if results:
            create_report(results, output_dir, f'VIDEO_DCLG_{scenario["name"].upper()}_REPORT')

        # Accumulate for a single combined loss/grad debug GIF (optional, all scenarios)
        all_loss_histories.extend(scenario_loss_hists)
        all_grad_histories.extend(scenario_grad_hists)
        all_lambda_labels.extend(scenario_lambda_labels)

    # 8. Combined loss curve GIF (all scenarios, wrestling scenario only for clarity)
    # We already saved per-scenario; optionally save the wrestling curves as the primary debug GIF
    wrestling_idx = [i for i, s in enumerate(scenarios) if s['name'] == 'wrestling']
    if wrestling_idx:
        ridx = wrestling_idx[0]
        n_lam = len(lambda_sweep)
        subset_losses = all_loss_histories[ridx * n_lam:(ridx + 1) * n_lam]
        subset_grads  = all_grad_histories[ridx * n_lam:(ridx + 1) * n_lam]
        subset_labels = all_lambda_labels[ridx * n_lam:(ridx + 1) * n_lam]

        save_debug_loss_curve_gif(
            subset_losses, subset_labels,
            os.path.join(gif_dir, "debug_loss_curve.gif")
        )
        save_debug_gradient_norm_gif(
            subset_grads, subset_labels,
            os.path.join(gif_dir, "debug_gradient_norm.gif")
        )

    print(f"\n{'='*60}")
    print(f"  All done. Results in {output_dir}/")
    print(f"  GIFs in {gif_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

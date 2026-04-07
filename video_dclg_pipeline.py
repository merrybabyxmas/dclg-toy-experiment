"""
Video DCLG Pipeline — AnimateDiff + 3-Zone Masked Spatial Loss.

Extends the original Phase 1 DCLG to video:
- AnimateDiff motion adapter on SD 1.5 backbone
- 3-zone masked loss: leakage + co-activation + erasure penalty
- Gradient guidance on noise_pred (not latents)
- Captures cross-attention maps from up_blocks attn2 layers

Iter 2: Replace global cosine similarity with spatial-aware masked loss.
"""
import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np


def bbox_to_mask(bbox, frame_resolution, latent_resolution):
    """Convert a single BBox [x1, y1, x2, y2] to a binary mask at latent resolution.

    Args:
        bbox: [x1, y1, x2, y2] in frame pixel coords
        frame_resolution: (H, W) of the output frame
        latent_resolution: (H_lat, W_lat) for the mask

    Returns:
        mask: float32 tensor of shape [H_lat, W_lat]
    """
    x1, y1, x2, y2 = bbox
    frame_H, frame_W = frame_resolution
    lat_H, lat_W = latent_resolution

    scale_x = lat_W / frame_W
    scale_y = lat_H / frame_H

    lx1 = int(x1 * scale_x)
    ly1 = int(y1 * scale_y)
    lx2 = int(x2 * scale_x)
    ly2 = int(y2 * scale_y)

    # Clamp to valid range
    lx1 = max(0, min(lx1, lat_W - 1))
    ly1 = max(0, min(ly1, lat_H - 1))
    lx2 = max(lx1 + 1, min(lx2, lat_W))
    ly2 = max(ly1 + 1, min(ly2, lat_H))

    mask = torch.zeros(lat_H, lat_W, dtype=torch.float32)
    mask[ly1:ly2, lx1:lx2] = 1.0
    return mask


def _create_3zone_masks(bboxes_A, bboxes_B, frame_resolution, latent_resolution,
                        collision_frames=(6, 7)):
    """Create per-frame 3-zone masks for entity A and entity B.

    For collision frames: produces exclusive_A, shared, exclusive_B zones.
    For non-collision frames: shared zone is all zeros (no overlap).

    Args:
        bboxes_A: list of [x1, y1, x2, y2] for entity A, one per frame
        bboxes_B: list of [x1, y1, x2, y2] for entity B, one per frame
        frame_resolution: (H, W) of the generated video frames
        latent_resolution: (H_lat, W_lat) of the UNet latent space
        collision_frames: frame indices where entities physically overlap

    Returns:
        mask_exclusive_A: [num_frames, 1, H_lat, W_lat] float32
        mask_exclusive_B: [num_frames, 1, H_lat, W_lat] float32
        mask_shared_AB:   [num_frames, 1, H_lat, W_lat] float32
    """
    num_frames = len(bboxes_A)
    assert len(bboxes_B) == num_frames, "bboxes_A and bboxes_B must have same length"

    lat_H, lat_W = latent_resolution
    excl_A_frames = []
    excl_B_frames = []
    shared_frames = []

    for f in range(num_frames):
        mask_A_raw = bbox_to_mask(bboxes_A[f], frame_resolution, latent_resolution)
        mask_B_raw = bbox_to_mask(bboxes_B[f], frame_resolution, latent_resolution)

        if f in collision_frames:
            mask_intersection = mask_A_raw * mask_B_raw
            excl_A = torch.clamp(mask_A_raw - mask_intersection, 0.0, 1.0)
            excl_B = torch.clamp(mask_B_raw - mask_intersection, 0.0, 1.0)
            shared  = mask_intersection
        else:
            # Simple left/right split — no shared zone
            excl_A = mask_A_raw
            excl_B = mask_B_raw
            shared  = torch.zeros(lat_H, lat_W, dtype=torch.float32)

        excl_A_frames.append(excl_A.unsqueeze(0))   # [1, H, W]
        excl_B_frames.append(excl_B.unsqueeze(0))
        shared_frames.append(shared.unsqueeze(0))

    mask_exclusive_A = torch.stack(excl_A_frames, dim=0)   # [F, 1, H, W]
    mask_exclusive_B = torch.stack(excl_B_frames, dim=0)
    mask_shared_AB   = torch.stack(shared_frames, dim=0)

    return mask_exclusive_A, mask_exclusive_B, mask_shared_AB


class VideoSaveAttnProcessor:
    """Captures text cross-attention maps with float32 attention computation
    for numerical stability in gradient tracking."""

    def __init__(self):
        self.attn_map = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, kwargs.get("temb"))

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to [B, heads, HW, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute attention in float32 for gradient stability
        scale_factor = head_dim ** -0.5
        attn_logits = torch.matmul(
            query.float(), key.float().transpose(-2, -1)
        ) * scale_factor

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attn_logits = attn_logits + attention_mask

        attention_probs = attn_logits.softmax(dim=-1)

        # Capture text cross-attention (seq_len <= 77)
        if attention_probs.shape[-1] <= 77:
            # Reshape to [B*heads, HW, seq_len] for compatibility with loss computation
            hw = attention_probs.shape[2]
            seq_len = attention_probs.shape[3]
            self.attn_map = attention_probs.reshape(
                batch_size * attn.heads, hw, seq_len
            )

        # Apply attention to values (back to original dtype)
        hidden_states = torch.matmul(attention_probs.to(value.dtype), value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class VideoHookManager:
    """Hook manager for AnimateDiff UNet cross-attention layers."""

    def __init__(self, target_min_hw=256):
        self.target_min_hw = target_min_hw
        self.processors = {}

    def register_hooks(self, unet):
        for name, module in unet.named_modules():
            # Target up_blocks cross-attention (attn2), skip motion_modules
            if ("up_blocks" in name and "attn2" in name
                    and "motion_modules" not in name
                    and hasattr(module, "processor")):
                if not name.endswith(".attn2"):
                    continue
                proc = VideoSaveAttnProcessor()
                module.set_processor(proc)
                self.processors[name] = proc

    def get_captured_maps(self):
        maps = {}
        for name, proc in self.processors.items():
            if proc.attn_map is not None and proc.attn_map.shape[1] >= self.target_min_hw:
                maps[name] = proc.attn_map
        return maps

    def clear(self):
        for proc in self.processors.values():
            proc.attn_map = None


class VideoDCLGPipeline:
    """AnimateDiff pipeline with DCLG 3-zone masked guidance."""

    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.pipe = None
        self.hook_manager = None

    def load_pipeline(self):
        print("Loading AnimateDiff pipeline...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            torch_dtype=torch.float16,
        )
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
        )
        self.pipe = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=adapter,
            vae=vae,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.enable_slicing()
        # NOTE: Do NOT enable gradient_checkpointing — it recomputes forward during
        # backward, overwriting stored attention maps and breaking the loss computation graph.

        # Register attention hooks
        self.hook_manager = VideoHookManager(
            target_min_hw=self.config['dclg']['target_min_hw']
        )
        self.hook_manager.register_hooks(self.pipe.unet)
        print(f"  Hooked {len(self.hook_manager.processors)} cross-attention layers")

    def get_token_index(self, prompt, target_word):
        inputs = self.pipe.tokenizer(prompt)
        tokens = self.pipe.tokenizer.convert_ids_to_tokens(inputs['input_ids'])
        for i, token in enumerate(tokens):
            clean = token.replace('</w>', '').lower()
            if clean == target_word.lower():
                return i
        return -1

    def compute_chimera_loss(self, captured_maps, idx_A, idx_B, num_frames):
        """Cosine similarity loss across video frames (legacy, kept for fallback).

        captured_maps values have shape [2*F*heads, HW, 77].
        We take the conditional half, reshape to [F, heads, HW, 77],
        average over heads and frames, then compute cosine similarity.
        """
        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            # attn_map: [2*F*heads, HW, seq_len]
            total_batch_heads = attn_map.shape[0]
            half = total_batch_heads // 2
            cond_attn = attn_map[half:]  # [F*heads, HW, seq_len]

            # Average over all frames and heads
            map_A = cond_attn[:, :, idx_A].mean(dim=0)  # [HW]
            map_B = cond_attn[:, :, idx_B].mean(dim=0)  # [HW]

            # Sharpening
            map_A = torch.pow(map_A, 2)
            map_B = torch.pow(map_B, 2)

            # Normalize
            map_A = map_A / (map_A.max() + 1e-8)
            map_B = map_B / (map_B.max() + 1e-8)

            # Cosine similarity (minimize overlap)
            cos_sim = F.cosine_similarity(map_A.view(-1), map_B.view(-1), dim=0)

            # Erasure penalty (prevent entity disappearance)
            erasure = torch.relu(0.5 - map_A.max()) + torch.relu(0.5 - map_B.max())

            total_loss += (cos_sim + erasure)

        return total_loss / num_layers

    def compute_per_frame_loss(self, captured_maps, idx_A, idx_B, num_frames, num_heads):
        """Per-frame cosine similarity loss for stronger per-frame guidance (legacy)."""
        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            total_batch_heads = attn_map.shape[0]
            half = total_batch_heads // 2
            cond_attn = attn_map[half:]  # [F*heads, HW, seq_len]

            # Reshape to [F, heads, HW, seq_len]
            hw = cond_attn.shape[1]
            seq_len = cond_attn.shape[2]
            cond_attn = cond_attn.view(num_frames, num_heads, hw, seq_len)

            # Per-frame loss
            frame_loss = 0.0
            for f in range(num_frames):
                frame_attn = cond_attn[f]  # [heads, HW, seq_len]
                map_A = frame_attn[:, :, idx_A].mean(dim=0)  # [HW]
                map_B = frame_attn[:, :, idx_B].mean(dim=0)

                map_A = torch.pow(map_A, 2)
                map_B = torch.pow(map_B, 2)
                map_A = map_A / (map_A.max() + 1e-8)
                map_B = map_B / (map_B.max() + 1e-8)

                cos_sim = F.cosine_similarity(map_A.view(-1), map_B.view(-1), dim=0)
                erasure = torch.relu(0.5 - map_A.max()) + torch.relu(0.5 - map_B.max())
                frame_loss += (cos_sim + erasure)

            total_loss += frame_loss / num_frames

        return total_loss / num_layers

    def compute_masked_loss(self, captured_maps, idx_A, idx_B, masks, num_frames, num_heads):
        """3-zone spatially-aware DCLG loss (Iter 2).

        Core philosophy: allow physical interaction, prevent identity overlap.

        Args:
            captured_maps: dict of {name: attn_map [2*F*heads, HW, seq_len]}
            idx_A, idx_B: token indices for entity A and B
            masks: dict with keys 'excl_A', 'excl_B', 'shared' each [F, 1, H_lat, W_lat]
            num_frames: number of video frames
            num_heads: number of attention heads

        Loss components:
            1. Leakage Loss: penalize entity A attention in B's exclusive zone (and vice versa)
            2. Co-activation Loss: penalize simultaneous A+B activation in shared zone (chimera)
            3. Activation Reward: prevent entity erasure (F2 failure mode)
        """
        excl_A_mask = masks['excl_A']  # [F, 1, H_lat, W_lat]
        excl_B_mask = masks['excl_B']
        shared_mask = masks['shared']

        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            total_batch_heads = attn_map.shape[0]
            half = total_batch_heads // 2
            cond_attn = attn_map[half:]  # [F*heads, HW, seq_len]

            hw = cond_attn.shape[1]
            seq_len = cond_attn.shape[2]

            # Infer actual num_heads per frame
            total_f_heads = cond_attn.shape[0]
            actual_heads = total_f_heads // num_frames
            if actual_heads == 0:
                continue

            # Reshape to [F, heads, HW, seq_len]
            try:
                cond_4d = cond_attn.view(num_frames, actual_heads, hw, seq_len)
            except RuntimeError:
                continue

            # Compute HW spatial side (attention maps are square)
            hw_side = int(hw ** 0.5)
            if hw_side * hw_side != hw:
                # Non-square attention map; fallback to global loss for this layer
                continue

            layer_loss = 0.0
            for f in range(num_frames):
                frame_attn = cond_4d[f]  # [heads, HW, seq_len]

                # Head-averaged attention maps for tokens A and B
                map_A_f = frame_attn[:, :, idx_A].mean(dim=0)  # [HW]
                map_B_f = frame_attn[:, :, idx_B].mean(dim=0)  # [HW]

                # Get zone masks for this frame → resize to attention map resolution
                excl_A_f = excl_A_mask[f, 0]  # [H_lat, W_lat]
                excl_B_f = excl_B_mask[f, 0]
                shared_f  = shared_mask[f, 0]

                lat_H, lat_W = excl_A_f.shape
                if lat_H != hw_side or lat_W != hw_side:
                    excl_A_f = F.interpolate(
                        excl_A_f.unsqueeze(0).unsqueeze(0),
                        size=(hw_side, hw_side), mode='nearest'
                    ).squeeze()
                    excl_B_f = F.interpolate(
                        excl_B_f.unsqueeze(0).unsqueeze(0),
                        size=(hw_side, hw_side), mode='nearest'
                    ).squeeze()
                    shared_f = F.interpolate(
                        shared_f.unsqueeze(0).unsqueeze(0),
                        size=(hw_side, hw_side), mode='nearest'
                    ).squeeze()

                # Flatten masks to [HW] and move to device
                excl_A_f = excl_A_f.to(self.device).view(-1).float()
                excl_B_f = excl_B_f.to(self.device).view(-1).float()
                shared_f  = shared_f.to(self.device).view(-1).float()

                # 1. Leakage Loss: penalize A leaking into B's exclusive zone (& vice versa)
                loss_leak_A = (map_A_f * excl_B_f).mean()
                loss_leak_B = (map_B_f * excl_A_f).mean()

                # 2. Co-activation Loss: penalize chimera in shared zone
                co_act_map = (map_A_f * shared_f) * (map_B_f * shared_f)
                loss_coact = co_act_map.sum() / (shared_f.sum() + 1e-8)

                # 3. Activation Reward: prevent entity erasure (F2)
                # Each entity must activate in its full region (exclusive + shared)
                full_mask_A = torch.clamp(excl_A_f + shared_f, 0.0, 1.0)
                full_mask_B = torch.clamp(excl_B_f + shared_f, 0.0, 1.0)
                loss_act = (torch.relu(0.3 - (map_A_f * full_mask_A).max()) +
                            torch.relu(0.3 - (map_B_f * full_mask_B).max()))

                frame_loss = (loss_leak_A + loss_leak_B) + 0.3 * loss_coact + 0.5 * loss_act
                layer_loss += frame_loss / num_frames

            total_loss += layer_loss

        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss / num_layers

    def generate(
        self,
        prompt,
        negative_prompt,
        entity_A_word,
        entity_B_word,
        lambda_max=50.0,
        seed=42,
        num_frames=16,
        per_frame_loss=False,
        bboxes_A=None,
        bboxes_B=None,
        collision_frames=(6, 7),
    ):
        """Generate video with DCLG guidance.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            entity_A_word: Word for entity A (e.g., "knight")
            entity_B_word: Word for entity B (e.g., "orc")
            lambda_max: Maximum guidance strength
            seed: Random seed
            num_frames: Number of video frames
            per_frame_loss: If True, compute legacy per-frame cosine loss
            bboxes_A: list of [x1,y1,x2,y2] per frame for entity A (enables masked loss)
            bboxes_B: list of [x1,y1,x2,y2] per frame for entity B
            collision_frames: tuple of frame indices where entities overlap

        Returns:
            frames: list of PIL images
            losses: list of per-step loss values
            final_maps: dict with attention maps for visualization
            debug_info: dict with grad_norms, masks
        """
        num_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']
        tau_threshold = self.config['dclg']['tau_threshold']
        grad_clip = self.config['dclg'].get('grad_clip', 1.0)
        img_size = self.config['generation']['image_size']

        # Token indices
        idx_A = self.get_token_index(prompt, entity_A_word)
        idx_B = self.get_token_index(prompt, entity_B_word)
        tokens = self.pipe.tokenizer(prompt)
        token_strs = self.pipe.tokenizer.convert_ids_to_tokens(tokens['input_ids'])
        print(f"  Tokens: {list(enumerate(token_strs))}")
        print(f"  {entity_A_word} idx: {idx_A}, {entity_B_word} idx: {idx_B}")

        if idx_A < 0 or idx_B < 0:
            raise ValueError(f"Could not find token indices: {entity_A_word}={idx_A}, {entity_B_word}={idx_B}")

        # Encode prompt
        text_cond, text_uncond = self.pipe.encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )

        # Prepare latents [B, C, F, H, W]
        device = torch.device(self.device)
        generator = torch.Generator(device).manual_seed(seed)
        latent_h = img_size // 8
        latent_w = img_size // 8
        latents = torch.randn(
            1, 4, num_frames, latent_h, latent_w,
            generator=generator, device=device, dtype=torch.float32,
        )

        self.pipe.scheduler.set_timesteps(num_steps, device=device)

        # SD 1.5 has 8 heads in most attention layers
        num_heads = 8

        # Create 3-zone masks from BBox trajectories (if provided)
        masks = None
        if bboxes_A is not None and bboxes_B is not None:
            excl_A, excl_B, shared = _create_3zone_masks(
                bboxes_A, bboxes_B,
                (img_size, img_size),
                (latent_h, latent_w),
                collision_frames=collision_frames,
            )
            masks = {'excl_A': excl_A, 'excl_B': excl_B, 'shared': shared}
            print(f"  Using 3-zone masked loss (excl_A, excl_B, shared zones)")
        else:
            print(f"  Using {'per-frame' if per_frame_loss else 'global'} cosine loss (no bbox masks)")

        losses = []
        grad_norms = []
        attn_maps_at_step_t = None
        target_step_for_debug = 5

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            apply_guidance = (i < tau_threshold and lambda_max > 0)

            latents = latents.detach().to(torch.float32)
            if apply_guidance:
                latents = latents.requires_grad_(True)

            self.hook_manager.clear()

            # CFG: duplicate latents
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            # AnimateDiff UNet merges frames into batch dim internally.
            # encoder_hidden_states must be repeated per frame so shapes align:
            # [2, 77, 768] → [2*F, 77, 768]
            enc_hs = torch.cat([text_uncond, text_cond], dim=0).to(torch.float32)
            enc_hs_rep = enc_hs.repeat_interleave(num_frames, dim=0)

            # UNet forward
            noise_pred_out = self.pipe.unet(
                latent_input.to(self.pipe.unet.dtype), t,
                encoder_hidden_states=enc_hs_rep.to(self.pipe.unet.dtype),
            ).sample.to(torch.float32)

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if apply_guidance:
                captured = self.hook_manager.get_captured_maps()

                # Capture attention maps at target step for debug GIF
                if i == target_step_for_debug and captured:
                    attn_maps_at_step_t = {k: v.detach().clone() for k, v in captured.items()}

                if not captured:
                    losses.append(0.0)
                    grad_norms.append(0.0)
                    latents = latents.detach()
                    latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                    continue

                # Select loss function: masked > per_frame > global
                if masks is not None:
                    loss = self.compute_masked_loss(
                        captured, idx_A, idx_B, masks, num_frames, num_heads
                    )
                elif per_frame_loss:
                    loss = self.compute_per_frame_loss(
                        captured, idx_A, idx_B, num_frames, num_heads
                    )
                else:
                    loss = self.compute_chimera_loss(
                        captured, idx_A, idx_B, num_frames
                    )
                losses.append(loss.item())

                grad = torch.autograd.grad(loss, latents, allow_unused=True)[0]
                if grad is None:
                    print(f"    Step {i}: grad is None (no gradient flow)")
                    grad_norms.append(0.0)
                    latents = latents.detach()
                    latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
                    continue

                gn = grad.norm().item()
                grad_norms.append(gn)
                if i == 0 or i == tau_threshold - 1:
                    print(f"    Step {i}: loss={loss.item():.4f}, grad_norm={gn:.6f}")

                if grad_clip:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                decay = 1.0 - i / tau_threshold
                noise_pred = noise_pred + lambda_max * decay * grad
                latents = latents.detach()
            else:
                losses.append(0.0)
                grad_norms.append(0.0)
                latents = latents.detach()

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode video
        with torch.no_grad():
            latents_decode = latents.to(self.pipe.vae.dtype)
            frames = self.decode_latents(latents_decode)

        # Get final attention maps for visualization
        final_maps = self._get_final_maps(latents, text_cond, text_uncond,
                                           self.pipe.scheduler.timesteps[-1],
                                           idx_A, idx_B, num_frames)

        debug_info = {
            'grad_norms': grad_norms,
            'masks': masks,
            'attn_maps_at_step_t': attn_maps_at_step_t,
        }

        return frames, losses, final_maps, debug_info

    def decode_latents(self, latents):
        """Decode [B, C, F, H, W] latents to list of PIL images."""
        # latents: [1, 4, F, H, W]
        latents = latents / self.pipe.vae.config.scaling_factor
        batch_size, channels, num_frames, height, width = latents.shape

        # Reshape to [B*F, C, H, W] for VAE
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        frames = []
        # Decode in chunks to avoid OOM
        chunk_size = 4
        for i in range(0, latents.shape[0], chunk_size):
            chunk = latents[i:i + chunk_size]
            decoded = self.pipe.vae.decode(chunk).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
            decoded = (decoded * 255).round().astype(np.uint8)
            for j in range(decoded.shape[0]):
                frames.append(Image.fromarray(decoded[j]))

        return frames

    def _get_final_maps(self, latents, text_cond, text_uncond, t, idx_A, idx_B, num_frames):
        """Run one more forward pass to get clean attention maps for visualization."""
        with torch.no_grad():
            self.hook_manager.clear()
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            enc_hs = torch.cat([text_uncond, text_cond], dim=0).to(torch.float32)
            enc_hs_rep = enc_hs.repeat_interleave(num_frames, dim=0)
            self.pipe.unet(
                latent_input.to(self.pipe.unet.dtype), t,
                encoder_hidden_states=enc_hs_rep.to(self.pipe.unet.dtype),
            )

        captured = self.hook_manager.get_captured_maps()
        if not captured:
            return {}

        name = list(captured.keys())[-1]
        attn_map = captured[name]
        half = attn_map.shape[0] // 2
        cond_attn = attn_map[half:]  # [F*heads, HW, 77]

        # Frame-averaged maps
        map_A = cond_attn[:, :, idx_A].mean(dim=0).cpu().float().numpy()
        map_B = cond_attn[:, :, idx_B].mean(dim=0).cpu().float().numpy()

        # Per-frame maps for chimera heatmap visualization
        hw = cond_attn.shape[1]
        total_f_heads = cond_attn.shape[0]
        actual_heads = total_f_heads // num_frames
        per_frame_A = []
        per_frame_B = []
        try:
            cond_4d = cond_attn.view(num_frames, actual_heads, hw, -1)
            for f in range(num_frames):
                pA = cond_4d[f, :, :, idx_A].mean(0).cpu().float().numpy()
                pB = cond_4d[f, :, :, idx_B].mean(0).cpu().float().numpy()
                per_frame_A.append(pA)
                per_frame_B.append(pB)
        except Exception:
            per_frame_A = [map_A] * num_frames
            per_frame_B = [map_B] * num_frames

        return {
            'map_A': map_A,
            'map_B': map_B,
            'per_frame_A': per_frame_A,
            'per_frame_B': per_frame_B,
        }

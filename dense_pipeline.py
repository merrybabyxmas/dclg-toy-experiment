"""
Dense Diffusion DCLG Pipeline (text-only, no IP-Adapter).

Pure text attention masking + DCLG gradient guidance.
Supports both non-overlapping (left/right) and overlapping (3-zone) modes.
"""
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
import os

from dense_attn_processor import DenseAttnProcessor


class DenseDCLGPipeline:
    def __init__(self, pipe, config, device="cuda"):
        self.pipe = pipe
        self.config = config
        self.device = device
        self.processors = {}
        self._setup_processors()

    def _setup_processors(self):
        unet = self.pipe.unet
        attn_procs = {}

        for name in unet.attn_processors.keys():
            is_cross = not name.endswith("attn1.processor")
            if is_cross:
                proc = DenseAttnProcessor()
                attn_procs[name] = proc
                self.processors[name] = proc
            else:
                # Self-attention: use default
                from diffusers.models.attention_processor import AttnProcessor2_0
                attn_procs[name] = AttnProcessor2_0()

        unet.set_attn_processor(attn_procs)

    def _create_masks(self, hw, overlap_width=0.0):
        """Create region masks.

        overlap_width=0: non-overlapping (left/right with gap)
        overlap_width>0: 3-zone (exclusive_A / shared / exclusive_B)
        """
        h = w = int(hw ** 0.5)
        mask_A = torch.zeros(h, w, device=self.device)
        mask_B = torch.zeros(h, w, device=self.device)
        mask_shared = None

        if overlap_width <= 0:
            # Non-overlapping: left/right with center gap
            left_end = int(0.45 * w)
            right_start = int(0.55 * w)
            mask_A[:, :left_end] = 1.0
            mask_B[:, right_start:] = 1.0
        else:
            # 3-zone overlapping
            half_ov = overlap_width / 2.0
            left_end = int((0.5 - half_ov) * w)
            right_start = int((0.5 + half_ov) * w)
            mask_A[:, :left_end] = 1.0
            mask_B[:, right_start:] = 1.0
            mask_shared = torch.zeros(h, w, device=self.device)
            mask_shared[:, left_end:right_start] = 1.0
            mask_shared = mask_shared.reshape(-1)

        return mask_A.reshape(-1), mask_B.reshape(-1), mask_shared

    def _get_token_indices(self, prompt, words_A, words_B):
        tokens = self.pipe.tokenizer(prompt)
        token_strs = self.pipe.tokenizer.convert_ids_to_tokens(tokens['input_ids'])
        print(f"  Tokens: {list(enumerate(token_strs))}")

        indices_A, indices_B = [], []
        for i, tok in enumerate(token_strs):
            clean = tok.replace('</w>', '').lower()
            for w in words_A:
                if w.lower() in clean:
                    indices_A.append(i)
            for w in words_B:
                if w.lower() in clean:
                    indices_B.append(i)

        print(f"  Entity A indices: {indices_A}")
        print(f"  Entity B indices: {indices_B}")
        return indices_A, indices_B

    def _setup_masks_and_tokens(self, prompt, words_A, words_B, overlap_width):
        indices_A, indices_B = self._get_token_indices(prompt, words_A, words_B)

        self._masks_cache = {}
        for hw in [64, 256, 1024, 4096]:
            mA, mB, mSh = self._create_masks(hw, overlap_width)
            self._masks_cache[hw] = (mA, mB, mSh)

        for proc in self.processors.values():
            proc.set_token_indices(indices_A, indices_B)
            for hw, (mA, mB, mSh) in self._masks_cache.items():
                proc.set_region_masks(mA, mB, mSh)

    def _update_masks_after_forward(self, overlap_width):
        for proc in self.processors.values():
            if proc.text_attn_A is not None:
                hw = proc.text_attn_A.shape[-1]
                if hw not in self._masks_cache:
                    mA, mB, mSh = self._create_masks(hw, overlap_width)
                    self._masks_cache[hw] = (mA, mB, mSh)
                proc.set_region_masks(*self._masks_cache[hw])

    def clear_maps(self):
        for proc in self.processors.values():
            proc.text_attn_A = None
            proc.text_attn_B = None

    def get_captured_maps(self, target_min_hw=256):
        maps = {}
        for name, proc in self.processors.items():
            if proc.text_attn_A is not None and proc.text_attn_B is not None:
                hw = proc.text_attn_A.shape[-1]
                if hw >= target_min_hw:
                    maps[name] = {
                        'energy_A': proc.text_attn_A,
                        'energy_B': proc.text_attn_B,
                    }
        return maps

    def compute_region_loss(self, captured_maps):
        """Region loss supporting both 2-zone and 3-zone modes."""
        total_loss = 0.0
        n = 0
        for name, data in captured_maps.items():
            eA = data['energy_A']
            eB = data['energy_B']
            if eA.shape[0] == 2:
                eA = eA[1:2]
                eB = eB[1:2]
            hw = eA.shape[-1]
            if hw not in self._masks_cache:
                continue
            mask_A, mask_B, mask_shared = self._masks_cache[hw]

            eA_norm = eA / (eA.max() + 1e-8)
            eB_norm = eB / (eB.max() + 1e-8)

            if mask_shared is not None:
                # 3-zone mode
                excl_A = mask_A
                excl_B = mask_B
                shared = mask_shared

                leak_A = (eA_norm * excl_B.unsqueeze(0)).mean()
                leak_B = (eB_norm * excl_A.unsqueeze(0)).mean()

                # Co-activation penalty in shared zone
                eA_sh = eA_norm * shared.unsqueeze(0)
                eB_sh = eB_norm * shared.unsqueeze(0)
                co_activation = (eA_sh * eB_sh).sum() / (shared.sum() + 1e-8)

                # Activation reward
                full_A = excl_A + shared
                full_B = excl_B + shared
                activate_A = torch.relu(0.3 - (eA_norm * full_A.unsqueeze(0)).mean())
                activate_B = torch.relu(0.3 - (eB_norm * full_B.unsqueeze(0)).mean())

                total_loss += (leak_A + leak_B +
                               0.3 * co_activation +
                               2.0 * (activate_A + activate_B))
            else:
                # 2-zone mode
                leak_A = (eA_norm * mask_B.unsqueeze(0)).mean()
                leak_B = (eB_norm * mask_A.unsqueeze(0)).mean()
                overlap = (eA_norm * eB_norm).mean()
                activate_A = torch.relu(0.3 - (eA_norm * mask_A.unsqueeze(0)).mean())
                activate_B = torch.relu(0.3 - (eB_norm * mask_B.unsqueeze(0)).mean())

                total_loss += (leak_A + leak_B + 0.5 * overlap + activate_A + activate_B)

            n += 1

        return total_loss / max(n, 1)

    def generate(
        self,
        prompt,
        negative_prompt,
        words_A,
        words_B,
        lambda_max=1000.0,
        seed=42,
        overlap_width=0.0,
    ):
        num_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']
        tau_threshold = self.config['dclg']['tau_threshold']
        target_min_hw = self.config['dclg']['target_min_hw']
        grad_clip = self.config['dclg'].get('grad_clip', 5.0)

        self._setup_masks_and_tokens(prompt, words_A, words_B, overlap_width)

        text_cond, text_uncond = self.pipe.encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )

        device = torch.device(self.device)
        generator = torch.Generator(device).manual_seed(seed)
        img_size = self.config['generation']['image_size']
        latents = self.pipe.prepare_latents(
            1, 4, img_size, img_size, torch.float32, device, generator
        )

        self.pipe.scheduler.set_timesteps(num_steps, device=device)

        losses = []
        masks_init = False

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            apply_guidance = (i < tau_threshold and lambda_max > 0)

            latents = latents.detach().to(torch.float32)
            if apply_guidance:
                latents = latents.requires_grad_(True)

            self.clear_maps()

            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            enc_hs = torch.cat([text_uncond, text_cond], dim=0).to(torch.float32)

            noise_pred_out = self.pipe.unet(
                latent_input.to(self.pipe.unet.dtype), t,
                encoder_hidden_states=enc_hs.to(self.pipe.unet.dtype),
            ).sample.to(torch.float32)

            if not masks_init:
                self._update_masks_after_forward(overlap_width)
                masks_init = True

            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if apply_guidance:
                captured = self.get_captured_maps(target_min_hw)
                loss = self.compute_region_loss(captured)
                losses.append(loss.item())

                grad = torch.autograd.grad(loss, latents)[0]
                if grad_clip:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                decay = 1.0 - i / tau_threshold
                noise_pred = noise_pred + lambda_max * decay * grad
                latents = latents.detach()
            else:
                losses.append(0.0)
                latents = latents.detach()

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            image = self.pipe.decode_latents(latents.to(self.pipe.vae.dtype))
            image = self.pipe.numpy_to_pil(image)[0]

        final_maps = self._get_final_maps(latents, text_cond, text_uncond,
                                          self.pipe.scheduler.timesteps[-1], target_min_hw)
        return image, losses, final_maps

    def _get_final_maps(self, latents, text_cond, text_uncond, t, target_min_hw):
        with torch.no_grad():
            self.clear_maps()
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
            enc_hs = torch.cat([text_uncond, text_cond], dim=0).to(torch.float32)
            self.pipe.unet(
                latent_input.to(self.pipe.unet.dtype), t,
                encoder_hidden_states=enc_hs.to(self.pipe.unet.dtype),
            )
        captured = self.get_captured_maps(target_min_hw)
        if not captured:
            return {}
        name = list(captured.keys())[-1]
        eA = captured[name]['energy_A']
        eB = captured[name]['energy_B']
        if eA.shape[0] == 2:
            eA = eA[1:2]
            eB = eB[1:2]
        hw = eA.shape[-1]
        masks = self._masks_cache.get(hw)
        result = {
            'map_A': eA[0].detach().cpu().float().numpy(),
            'map_B': eB[0].detach().cpu().float().numpy(),
        }
        if masks:
            mA, mB, mSh = masks
            result['mask_A'] = mA.cpu().numpy()
            result['mask_B'] = mB.cpu().numpy()
            if mSh is not None:
                result['mask_shared'] = mSh.cpu().numpy()
        return result

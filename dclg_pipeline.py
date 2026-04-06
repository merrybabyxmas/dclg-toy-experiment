import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os

class DCLGPipeline:
    def __init__(self, pipe: StableDiffusionPipeline, hook_manager, config):
        self.pipe = pipe
        self.hook_manager = hook_manager
        self.config = config
        self.device = pipe.device

    def compute_chimera_loss(self, captured_maps: dict, idx_A: int, idx_B: int):
        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            num_heads = attn_map.shape[0] // 2
            cond_attn = attn_map[num_heads:] 
            
            map_A = cond_attn[:, :, idx_A].mean(dim=0)
            map_B = cond_attn[:, :, idx_B].mean(dim=0)
            
            overlap_loss = (map_A * map_B).mean()
            erasure_penalty = torch.relu(0.1 - map_A.max()) + torch.relu(0.1 - map_B.max())
            
            total_loss += (overlap_loss + erasure_penalty)
            
        return total_loss / num_layers

    def generate(
        self,
        prompt,
        negative_prompt,
        idx_A,
        idx_B,
        lambda_max=10.0,
        seed=42,
        save_intermediate=False,
        lambda_val_label="0.0"
    ):
        num_inference_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']
        tau_threshold = self.config['dclg']['tau_threshold']
        
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )
        
        generator = torch.Generator(self.device).manual_seed(seed)
        latents = self.pipe.prepare_latents(
            1, 4, self.config['generation']['image_size'], self.config['generation']['image_size'],
            prompt_embeds.dtype, self.device, generator
        )
        
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        losses = []
        # 시각화를 위한 중간 스텝 정의
        save_steps = [25, 15, 5]
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # 1. Latent에 gradient 설정
            latents = latents.detach().requires_grad_(True)
            self.hook_manager.clear()
            
            # 2. Forward pass
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            ).sample
            
            # 3. CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 4. Guidance (방법 A: Latent 직접 업데이트)
            if i < tau_threshold and lambda_max > 0:
                captured_maps = self.hook_manager.get_captured_maps()
                loss = self.compute_chimera_loss(captured_maps, idx_A, idx_B)
                losses.append(loss.item())
                
                # 중간 단계 시각화 저장
                if save_intermediate and i in save_steps:
                    self.save_debug_maps(captured_maps, idx_A, idx_B, i, lambda_val_label)

                # Gradient 계산 및 Latent 업데이트
                grad = torch.autograd.grad(loss, latents)[0]
                
                grad_clip = self.config['dclg'].get('grad_clip')
                if grad_clip:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                # Update latents: grad 방향의 반대로 (Loss 최소화)
                latents = latents - lambda_max * (1 - i / tau_threshold) * grad
                latents = latents.detach()
            else:
                losses.append(0.0)
                latents = latents.detach()
            
            # 5. Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        with torch.no_grad():
            image = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(image)[0]
        
        return image, losses

    def save_debug_maps(self, captured_maps, idx_A, idx_B, step, label):
        import matplotlib.pyplot as plt
        if not captured_maps: return
        
        name = list(captured_maps.keys())[-1]
        attn_map = captured_maps[name]
        num_heads = attn_map.shape[0] // 2
        cond_attn = attn_map[num_heads:].mean(dim=0)
        
        map_A = cond_attn[:, idx_A].detach().cpu().float().numpy()
        map_B = cond_attn[:, idx_B].detach().cpu().float().numpy()
        
        hw = map_A.shape[0]
        h = w = int(np.sqrt(hw))
        map_A = map_A.reshape(h, w)
        map_B = map_B.reshape(h, w)
        
        os.makedirs("dclg_toy/outputs/attention_maps", exist_ok=True)
        
        # Knight (A) - Red scale
        plt.imshow(map_A, cmap='Reds', alpha=0.8)
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_A.png")
        plt.close()
        
        # Orc (B) - Blues scale
        plt.imshow(map_B, cmap='Blues', alpha=0.8)
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_B.png")
        plt.close()
        
        # Overlap
        overlap = (map_A * map_B)
        plt.imshow(overlap, cmap='YlOrRd')
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_overlap.png")
        plt.close()

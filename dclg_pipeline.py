import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

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
            
            map_A = cond_attn[:, :, idx_A].mean(dim=0) # [HW]
            map_B = cond_attn[:, :, idx_B].mean(dim=0) # [HW]
            
            # 1. Overlap Loss (Mean으로 정규화)
            overlap_loss = (map_A * map_B).mean()
            
            # 2. Erasure Penalty (객체 사라짐 방지)
            # 각 맵의 최대값이 0.1 이하로 떨어지면 페널티 부여
            erasure_penalty = torch.relu(0.1 - map_A.max()) + torch.relu(0.1 - map_B.max())
            
            total_loss += (overlap_loss + erasure_penalty)
            
        return total_loss / num_layers

    def generate(
        self,
        prompt,
        negative_prompt,
        idx_A,
        idx_B,
        lambda_max=10.0, # mean 스케일에 맞춰 기본값 상향
        seed=42,
        **kwargs
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
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Gradient가 흘러야 하므로 requires_grad=True
            latents = latents.detach().requires_grad_(True)
            self.hook_manager.clear()
            
            # Forward
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            ).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DCLG Guidance (Score Modification)
            if i < tau_threshold:
                lambda_t = lambda_max * (1 - i / tau_threshold)
                
                # Loss & Grad 계산
                captured_maps = self.hook_manager.get_captured_maps()
                loss = self.compute_chimera_loss(captured_maps, idx_A, idx_B)
                losses.append(loss.item())
                
                grad = torch.autograd.grad(loss, latents)[0]
                
                # Clip grad if needed
                grad_clip = self.config['dclg'].get('grad_clip')
                if grad_clip:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                # Score Modification (Method A): noise_pred 수정
                # Grad의 방향은 Loss가 증가하는 방향이므로, 
                # 노이즈 예측값에서 grad를 빼주는 것은 (또는 더해주는 것은 스케줄러 수식에 따라 다름)
                # 여기서는 'Attend-and-Excite' 스타일의 guidance 방향을 따름
                # noise_pred = noise_pred + lambda_t * grad (Denoising 방향 조정)
                noise_pred = noise_pred + lambda_t * grad
            else:
                losses.append(0.0)
            
            # Step (수정된 noise_pred 사용)
            latents = self.pipe.scheduler.step(noise_pred, t, latents.detach()).prev_sample
            
        with torch.no_grad():
            image = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(image)[0]
        
        return image, losses

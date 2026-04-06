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
        """
        captured_maps: {layer_name: [B*heads, HW, 77]}
        idx_A, idx_B: 텍스트 토큰 인덱스
        """
        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            # attn_map: [2 * heads, HW, 77] (CFG 사용 시)
            num_heads = attn_map.shape[0] // 2
            cond_attn = attn_map[num_heads:] # [num_heads, HW, 77]
            
            # 특정 토큰의 맵 추출
            map_A = cond_attn[:, :, idx_A].mean(dim=0) # [HW]
            map_B = cond_attn[:, :, idx_B].mean(dim=0) # [HW]
            
            # Hadamard product 및 공간 합산
            overlap = map_A * map_B
            total_loss += overlap.sum()
            
        return total_loss / num_layers

    def apply_dclg_guidance(self, latents, loss, lambda_t):
        if lambda_t <= 0:
            return latents
        
        grad = torch.autograd.grad(loss, latents, retain_graph=False)[0]
        
        grad_clip = self.config['dclg'].get('grad_clip')
        if grad_clip is not None:
            grad = torch.clamp(grad, -grad_clip, grad_clip)
            
        updated_latents = latents - lambda_t * grad
        return updated_latents.detach()

    def generate(
        self,
        prompt,
        negative_prompt,
        idx_A,
        idx_B,
        lambda_max=0.5,
        seed=42,
        **kwargs
    ):
        num_inference_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']
        tau_threshold = self.config['dclg']['tau_threshold']
        
        # Text embeddings
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        
        # Initialize latents
        generator = torch.Generator(self.device).manual_seed(seed)
        latents = self.pipe.prepare_latents(
            1, 4, self.config['generation']['image_size'], self.config['generation']['image_size'],
            prompt_embeds.dtype, self.device, generator
        )
        
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        losses = []
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            latents = latents.detach().requires_grad_(True)
            self.hook_manager.clear()
            
            # Predict noise
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            ).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DCLG Guidance
            if i < tau_threshold:
                lambda_t = lambda_max * (1 - i / tau_threshold)
            else:
                lambda_t = 0
            
            if lambda_t > 0:
                captured_maps = self.hook_manager.get_captured_maps()
                loss = self.compute_chimera_loss(captured_maps, idx_A, idx_B)
                losses.append(loss.item())
                
                # Update latents
                latents = self.apply_dclg_guidance(latents, loss, lambda_t)
            else:
                losses.append(0.0)
                latents = latents.detach()
            
            # Step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode
        with torch.no_grad():
            image = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(image)[0]
        
        return image, losses

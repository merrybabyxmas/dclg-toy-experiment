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

    def compute_chimera_loss(self, captured_maps: dict):
        """
        IP-Adapter용 Chimera Loss: 이미지 토큰 간의 겹침 억제
        captured_maps: {layer_name: [B*heads, HW, 8]} (4 tokens for A, 4 tokens for B)
        """
        total_loss = 0.0
        num_layers = len(captured_maps)
        if num_layers == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        for name, attn_map in captured_maps.items():
            # CFG assume batch=2, num_heads = shape[0]//2
            num_heads = attn_map.shape[0] // 2
            cond_attn = attn_map[num_heads:] # [heads, HW, 8]
            
            # 1. 기사(A)와 오크(B) 이미지 토큰 분리 (각 4토큰)
            map_A = cond_attn[:, :, 0:4].mean(dim=-1).mean(dim=0) # [HW]
            map_B = cond_attn[:, :, 4:8].mean(dim=-1).mean(dim=0) # [HW]
            
            # 2. Sharpening & Normalization
            map_A = torch.pow(map_A, 2)
            map_B = torch.pow(map_B, 2)
            map_A = map_A / (map_A.max() + 1e-8)
            map_B = map_B / (map_B.max() + 1e-8)
            
            # 3. Cosine Similarity (배타성)
            cos_sim = F.cosine_similarity(map_A.view(-1), map_B.view(-1), dim=0)
            
            # 4. Erasure Penalty
            erasure_penalty = torch.relu(0.5 - map_A.max()) + torch.relu(0.5 - map_B.max())
            
            total_loss += (cos_sim + erasure_penalty)
            
        return total_loss / num_layers

    @torch.no_grad()
    def prepare_ip_adapter_embeddings(self, ip_adapter, image_A, image_B, prompt, negative_prompt):
        # Image A
        clip_A = ip_adapter.clip_image_processor(images=image_A, return_tensors="pt").pixel_values
        embeds_A = ip_adapter.image_encoder(clip_A.to(self.device, dtype=torch.float32)).image_embeds
        tokens_A = ip_adapter.image_proj_model(embeds_A) # [1, 4, 1024]
        
        # Image B
        clip_B = ip_adapter.clip_image_processor(images=image_B, return_tensors="pt").pixel_values
        embeds_B = ip_adapter.image_encoder(clip_B.to(self.device, dtype=torch.float32)).image_embeds
        tokens_B = ip_adapter.image_proj_model(embeds_B) # [1, 4, 1024]
        
        # Concatenate: 8 image tokens
        image_prompt_embeds = torch.cat([tokens_A, tokens_B], dim=1) # [1, 8, 1024]
        
        # Uncond image embeds
        uncond_image_prompt_embeds = ip_adapter.image_proj_model(torch.zeros_like(torch.cat([embeds_A, embeds_B], dim=0)))
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(1, 8, -1)
        
        # Text embeds
        prompt_embeds_, neg_embeds_ = self.pipe.encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )
        
        # Merge Text + Image
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([neg_embeds_, uncond_image_prompt_embeds], dim=1)
        
        return prompt_embeds, negative_prompt_embeds

    def generate(
        self,
        ip_adapter,
        image_A,
        image_B,
        prompt,
        negative_prompt,
        lambda_max=10.0,
        seed=42,
        save_intermediate=False,
        lambda_val_label="0.0"
    ):
        num_inference_steps = self.config['generation']['num_inference_steps']
        guidance_scale = self.config['generation']['guidance_scale']
        tau_threshold = self.config['dclg']['tau_threshold']
        
        # Embeddings
        prompt_embeds, negative_prompt_embeds = self.prepare_ip_adapter_embeddings(
            ip_adapter, image_A, image_B, prompt, negative_prompt
        )
        
        # Set IP processors num_tokens=8
        for name, module in self.pipe.unet.named_modules():
            if hasattr(module, "processor") and hasattr(module.processor, "num_tokens"):
                module.processor.num_tokens = 8
        
        generator = torch.Generator(self.device).manual_seed(seed)
        latents = self.pipe.prepare_latents(
            1, 4, self.config['generation']['image_size'], self.config['generation']['image_size'],
            prompt_embeds.dtype, self.device, generator
        )
        
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        losses = []
        save_steps = [25, 15, 5]
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            latents = latents.detach().requires_grad_(True)
            self.hook_manager.clear()
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred_out = self.pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if i < tau_threshold and lambda_max > 0:
                loss = self.compute_chimera_loss(self.hook_manager.captured_maps)
                losses.append(loss.item())
                
                if save_intermediate and i in save_steps:
                    self.save_debug_maps(self.hook_manager.captured_maps, i, lambda_val_label)

                grad = torch.autograd.grad(loss, latents)[0]
                
                grad_clip = self.config['dclg'].get('grad_clip')
                if grad_clip:
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                # Noise pred modification
                noise_pred = noise_pred + lambda_max * (1 - i / tau_threshold) * grad
                latents = latents.detach()
            else:
                losses.append(0.0)
                latents = latents.detach()
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        with torch.no_grad():
            image = self.pipe.decode_latents(latents)
            image = self.pipe.numpy_to_pil(image)[0]
        
        return image, losses

    def save_debug_maps(self, captured_maps, step, label):
        import matplotlib.pyplot as plt
        if not captured_maps: return
        
        name = list(captured_maps.keys())[-1]
        attn_map = captured_maps[name]
        num_heads = attn_map.shape[0] // 2
        cond_attn = attn_map[num_heads:].mean(dim=0)
        
        map_A = cond_attn[:, 0:4].mean(dim=-1).detach().cpu().float().numpy()
        map_B = cond_attn[:, 4:8].mean(dim=-1).detach().cpu().float().numpy()
        
        map_A = np.power(map_A, 2)
        map_B = np.power(map_B, 2)
        map_A = map_A / (map_A.max() + 1e-8)
        map_B = map_B / (map_B.max() + 1e-8)
        
        hw = map_A.shape[0]
        h = w = int(np.sqrt(hw))
        map_A = map_A.reshape(h, w)
        map_B = map_B.reshape(h, w)
        
        os.makedirs("dclg_toy/outputs/attention_maps", exist_ok=True)
        plt.imshow(map_A, cmap='Reds', alpha=0.8)
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_A.png")
        plt.close()
        plt.imshow(map_B, cmap='Blues', alpha=0.8)
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_B.png")
        plt.close()
        overlap = (map_A * map_B)
        plt.imshow(overlap, cmap='YlOrRd')
        plt.colorbar()
        plt.savefig(f"dclg_toy/outputs/attention_maps/step{step}_lambda{label}_overlap.png")
        plt.close()

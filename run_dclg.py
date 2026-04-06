import os
import torch
import yaml
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import sys

# Add dclg_toy to path
sys.path.append(os.path.join(os.getcwd(), 'dclg_toy'))
from dclg_toy.hook_manager import HookManager
from dclg_toy.dclg_pipeline import DCLGPipeline
from dclg_toy.utils import visualize_attn_map, visualize_overlap, save_grid, log_loss_curve

def get_token_index(tokenizer, prompt, target_word):
    inputs = tokenizer(prompt)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
    print(f"Tokens: {tokens}")
    for i, token in enumerate(tokens):
        # Handle BPE style (e.g., 'knight</w>')
        clean_token = token.replace('</w>', '')
        if clean_token.lower() == target_word.lower():
            return i
    return -1

def main():
    with open("dclg_toy/configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = config['model']['base']
    
    # Load SD pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Target Tokens
    prompt = config['prompt']
    idx_knight = get_token_index(pipe.tokenizer, prompt, "knight")
    idx_orc = get_token_index(pipe.tokenizer, prompt, "orc")
    
    print(f"Target Token Indices: knight={idx_knight}, orc={idx_orc}")
    if idx_knight == -1 or idx_orc == -1:
        print("Error: Target words not found in prompt.")
        return

    hook_manager = HookManager(target_min_hw=config['dclg']['target_min_hw'])
    hook_manager.register_hooks(pipe.unet)
    
    dclg_pipe = DCLGPipeline(pipe, hook_manager, config)
    
    lambda_sweep = [0.0, 2.0, 5.0, 10.0] # 좀 더 극단적인 변화 관찰을 위해 확장
    results = []
    labels = []
    
    os.makedirs("dclg_toy/outputs/images", exist_ok=True)
    os.makedirs("dclg_toy/outputs/attention_maps", exist_ok=True)

    for lambda_val in lambda_sweep:
        print(f"\n>>> Running experiment with lambda = {lambda_val}")
        image, losses = dclg_pipe.generate(
            prompt=prompt,
            negative_prompt=config['negative_prompt'],
            idx_A=idx_knight,
            idx_B=idx_orc,
            lambda_max=lambda_val,
            seed=config['generation']['seed']
        )
        
        save_path = f"dclg_toy/outputs/images/text_lambda_{lambda_val}.png"
        image.save(save_path)
        results.append(image)
        labels.append(f"lambda={lambda_val}")
        
        # Log loss curve
        log_loss_curve(losses, f"dclg_toy/outputs/images/text_loss_lambda_{lambda_val}.png")
        
        # Save last step attention maps
        captured_maps = hook_manager.get_captured_maps()
        if captured_maps:
            # Pick a representative layer
            name = list(captured_maps.keys())[-1]
            attn_map = captured_maps[name]
            num_heads = attn_map.shape[0] // 2
            cond_attn = attn_map[num_heads:].mean(dim=0)
            
            map_A = cond_attn[:, idx_knight]
            map_B = cond_attn[:, idx_orc]
            
            visualize_attn_map(map_A, f"dclg_toy/outputs/attention_maps/text_lambda_{lambda_val}_knight.png")
            visualize_attn_map(map_B, f"dclg_toy/outputs/attention_maps/text_lambda_{lambda_val}_orc.png")
            visualize_overlap(map_A, map_B, f"dclg_toy/outputs/attention_maps/text_lambda_{lambda_val}_overlap.png")

    save_grid(results, labels, "dclg_toy/outputs/images/text_grid_results.png")
    print("\nExperiments finished. Results saved in dclg_toy/outputs/")

if __name__ == "__main__":
    main()

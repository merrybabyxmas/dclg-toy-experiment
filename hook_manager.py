import torch

class HookManager:
    def __init__(self, target_min_hw=256):
        self.target_min_hw = target_min_hw
        self.captured_maps = {}

    def make_hook(self, name):
        def hook(module, input, output):
            # IPAttnProcessor는 내부에 attn_map을 저장하도록 설계됨
            if hasattr(module.processor, "attn_map") and module.processor.attn_map is not None:
                attn_map = module.processor.attn_map # [B*heads, HW, num_tokens]
                if attn_map.shape[1] >= self.target_min_hw:
                    self.captured_maps[name] = attn_map
        return hook

    def register_hooks(self, unet):
        self.hooks = []
        for name, module in unet.named_modules():
            # Up-blocks의 Cross-Attention (attn2) 레이어 타겟팅
            if "up_blocks" in name and "attn2" in name and not name.endswith("attn2"):
                pass
            if "up_blocks" in name and "attn2" in name and hasattr(module, "processor"):
                h = module.register_forward_hook(self.make_hook(name))
                self.hooks.append(h)

    def clear(self):
        self.captured_maps = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

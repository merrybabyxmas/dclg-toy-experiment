import torch

class SaveAttnProcessor:
    def __init__(self):
        self.attn_map = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
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
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 캡처 지점: Text Cross-Attention (77 tokens)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attn_map = attention_probs  # [B*heads, HW, 77]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class HookManager:
    def __init__(self, target_min_hw=256):
        self.target_min_hw = target_min_hw
        self.processors = {}

    def register_hooks(self, unet):
        for name, module in unet.named_modules():
            # Up-blocks의 Cross-Attention (attn2) 레이어만 타겟팅
            if "up_blocks" in name and "attn2" in name and hasattr(module, "processor"):
                if not name.endswith(".attn2"): continue 
                
                proc = SaveAttnProcessor()
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

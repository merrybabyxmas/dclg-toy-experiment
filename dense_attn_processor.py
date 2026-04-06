"""
Dense Diffusion Attention Processor (text-only, no IP-Adapter).

Per-token spatial masking on text cross-attention:
- Entity A tokens → assigned region only
- Entity B tokens → assigned region only
- Shared tokens (style, quality) → attend everywhere

Supports both non-overlapping and 3-zone overlapping masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttnProcessor(nn.Module):
    def __init__(self):
        super().__init__()

        # Region masks [HW] float
        self.mask_A = None   # entity A region (or exclusive A)
        self.mask_B = None   # entity B region (or exclusive B)
        self.mask_shared = None  # overlap zone (optional, None for non-overlapping)

        # Text token indices for each entity
        self.token_indices_A = None
        self.token_indices_B = None

        # Captured text attention maps for loss
        self.text_attn_A = None  # [B, HW] energy for entity A
        self.text_attn_B = None  # [B, HW] energy for entity B

        self.suppress_strength = 20.0

    def set_region_masks(self, mask_A, mask_B, mask_shared=None):
        self.mask_A = mask_A
        self.mask_B = mask_B
        self.mask_shared = mask_shared

    def set_token_indices(self, indices_A, indices_B):
        self.token_indices_A = indices_A
        self.token_indices_B = indices_B

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hw = query.shape[2]

        # ========== REGION-MASKED TEXT ATTENTION ==========
        scale_factor = head_dim ** -0.5
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

        if (self.mask_A is not None and
            self.token_indices_A is not None and
            self.mask_A.shape[0] == hw):

            dev = attn_logits.device
            dtype = attn_logits.dtype
            mA = self.mask_A.to(dev, dtype=dtype)
            mB = self.mask_B.to(dev, dtype=dtype)

            bias = torch.zeros_like(attn_logits)

            if self.mask_shared is not None:
                # 3-zone: suppress only in exclusive-OTHER zone
                mSh = self.mask_shared.to(dev, dtype=dtype)
                suppress_A_in = mB * (1.0 - mSh)  # suppress A only in exclusive-B
                suppress_B_in = mA * (1.0 - mSh)  # suppress B only in exclusive-A
            else:
                # 2-zone: suppress in other's region
                suppress_A_in = 1.0 - mA
                suppress_B_in = 1.0 - mB

            for idx in self.token_indices_A:
                if idx < attn_logits.shape[-1]:
                    bias[:, :, :, idx] = -self.suppress_strength * suppress_A_in.unsqueeze(0).unsqueeze(0)

            for idx in self.token_indices_B:
                if idx < attn_logits.shape[-1]:
                    bias[:, :, :, idx] = -self.suppress_strength * suppress_B_in.unsqueeze(0).unsqueeze(0)

            attn_logits = attn_logits + bias

        attn_probs = attn_logits.softmax(dim=-1)

        # Capture per-entity text attention energy (NO torch.no_grad — gradient must flow for DCLG)
        if self.token_indices_A is not None and self.mask_A is not None and self.mask_A.shape[0] == hw:
            energy_A = torch.zeros(batch_size, hw, device=dev, dtype=dtype)
            energy_B = torch.zeros(batch_size, hw, device=dev, dtype=dtype)
            for idx in self.token_indices_A:
                if idx < attn_probs.shape[-1]:
                    energy_A = energy_A + attn_probs[:, :, :, idx].mean(dim=1)
            for idx in self.token_indices_B:
                if idx < attn_probs.shape[-1]:
                    energy_B = energy_B + attn_probs[:, :, :, idx].mean(dim=1)
            self.text_attn_A = energy_A
            self.text_attn_B = energy_B

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

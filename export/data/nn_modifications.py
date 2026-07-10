# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast


def _apply_rotary_real(x, cos, sin):
    """ONNX-friendly replacement for Qwen3-VL's complex rotary helper."""
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + rotated * sin).to(orig_dtype)


def _make_onnx_vision_attention_forward(attn_module):
    """Patch Qwen3-VL vision attention to avoid complex ops and trace-unfriendly splits."""

    def forward(
        hidden_states,
        cu_seqlens=None,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[0]
        qkv = attn_module.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, attn_module.num_heads, -1)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)

        cos, sin = position_embeddings
        q = _apply_rotary_real(q, cos, sin)
        k = _apply_rotary_real(k, cos, sin)

        # Preserve Qwen3-VL's per-image attention chunks while avoiding Python lists derived from
        # tensor lengths in the original implementation.
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            q_chunks = torch.split(q, lengths.tolist(), dim=0)
            k_chunks = torch.split(k, lengths.tolist(), dim=0)
            v_chunks = torch.split(v, lengths.tolist(), dim=0)
        else:
            q_chunks = (q,)
            k_chunks = (k,)
            v_chunks = (v,)

        attn_outputs = []
        for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
            q_c = q_c.transpose(0, 1)
            k_c = k_c.transpose(0, 1)
            v_c = v_c.transpose(0, 1)
            attn_weights = torch.matmul(q_c, k_c.transpose(-2, -1)) * attn_module.scaling
            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(v_c.dtype)
            attn_outputs.append(torch.matmul(attn_weights, v_c).transpose(0, 1))

        attn_output = torch.cat(attn_outputs, dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn_module.proj(attn_output)

    return forward


def get_modified_vision_model(vision_model):
    """Monkey-patch the live N1.7 Qwen3-VL vision model in place."""
    for block in vision_model.blocks:
        block.attn.forward = _make_onnx_vision_attention_forward(block.attn)
        block.attn.config._attn_implementation = "eager"
    return vision_model.eval()


def freeze_vision_pos_embeds(vision_model, grid_thw):
    """Pre-compute ``fast_pos_embed_interpolate`` and ``rot_pos_emb`` for a fixed
    ``grid_thw`` and replace those methods with constant-returning closures.

    Why: ``Qwen3VLVisionEncoder.fast_pos_embed_interpolate`` builds Gather indices
    into the 2304-entry pos_embed table via Python loops over tensor values and
    ``.tolist()`` calls. Tracing this is brittle — the resulting ONNX may leave
    the Gather op wired to a runtime int64 input, which causes Triton instance
    creation to read uninitialized device memory and fail with
    ``indices element out of data bounds``. ``rot_pos_emb`` has the same kind of
    grid-thw-dependent shape build-up. Freezing both removes that risk for a
    shape-fixed export.

    Mirrors the precomputation done in
    ``scripts/deployment/export_onnx_n1d7.py::Qwen3VisionForExport.__init__``.
    """
    with torch.no_grad():
        frozen_pos = vision_model.fast_pos_embed_interpolate(grid_thw).detach().clone()
        frozen_rot = vision_model.rot_pos_emb(grid_thw).detach().clone()

    def _frozen_pos_embed_interpolate(self, grid_thw):
        return frozen_pos.to(device=grid_thw.device, dtype=frozen_pos.dtype)

    def _frozen_rot_pos_emb(self, grid_thw):
        return frozen_rot.to(device=grid_thw.device, dtype=frozen_rot.dtype)

    vision_model.fast_pos_embed_interpolate = types.MethodType(
        _frozen_pos_embed_interpolate, vision_model
    )
    vision_model.rot_pos_emb = types.MethodType(_frozen_rot_pos_emb, vision_model)
    return vision_model


def _deepstack_process_onnx(self, hidden_states, visual_pos_masks, visual_embeds):
    """ONNX-friendly replacement for boolean indexed in-place assignment."""
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    delta = torch.zeros_like(hidden_states)
    delta = delta.masked_scatter(visual_pos_masks.unsqueeze(-1).expand_as(delta), visual_embeds)
    return hidden_states + delta


def _simple_causal_mask(dtype, device, batch_size, seq_len, attention_mask):
    mask_value = torch.finfo(dtype).min * 0.5
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype),
        diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None and attention_mask.dim() == 2:
        padding_mask = attention_mask[:, None, None, :].to(dtype)
        causal_mask = causal_mask + (1.0 - padding_mask) * mask_value
    return causal_mask


def _qwen3_vl_text_forward_onnx(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    cache_position=None,
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
    output_hidden_states=False,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_len = inputs_embeds.shape[:2]
    device = inputs_embeds.device
    if cache_position is None:
        cache_position = torch.arange(seq_len, device=device)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    text_position_ids = position_ids[0] if position_ids.shape[0] != 4 else position_ids[1]
    rotary_position_ids = position_ids if position_ids.shape[0] != 4 else position_ids[1:]
    causal_mask = _simple_causal_mask(
        inputs_embeds.dtype, device, batch_size, seq_len, attention_mask
    )
    position_embeddings = self.rotary_emb(inputs_embeds, rotary_position_ids)

    hidden_states = inputs_embeds
    hidden_states_history = () if output_hidden_states else None
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            hidden_states_history += (hidden_states,)
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    if output_hidden_states:
        hidden_states_history += (hidden_states,)

    # Qwen3Backbone consumes hidden_states[-1] from the full model output. Keep this pre-norm
    # state to match scripts/deployment/export_onnx_n1d7.py.
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        hidden_states=hidden_states_history,
        past_key_values=None,
    )


def get_modified_language_model(language_model):
    """Monkey-patch the live N1.7 Qwen3-VL text model in place."""
    language_model.config._attn_implementation = "eager"
    for layer in language_model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    language_model._deepstack_process = types.MethodType(_deepstack_process_onnx, language_model)
    language_model.forward = types.MethodType(_qwen3_vl_text_forward_onnx, language_model)
    return language_model.eval()
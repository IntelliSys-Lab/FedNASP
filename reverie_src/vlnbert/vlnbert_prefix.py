"""Step-wise Conditional Prefix Personalization for VLN-BERT (CA variant).

Copies of attention / encoder / VLNBert classes from vlnbert_CA.py,
modified to accept optional prefix K/V tensors in the attention layers.

New modules
-----------
* PrefixLayer      – learnable KV prefix for one attention layer
* GateNet          – MLP  z_t  →  σ(·) ∈ [0,1]^M  (step-wise gate)
* PrefixBertImageSelfAttention  – visual self-attention + prefix
* PrefixBertBiAttention         – cross-attention + prefix on text KV
* PrefixBertImageAttention / PrefixBertImageLayer /
  PrefixBertConnectionLayer / PrefixBertEncoder / PrefixVLNBert
  – wrappers that route prefix tensors down the hierarchy
"""

import copy
import json
import logging
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- unmodified classes imported from the original ----------------
from vlnbert.vlnbert_CA import (
    BertConfig,
    BertLayerNorm,
    BertEmbeddings,
    # text self-attention (unchanged)
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    # visual branch helpers (unchanged)
    BertImageSelfOutput,
    BertImageIntermediate,
    BertImageOutput,
    # cross-attention helpers (unchanged)
    BertBiOutput,
    # poolers
    BertTextPooler,
    BertImagePooler,
    # embeddings / vision encoder
    BertImageEmbeddings,
    BertObjectEmbeddings,
    VisionEncoder,
    # base class for weight init + from_pretrained
    BertPreTrainedModel,
)

logger = logging.getLogger(__name__)


# ======================================================================
#  1.  PrefixLayer – learnable KV prefix for a single attention layer
# ======================================================================

class PrefixLayer(nn.Module):
    """Learnable key / value prefix tokens for one attention layer.

    Parameters are stored as ``[num_heads, prefix_len, head_size]`` and
    expanded to the batch dimension at forward time.
    """

    def __init__(self, num_heads: int, head_size: int, prefix_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.prefix_len = prefix_len

        self.prefix_k = nn.Parameter(
            torch.randn(num_heads, prefix_len, head_size) * 0.02)
        self.prefix_v = nn.Parameter(
            torch.randn(num_heads, prefix_len, head_size) * 0.02)

    def forward(self, batch_size: int, gate_scale):
        """Return batch-expanded, gate-scaled prefix K/V.

        Args
        ----
        batch_size : int
        gate_scale : ``[B]`` tensor  **or**  python scalar

        Returns
        -------
        (prefix_k, prefix_v)  each ``[B, H, P, D]``
        """
        pk = self.prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pv = self.prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if isinstance(gate_scale, torch.Tensor):
            gs = gate_scale.view(-1, 1, 1, 1)       # [B,1,1,1]
        else:
            gs = gate_scale

        return pk * gs, pv * gs


# ======================================================================
#  2.  FedPerfix-style additive adapters
# ======================================================================

class AdditiveQKVAdapter(nn.Module):
    """Bottleneck MLP producing additive deltas for Q/K/V projections."""

    def __init__(self, in_dim: int, out_dim: int, mid_dim: int = 256, kv_only: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.mid_dim = int(mid_dim)
        self.kv_only = bool(kv_only)

        self.down = nn.Linear(self.in_dim, self.mid_dim)
        self.act = nn.Tanh()
        mul = 2 if self.kv_only else 3
        self.up = nn.Linear(self.mid_dim, self.out_dim * mul)

    def forward(self, x):
        delta = self.up(self.act(self.down(x)))
        if self.kv_only:
            zero_q = torch.zeros(
                delta.size(0), delta.size(1), self.out_dim,
                dtype=delta.dtype, device=delta.device
            )
            delta = torch.cat([zero_q, delta], dim=-1)
        return delta


# ======================================================================
#  3.  GateNet – (legacy) step-wise conditional gate MLP
# ======================================================================

class GateNet(nn.Module):
    """Produces per-module gate values  g_t ∈ [0,1]^M  from a feature
    vector z_t."""

    def __init__(self, input_dim: int, hidden_dim: int, num_modules: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modules),
        )
        # initialise last layer near zero → sigmoid ≈ 0.5 (neutral)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t):
        """z_t : [B, input_dim]  →  g_t : [B, M] in [0,1]."""
        return torch.sigmoid(self.net(z_t))


# ======================================================================
#  3.  Modified attention layers  (copies with prefix injection)
# ======================================================================

class PrefixBertImageSelfAttention(nn.Module):
    """Visual self-attention with optional prefix K/V concatenation.

    Identical to ``BertImageSelfAttention`` except that ``forward``
    accepts optional ``prefix_k`` / ``prefix_v`` tensors which are
    concatenated to the projected key / value *before* the dot-product.

    The returned ``attention_scores`` have the prefix columns **stripped**
    so that the downstream code (which uses these scores as action logits)
    sees the same tensor shape as the original.
    """

    def __init__(self, config):
        super().__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "v_hidden_size (%d) not divisible by v_num_attention_heads (%d)"
                % (config.v_hidden_size, config.v_num_attention_heads))

        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)
        self.attn_prefix_mode = getattr(config, 'attn_prefix_mode', 'fedperfix_add')
        self.attn_prefix_scale = float(getattr(config, 'prefix_scale', 1.0))
        mid_dim = int(getattr(config, 'prefix_mid_dim', 256))
        self.attn_prefix_qkv_adapter = AdditiveQKVAdapter(
            in_dim=config.v_hidden_size, out_dim=self.all_head_size, mid_dim=mid_dim, kv_only=False
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask,
                prefix_k=None, prefix_v=None, gate_scale=None):
        """
        Args
        ----
        hidden_states  : [B, S, D]
        attention_mask  : [B, 1, 1, S]   (0 or -10000)
        prefix_k / prefix_v : optional  [B, H, P, d]

        Returns
        -------
        context_layer     : [B, S, D]
        attention_scores  : [B, H, S, S]  (prefix cols stripped)
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.attn_prefix_mode == 'fedperfix_add':
            delta_qkv = self.attn_prefix_qkv_adapter(hidden_states)
            delta_q, delta_k, delta_v = torch.chunk(delta_qkv, 3, dim=-1)
            if isinstance(gate_scale, torch.Tensor):
                gs = gate_scale.view(-1, 1, 1)
            elif gate_scale is None:
                gs = 0.0
            else:
                gs = float(gate_scale)
            alpha = self.attn_prefix_scale * gs
            mixed_query_layer = mixed_query_layer + alpha * delta_q
            mixed_key_layer = mixed_key_layer + alpha * delta_k
            mixed_value_layer = mixed_value_layer + alpha * delta_v

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        prefix_len = 0
        if self.attn_prefix_mode == 'prefix_kv_concat' and prefix_k is not None and prefix_v is not None:
            prefix_len = prefix_k.size(2)
            key_layer   = torch.cat([prefix_k, key_layer],   dim=2)
            value_layer = torch.cat([prefix_v, value_layer], dim=2)
            # extend mask: prefix always attended → 0.0
            prefix_mask = torch.zeros(
                hidden_states.size(0), 1, 1, prefix_len,
                device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.all_head_size,))

        # strip prefix columns from returned scores
        scores_out = attention_scores[:, :, :, prefix_len:]
        return context_layer, scores_out


class PrefixBertBiAttention(nn.Module):
    """Cross-attention (vision queries text) with optional prefix on text K/V.

    Identical to ``BertBiAttention`` except that ``forward`` accepts
    optional ``prefix_k / prefix_v`` tensors concatenated to the text
    key / value *before* the dot-product.
    """

    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "bi_hidden_size (%d) not divisible by bi_num_attention_heads (%d)"
                % (config.bi_hidden_size, config.bi_num_attention_heads))

        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1  = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1    = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1  = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2  = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2    = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2  = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn_prefix_mode = getattr(config, 'attn_prefix_mode', 'fedperfix_add')
        self.attn_prefix_scale = float(getattr(config, 'prefix_scale', 1.0))
        mid_dim = int(getattr(config, 'prefix_mid_dim', 256))
        self.attn_prefix_query_adapter = AdditiveQKVAdapter(
            in_dim=config.v_hidden_size, out_dim=self.all_head_size, mid_dim=mid_dim, kv_only=False
        )
        # Use full Q/K/V-capable adapter for language branch as requested.
        # (In this CA direction we apply K/V to the attended language stream; Q slice is unused.)
        self.attn_prefix_kv_adapter = AdditiveQKVAdapter(
            in_dim=config.hidden_size, out_dim=self.all_head_size, mid_dim=mid_dim, kv_only=False
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1,
                input_tensor2, attention_mask2,
                co_attention_mask=None, use_co_attention_mask=False,
                prefix_k=None, prefix_v=None, gate_scale=None):
        """
        Args
        ----
        input_tensor1  : visn  [B, V, v_dim]
        input_tensor2  : lang  [B, T, h_dim]
        prefix_k / prefix_v : [B, H, P, d]  concatenated to text K/V

        Returns
        -------
        context_layer2     : [B, V, bi_dim]
        attention_scores2  : [B, H, V, T]  (prefix stripped)
        """
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        if self.attn_prefix_mode == 'fedperfix_add':
            dq = self.attn_prefix_query_adapter(input_tensor1)[..., :self.all_head_size]
            d_qkv2 = self.attn_prefix_kv_adapter(input_tensor2)
            _, dk2, dv2 = torch.chunk(d_qkv2, 3, dim=-1)
            if isinstance(gate_scale, torch.Tensor):
                gs = gate_scale.view(-1, 1, 1)
            elif gate_scale is None:
                gs = 0.0
            else:
                gs = float(gate_scale)
            alpha = self.attn_prefix_scale * gs
            mixed_query_layer1 = mixed_query_layer1 + alpha * dq
            mixed_key_layer2 = mixed_key_layer2 + alpha * dk2
            mixed_value_layer2 = mixed_value_layer2 + alpha * dv2

        # vision query
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)

        # text key / value
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        prefix_len = 0
        if self.attn_prefix_mode == 'prefix_kv_concat' and prefix_k is not None and prefix_v is not None:
            prefix_len = prefix_k.size(2)
            key_layer2   = torch.cat([prefix_k, key_layer2],   dim=2)
            value_layer2 = torch.cat([prefix_v, value_layer2], dim=2)
            prefix_mask = torch.zeros(
                input_tensor1.size(0), 1, 1, prefix_len,
                device=attention_mask2.device, dtype=attention_mask2.dtype)
            attention_mask2 = torch.cat([prefix_mask, attention_mask2], dim=-1)

        attention_scores2 = torch.matmul(
            query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2

        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        context_layer2 = context_layer2.view(
            context_layer2.size()[:-2] + (self.all_head_size,))

        scores_out = attention_scores2[:, :, :, prefix_len:]
        return context_layer2, scores_out


# ---------- thin wrappers that route prefix through -----------

class PrefixBertImageAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = PrefixBertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask,
                prefix_k=None, prefix_v=None, gate_scale=None):
        self_output, attention_probs = self.self(
            input_tensor, attention_mask,
            prefix_k=prefix_k, prefix_v=prefix_v, gate_scale=gate_scale)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class PrefixBertImageLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PrefixBertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask,
                prefix_k=None, prefix_v=None, gate_scale=None):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask,
            prefix_k=prefix_k, prefix_v=prefix_v, gate_scale=gate_scale)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class PrefixBertConnectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.biattention = PrefixBertBiAttention(config)
        self.biOutput = BertBiOutput(config)
        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)
        # Keep parameter-key parity with original BertConnectionLayer.
        # These two modules are unused in current forward, matching the upstream behavior.
        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(self, input_tensor1, attention_mask1,
                input_tensor2, attention_mask2,
                co_attention_mask=None, use_co_attention_mask=False,
                prefix_k=None, prefix_v=None, gate_scale=None):
        bi_output2, co_attention_probs = self.biattention(
            input_tensor1, attention_mask1,
            input_tensor2, attention_mask2,
            co_attention_mask, use_co_attention_mask,
            prefix_k=prefix_k, prefix_v=prefix_v, gate_scale=gate_scale)
        attention_output1 = self.biOutput(bi_output2, input_tensor1)
        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)
        return layer_output1, co_attention_probs


# ======================================================================
#  4.  PrefixBertEncoder
# ======================================================================

class PrefixBertEncoder(nn.Module):
    """``BertEncoder`` that routes prefix K/V tensors to specified layers.

    ``prefix_dict`` is a mapping  ``{ "v_layer.{i}" | "c_layer.{i}" :
    (prefix_k, prefix_v) }``  where each value is a pair of tensors
    ``[B, H, P, d]``.
    """

    def __init__(self, config):
        super().__init__()

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer

        # text layers – unchanged
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        # visual layers – prefix-aware
        v_layer = PrefixBertImageLayer(config)
        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)])

        # cross-attention layers – prefix-aware
        connect_layer = PrefixBertConnectionLayer(config)
        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer)
             for _ in range(len(config.v_biattention_id))])

    def forward(
        self,
        txt_embedding,
        image_embedding,
        txt_attention_mask,
        image_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
        prefix_dict=None,
        scale_dict=None,
    ):
        if prefix_dict is None:
            prefix_dict = {}
        if scale_dict is None:
            scale_dict = {}

        v_start = 0
        count = 0
        use_co_attention_mask = False

        state_lang_attn_scores = None
        state_visn_attn_scores = None

        for v_layer_id, t_layer_id in zip(
                self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id

            for idx in range(v_start, v_end):
                pk, pv = prefix_dict.get(f"v_layer.{idx}", (None, None))
                gs = scale_dict.get(f"v_layer.{idx}", None)
                image_embedding, state_visn_attn_scores = self.v_layer[idx](
                    image_embedding, image_attention_mask,
                    prefix_k=pk, prefix_v=pv, gate_scale=gs)

            if self.with_coattention:
                pk, pv = prefix_dict.get(f"c_layer.{count}", (None, None))
                gs = scale_dict.get(f"c_layer.{count}", None)
                image_embedding, state_lang_attn_scores = self.c_layer[count](
                    image_embedding, image_attention_mask,
                    txt_embedding, txt_attention_mask,
                    co_attention_mask, use_co_attention_mask,
                    prefix_k=pk, prefix_v=pv, gate_scale=gs)

            v_start = v_end
            count += 1

        for idx in range(v_start, len(self.v_layer)):
            pk, pv = prefix_dict.get(f"v_layer.{idx}", (None, None))
            gs = scale_dict.get(f"v_layer.{idx}", None)
            image_embedding, state_visn_attn_scores = self.v_layer[idx](
                image_embedding, image_attention_mask,
                prefix_k=pk, prefix_v=pv, gate_scale=gs)

        state_output = image_embedding[:, 0]
        return (state_output,
                state_lang_attn_scores[:, :, 0],
                state_visn_attn_scores[:, :, 0, 1:])


# ======================================================================
#  5.  PrefixVLNBert
# ======================================================================

class PrefixVLNBert(BertPreTrainedModel):
    """``VLNBert`` with ``PrefixBertEncoder`` for prefix-aware processing.

    Weight names are identical to the original ``VLNBert`` so that
    ``from_pretrained`` loads the same checkpoint transparently.
    """

    def __init__(self, config):
        super().__init__(config)

        self.img_feature_type = config.img_feature_type
        self.img_dim = config.img_feature_dim
        logger.info('PrefixVLNBert Image Dimension: {}'.format(self.img_dim))

        # word embedding
        self.embeddings = BertEmbeddings(config)

        # vision embedding
        self.v_embeddings = BertObjectEmbeddings(config)   # object
        self.scene_encoder = VisionEncoder(config)         # scene

        # encoder  ← prefix-aware
        self.encoder = PrefixBertEncoder(config)

        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn_prefix_mode = getattr(config, 'attn_prefix_mode', 'fedperfix_add')
        self.enable_lang_prefix = bool(getattr(config, 'enable_lang_prefix', True))
        self.attn_prefix_scale = float(getattr(config, 'prefix_scale', 1.0))
        mid_dim = int(getattr(config, 'prefix_mid_dim', 256))
        self.lang_last_adapter = AdditiveQKVAdapter(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size,
            mid_dim=mid_dim,
            kv_only=False,
        )

        self.apply(self.init_bert_weights)

    def _forward_lang_last(self, layer, hidden_states, attention_mask, gate_scale=None, prefix_k=None, prefix_v=None):
        """Run last text layer with optional additive/concat prefix injection."""
        attn = layer.attention.self
        mixed_query_layer = attn.query(hidden_states)
        mixed_key_layer = attn.key(hidden_states)
        mixed_value_layer = attn.value(hidden_states)

        if self.attn_prefix_mode == 'fedperfix_add':
            delta_qkv = self.lang_last_adapter(hidden_states)
            delta_q, delta_k, delta_v = torch.chunk(delta_qkv, 3, dim=-1)
            if isinstance(gate_scale, torch.Tensor):
                gs = gate_scale.view(-1, 1, 1)
            elif gate_scale is None:
                # Keep no-scale behavior consistent with visual/cross branches.
                gs = 0.0
            else:
                gs = float(gate_scale)
            alpha = self.attn_prefix_scale * gs
            mixed_query_layer = mixed_query_layer + alpha * delta_q
            mixed_key_layer = mixed_key_layer + alpha * delta_k
            mixed_value_layer = mixed_value_layer + alpha * delta_v

        query_layer = attn.transpose_for_scores(mixed_query_layer)
        key_layer = attn.transpose_for_scores(mixed_key_layer)
        value_layer = attn.transpose_for_scores(mixed_value_layer)

        if self.attn_prefix_mode == 'prefix_kv_concat' and prefix_k is not None and prefix_v is not None:
            prefix_len = prefix_k.size(2)
            key_layer = torch.cat([prefix_k, key_layer], dim=2)
            value_layer = torch.cat([prefix_v, value_layer], dim=2)
            prefix_mask = torch.zeros(
                hidden_states.size(0), 1, 1, prefix_len,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attn.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = attn.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (attn.all_head_size,))

        attention_output = layer.attention.output(context_layer, hidden_states)
        intermediate_output = layer.intermediate(attention_output)
        layer_output = layer.output(intermediate_output, attention_output)
        return layer_output

    def forward(self, mode, input_ids, token_type_ids=None,
                lang_masks=None, cand_feats=None, cand_masks=None,
                obj_feats=None, obj_pos=None, obj_masks=None,
                state_embeds=None, act_t=None,
                prefix_dict=None, scale_dict=None):
        """Same as ``VLNBert.forward`` with an additional *prefix_dict*.

        prefix_dict : dict | None
            ``{ "v_layer.{i}" | "c_layer.{i}" : (prefix_k, prefix_v) }``
        """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if scale_dict is None:
            scale_dict = {}

        extended_lang_attention_mask = lang_masks.unsqueeze(1).unsqueeze(2)
        extended_lang_attention_mask = extended_lang_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_lang_attention_mask = (
            1.0 - extended_lang_attention_mask) * -10000.0

        if mode == 'language':
            txt_embeds = self.embeddings(
                input_ids, token_type_ids=token_type_ids)

            for idx in range(self.config.num_hidden_layers):
                if idx == (self.config.num_hidden_layers - 1) and self.enable_lang_prefix:
                    lang_scale = scale_dict.get('lang_last', None)
                    lang_pk, lang_pv = (None, None)
                    if prefix_dict is not None:
                        lang_pk, lang_pv = prefix_dict.get('lang_last', (None, None))
                    txt_embeds = self._forward_lang_last(
                        self.encoder.layer[idx],
                        txt_embeds,
                        extended_lang_attention_mask,
                        gate_scale=lang_scale,
                        prefix_k=lang_pk,
                        prefix_v=lang_pv,
                    )
                else:
                    txt_embeds, _ = self.encoder.layer[idx](
                        txt_embeds, extended_lang_attention_mask)

            sequence_output = self.dropout(txt_embeds)
            pooled_output = self.t_pooler(sequence_output)
            return pooled_output, sequence_output[:, 1:]

        elif mode == 'visual':
            text_embeds = input_ids           # encoded language features
            device = input_ids.device
            batch_size, cand_len, _ = cand_feats.size()

            obj_embeds = self.v_embeddings(obj_feats, obj_pos, act_t)
            cand_embeds = self.scene_encoder(cand_feats)

            state_visn_embeds = torch.cat(
                [state_embeds.unsqueeze(1), cand_embeds, obj_embeds], dim=1)
            state_visn_masks = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                cand_masks, obj_masks], dim=1)

            extended_img_mask = state_visn_masks.unsqueeze(1).unsqueeze(2)
            extended_img_mask = extended_img_mask.to(
                dtype=next(self.parameters()).dtype)
            extended_img_mask = (1.0 - extended_img_mask) * -10000.0

            state_output, state_lang_attn_scores, state_visn_attn_scores = \
                self.encoder(
                    text_embeds, state_visn_embeds,
                    extended_lang_attention_mask, extended_img_mask,
                    output_all_attention_masks=True,
                    prefix_dict=prefix_dict, scale_dict=scale_dict)

            state_proj = self.v_pooler(
                self.dropout(state_output.unsqueeze(1)))

            visual_scores = state_visn_attn_scores.mean(dim=1)
            visual_action_scores = visual_scores[:, :cand_len]
            visual_object_scores = visual_scores[:, cand_len:]

            stop_scores, _ = visual_object_scores.max(1)
            visual_action_scores = torch.cat(
                [visual_action_scores, stop_scores.unsqueeze(1)], dim=-1)

            return state_proj, visual_action_scores, visual_object_scores

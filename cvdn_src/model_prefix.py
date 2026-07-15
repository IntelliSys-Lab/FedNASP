

import torch
import torch.nn as nn

from model import AttnDecoderLSTM


_PERSONAL_PREFIXES = ('prefix_tokens', 'adapter_down', 'adapter_up')


def is_personal_key(name: str) -> bool:
    
    return any(name.startswith(p) for p in _PERSONAL_PREFIXES)


class PrefixAttnDecoderLSTM(nn.Module):
    

    def __init__(
        self,
        input_action_size,
        output_action_size,
        embedding_size,
        hidden_size,
        dropout_ratio,
        feature_size=2048,
        prefix_len=4,
        adapter_mid_dim=64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_len = prefix_len

        # ---- Backbone (original decoder) ----
        self.base = AttnDecoderLSTM(
            input_action_size, output_action_size, embedding_size,
            hidden_size, dropout_ratio, feature_size,
        )

       
        self.prefix_tokens = nn.Parameter(
            torch.randn(1, prefix_len, hidden_size) * 0.02
        )

        
        self.adapter_down = nn.Linear(hidden_size, adapter_mid_dim)
        self.adapter_up = nn.Linear(adapter_mid_dim, hidden_size)
        
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.adapter_act = nn.ReLU()
        self.adapter_drop = nn.Dropout(dropout_ratio)

   

    def backbone_state_dict(self):
        """Return only the backbone (base.*) parameters on CPU."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()
                if not is_personal_key(k)}

    def personal_state_dict(self):
        """Return only the personal (prefix + adapter) parameters on CPU."""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()
                if is_personal_key(k)}
    

    # ------------------------------------------------------------------
    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None,
                gate_scales=None):
        """Single decoder step with prefix context + state adapter.

        Args:
            action:      [B, 1]
            feature:     [B, feature_size]
            h_0, c_0:   [B, hidden_size]
            ctx:         [B, seq_len, hidden_size]
            ctx_mask:    [B, seq_len] or None
            gate_scales: [B, 2]  — [:,0] prefix scale, [:,1] adapter scale.
                         If None both scales default to 1.

        Returns:
            h_1, c_1, alpha, logit   (same interface as AttnDecoderLSTM)
        """
        B = h_0.size(0)

        # ---- Block 1: prepend learnable prefix tokens to ctx ----
        prefix = self.prefix_tokens.expand(B, -1, -1)          # [B, P, H]
        if gate_scales is not None:
            s_prefix = gate_scales[:, 0].view(B, 1, 1)         # [B, 1, 1]
            prefix = prefix * s_prefix
        ctx_ext = torch.cat([prefix, ctx], dim=1)              # [B, P+L, H]

        # extend mask (prefix tokens are never masked)
        if ctx_mask is not None:
            prefix_mask = torch.zeros(
                B, self.prefix_len, dtype=ctx_mask.dtype, device=ctx_mask.device)
            ctx_mask_ext = torch.cat([prefix_mask, ctx_mask], dim=1)
        else:
            ctx_mask_ext = None

       
        action_embeds = self.base.embedding(action)             # [B, 1, E]
        action_embeds = action_embeds.squeeze(1)
        if action_embeds.dim() == 1:                            # batch_size == 1
            action_embeds = action_embeds.unsqueeze(0)
        concat_input = torch.cat((action_embeds, feature), 1)  # [B, E+F]
        drop = self.base.drop(concat_input)
        h_1, c_1 = self.base.lstm(drop, (h_0, c_0))
        h_1_drop = self.base.drop(h_1)
        h_tilde, alpha = self.base.attention_layer(h_1_drop, ctx_ext, ctx_mask_ext)

      
        residual = self.adapter_up(self.adapter_act(self.adapter_down(h_tilde)))
        residual = self.adapter_drop(residual)
        if gate_scales is not None:
            s_adapter = gate_scales[:, 1].view(B, 1)            # [B, 1]
            residual = residual * s_adapter
        h_tilde = h_tilde + residual

        logit = self.base.decoder2action(h_tilde)

        # Trim the prefix portion of the attention weights for logging compat.
        alpha_trimmed = alpha[:, self.prefix_len:]

        return h_1, c_1, alpha_trimmed, logit

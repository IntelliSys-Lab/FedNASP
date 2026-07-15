import torch
import torch.nn as nn
import torch.nn.functional as F


class GatePolicyTwoTower(nn.Module):
    

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        stats_dim: int = 2,
        tower_dim: int = 128,
        obs_dim: int = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_blocks = int(num_blocks)
        self.stats_dim = int(stats_dim)
        self.tower_dim = int(tower_dim)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None

        
        self.h_fc = nn.Linear(self.hidden_size, self.tower_dim)
        self.h_ln = nn.LayerNorm(self.tower_dim)
        self.h_drop = nn.Dropout(0.1)

        
        self.s_fc1 = nn.Linear(self.stats_dim, 32)
        self.s_fc2 = nn.Linear(32, self.tower_dim)
        self.s_drop = nn.Dropout(0.1)

        
        if self.obs_dim is not None:
            self.obs_proj = nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.obs_fc = nn.Linear(self.hidden_size, self.tower_dim)
            self.obs_ln = nn.LayerNorm(self.tower_dim)
            self.obs_drop = nn.Dropout(0.1)
        else:
            self.obs_proj = None
            self.obs_fc = None
            self.obs_ln = None
            self.obs_drop = None

        # Fusion head.
        fuse_in_dim = self.tower_dim * 2 + (self.tower_dim if self.obs_dim is not None else 0)
        self.fuse_fc1 = nn.Linear(fuse_in_dim, 128)
        self.fuse_drop = nn.Dropout(0.1)
        self.fuse_fc2 = nn.Linear(128, self.num_blocks)
        self.out_act = nn.Sigmoid()

    def forward(
        self,
        h_t: torch.Tensor,
        stats: torch.Tensor,
        obs_summary: torch.Tensor = None,
    ):
        stats = stats.to(device=h_t.device, dtype=torch.float32)
        h_t = h_t.to(dtype=torch.float32)

        h_embed = self.h_fc(h_t)
        h_embed = self.h_ln(h_embed)
        h_embed = F.gelu(h_embed)
        h_embed = self.h_drop(h_embed)

        s_embed = F.gelu(self.s_fc1(stats))
        s_embed = F.gelu(self.s_fc2(s_embed))
        s_embed = self.s_drop(s_embed)

        fused_parts = [h_embed, s_embed]
        if self.obs_dim is not None:
            if obs_summary is None:
                obs_summary = torch.zeros(
                    h_t.size(0), self.obs_dim, device=h_t.device, dtype=torch.float32
                )
            else:
                obs_summary = obs_summary.to(device=h_t.device, dtype=torch.float32)
            o_embed = self.obs_proj(obs_summary)
            o_embed = self.obs_fc(o_embed)
            o_embed = self.obs_ln(o_embed)
            o_embed = F.gelu(o_embed)
            o_embed = self.obs_drop(o_embed)
            fused_parts.append(o_embed)

        fused = torch.cat(fused_parts, dim=-1)
        fused = F.gelu(self.fuse_fc1(fused))
        fused = self.fuse_drop(fused)
        s_raw = self.fuse_fc2(fused)
        s = self.out_act(s_raw).to(dtype=torch.float32)

        return s

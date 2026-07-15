"""PrefixVLNBERT wrapper for local prefix personalization (vilbert path)."""

import os
import torch
import torch.nn as nn
from param import args

from vlnbert.vlnbert_prefix import (
    PrefixLayer,
    PrefixVLNBert,
    BertConfig,
)


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def get_prefix_vlnbert(args_obj, config=None):
    """Instantiate PrefixVLNBert (vilbert) with prefix-related config fields."""
    model_name_or_path = args_obj.init_bert_file

    vis_config = BertConfig.from_json_file(
        os.path.join('datasets/vln-bert', 'bert_base_6_layer_6_connect.json')
    )
    vis_config.img_feature_dim = 2048 + args_obj.angle_feat_size
    vis_config.img_feature_type = args_obj.features
    vis_config.layer_norm_eps = 1e-12
    vis_config.hidden_dropout_prob = 0.3
    vis_config.v_hidden_dropout_prob = 0.3

    vis_config.attn_prefix_mode = getattr(args_obj, 'attn_prefix_mode', 'fedperfix_add')
    vis_config.prefix_mid_dim = int(getattr(args_obj, 'prefix_mid_dim', 256))
    vis_config.prefix_scale = float(getattr(args_obj, 'prefix_scale', 1.0))
    vis_config.enable_lang_prefix = bool(getattr(args_obj, 'enable_lang_prefix', True))
    vis_config.enable_vis_prefix = bool(getattr(args_obj, 'enable_vis_prefix', True))
    vis_config.enable_bi_prefix = bool(getattr(args_obj, 'enable_bi_prefix', True))

    if model_name_or_path:
        visual_model = PrefixVLNBert.from_pretrained(model_name_or_path, config=vis_config)
    else:
        visual_model = PrefixVLNBert(vis_config)
    return visual_model


class PrefixVLNBERT(nn.Module):
    """VLNBERT wrapper with per-block local prefixes and external gate scales."""

    def __init__(self, directions=4, feature_size=2048 + 128,
                 prefix_len=8, prefix_modules='infer', gate_hidden=256):
        super().__init__()
        print('\nInitializing PrefixVLNBERT ...')

        self.vln_bert = get_prefix_vlnbert(args, config=None)
        self.vln_bert.config.directions = directions

        v_hidden_size = self.vln_bert.config.v_hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(v_hidden_size + args.angle_feat_size, v_hidden_size),
            nn.Tanh(),
        )
        self.action_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)
        self.drop_env = nn.Dropout(p=args.featdropout)

        self.enable_lang_prefix = bool(getattr(args, 'enable_lang_prefix', True))
        self.enable_vis_prefix = bool(getattr(args, 'enable_vis_prefix', True))
        self.enable_bi_prefix = bool(getattr(args, 'enable_bi_prefix', True))
        self.attn_prefix_mode = getattr(args, 'attn_prefix_mode', 'fedperfix_add')

        if isinstance(prefix_modules, str) and prefix_modules.strip().lower() in ('', 'infer', 'auto'):
            self.prefix_module_names = self._infer_prefix_module_names()
        else:
            self.prefix_module_names = [m.strip() for m in str(prefix_modules).split(',') if m.strip()]
        self.num_prefix_modules = len(self.prefix_module_names)

        # Prefix layers provide the additive attention adapters used by ours.
        self.prefix_layers = nn.ModuleDict()
        for name in self.prefix_module_names:
            if name == 'lang_last':
                num_heads = self.vln_bert.config.num_attention_heads
                head_size = self.vln_bert.config.hidden_size // num_heads
            elif name.startswith('v_layer'):
                num_heads = self.vln_bert.config.v_num_attention_heads
                head_size = self.vln_bert.config.v_hidden_size // num_heads
            elif name.startswith('c_layer'):
                num_heads = self.vln_bert.config.bi_num_attention_heads
                head_size = self.vln_bert.config.bi_hidden_size // num_heads
            else:
                raise ValueError(f'Unknown prefix module name: {name}')
            self.prefix_layers[name.replace('.', '_')] = PrefixLayer(
                num_heads=num_heads, head_size=head_size, prefix_len=prefix_len
            )

        print(f'  attn_prefix_mode = {self.attn_prefix_mode}')
        print(f'  prefix_modules   = {self.prefix_module_names}')
        print(f'  prefix_len       = {prefix_len}')
        print(f'  num_blocks       = {self.num_prefix_modules}')

    def _infer_prefix_module_names(self):
        cfg = self.vln_bert.config
        names = []
        if self.enable_lang_prefix:
            names.append('lang_last')

        v_start = 0
        count = 0
        for v_end in cfg.v_biattention_id:
            if self.enable_vis_prefix:
                for idx in range(v_start, v_end):
                    names.append(f'v_layer.{idx}')
            if self.enable_bi_prefix:
                names.append(f'c_layer.{count}')
            v_start = v_end
            count += 1

        if self.enable_vis_prefix:
            for idx in range(v_start, cfg.v_num_hidden_layers):
                names.append(f'v_layer.{idx}')
        return names

    def build_scale_dict(self, prefix_scales):
        if prefix_scales is None:
            return {}
        if prefix_scales.dim() != 2 or prefix_scales.size(1) != self.num_prefix_modules:
            raise ValueError(
                f'prefix_scales shape mismatch: got {tuple(prefix_scales.shape)}, '
                f'expected [B, {self.num_prefix_modules}]'
            )
        return {name: prefix_scales[:, i] for i, name in enumerate(self.prefix_module_names)}

    def compute_prefix_dict(self, batch_size, prefix_scales=None, scale_dict=None):
        # REMOVE: the legacy prefix_kv_concat path is excluded; additive
        # adapters consume scale_dict directly in vlnbert_prefix.
        return None
        '''
        if scale_dict is None:
            scale_dict = self.build_scale_dict(prefix_scales)
        out = {}
        for name in self.prefix_module_names:
            key = name.replace('.', '_')
            gate_scale = scale_dict.get(name, None)
            if gate_scale is None:
                gate_scale = 1.0
            pk, pv = self.prefix_layers[key](batch_size, gate_scale)
            out[name] = (pk, pv)
        return out
        '''

    def forward(self, mode, sentence, token_type_ids=None, position_ids=None,
                lang_masks=None, action_feats=None, pano_feats=None,
                cand_feats=None, cand_masks=None,
                obj_feats=None, obj_pos=None, obj_masks=None,
                h_t=None, already_dropfeat=False, act_t=None,
                prefix_dict=None, prefix_scales=None, scale_dict=None):
        if scale_dict is None:
            scale_dict = self.build_scale_dict(prefix_scales)

        if mode == 'language':
            if prefix_dict is None:
                prefix_dict = self.compute_prefix_dict(
                    batch_size=sentence.size(0),
                    prefix_scales=prefix_scales,
                    scale_dict=scale_dict,
                )
            init_state, encoded_sentence = self.vln_bert(
                mode, sentence, lang_masks=lang_masks,
                prefix_dict=prefix_dict, scale_dict=scale_dict
            )
            return init_state, encoded_sentence

        if mode == 'visual':
            state_action_embed = torch.cat((h_t, action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)

            if not already_dropfeat:
                cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
                obj_feats[..., :-4] = self.drop_env(obj_feats[..., :-4])

            if prefix_dict is None:
                prefix_dict = self.compute_prefix_dict(
                    batch_size=cand_feats.size(0),
                    prefix_scales=prefix_scales,
                    scale_dict=scale_dict,
                )

            h_t, logit, logit_obj = self.vln_bert(
                mode, sentence,
                cand_feats=cand_feats,
                obj_feats=obj_feats, obj_pos=obj_pos, act_t=act_t,
                lang_masks=lang_masks,
                cand_masks=cand_masks, obj_masks=obj_masks,
                state_embeds=state_with_action,
                prefix_dict=prefix_dict, scale_dict=scale_dict,
            )
            return h_t, logit, logit_obj

        raise ValueError(f'Unknown mode: {mode}')


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state2value = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

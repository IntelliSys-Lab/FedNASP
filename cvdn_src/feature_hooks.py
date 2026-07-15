"""
REMOVE: This utility was used exclusively by the deleted Ditto+MK-MMD branch.
FedAvg and ours do not register feature hooks. Delete this file after review.

Feature hook utilities.

Register forward hooks on named sub-modules to capture intermediate activations
(e.g. h_tilde from SoftDotAttention) without changing the model code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class FeatureHookBuffer:
    """Manages forward hooks and stores captured features for a set of layers.

    Usage:
        buf = FeatureHookBuffer(model, ["decoder.attention_layer"])
        buf.enable()
        model(...)          # features are accumulated
        feats = buf.get()   # {"decoder.attention_layer": [N*T, D]}
        buf.clear()
        buf.disable()

    For SoftDotAttention, forward() returns (h_tilde, attn).
    The hook stores the *first* element of the output tuple (h_tilde).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        pick_index: int = 0,
        detach: bool = True,
    ):
        """
        Args:
            model: the nn.Module to hook.
            layer_names: list of dot-separated names, e.g. "attention_layer".
            pick_index: which element to pick when the layer returns a tuple.
                        0 → h_tilde  for SoftDotAttention.
            detach: if True, detach captured tensors from the computation graph.
                    Set False for the local model so MK-MMD gradients flow back.
                    Set True for the frozen global model (no grads needed).
        """
        self._model = model
        self._layer_names = layer_names
        self._pick_index = pick_index
        self._detach = detach
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._buffers: Dict[str, List[torch.Tensor]] = {
            name: [] for name in layer_names
        }
        self._enabled = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _make_hook(self, layer_name: str):
        """Return a hook function that stores the output in self._buffers."""
        def hook_fn(module, input, output):
            if not self._enabled:
                return
            if isinstance(output, (tuple, list)):
                feat = output[self._pick_index]
            else:
                feat = output
            if self._detach:
                feat = feat.detach()
            self._buffers[layer_name].append(feat)
        return hook_fn

    def _resolve_layer(self, layer_name: str) -> nn.Module:
        """Resolve a dot-separated name to the actual sub-module."""
        parts = layer_name.split(".")
        mod = self._model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                raise AttributeError(
                    f"Module {self._model.__class__.__name__} has no "
                    f"sub-module '{layer_name}' (failed at '{p}')"
                )
        return mod

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enable(self) -> "FeatureHookBuffer":
        """Register hooks and start capturing."""
        if self._hooks:
            self.disable()
        for name in self._layer_names:
            layer = self._resolve_layer(name)
            h = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)
        self._enabled = True
        return self

    def disable(self) -> "FeatureHookBuffer":
        """Remove hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._enabled = False
        return self

    def clear(self) -> "FeatureHookBuffer":
        """Clear accumulated features."""
        for name in self._layer_names:
            self._buffers[name].clear()
        return self

    def get(self) -> Dict[str, torch.Tensor]:
        """Return concatenated features per layer.

        Returns:
            dict mapping layer_name -> Tensor [total_samples, D].
            If no features were captured, the tensor has shape [0, 0].
        """
        out: Dict[str, torch.Tensor] = {}
        for name in self._layer_names:
            buf = self._buffers[name]
            if buf:
                out[name] = torch.cat(buf, dim=0)
            else:
                out[name] = torch.empty(0, 0)
        return out

    def get_and_clear(self) -> Dict[str, torch.Tensor]:
        """Convenience: get() then clear()."""
        feats = self.get()
        self.clear()
        return feats

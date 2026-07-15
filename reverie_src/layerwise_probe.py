#!/usr/bin/env python3
"""Layer-wise shape probe for REVERIE model_CA (ViLBERT/CA branch).

This script runs one forward pass for both branches:
- language branch
- visual branch

It records module-level input/output tensor shapes via forward hooks and exports:
- CSV (one row per module call)
- JSON (full structured records)

Usage example:
  python reverie_src/layerwise_probe.py \
    --probe_batch 2 --probe_seq_len 20 --probe_cand_len 8 --probe_obj_len 6 \
    --vlnbert vilbert --angleFeatSize 128 --init_bert_file datasets/vln-bert/xxx.bin
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch


def _build_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe per-layer input/output shapes for model_CA language/visual branches."
    )
    parser.add_argument("--probe_batch", type=int, default=2, help="Dummy batch size")
    parser.add_argument("--probe_seq_len", type=int, default=20, help="Instruction token length (>=2)")
    parser.add_argument("--probe_cand_len", type=int, default=8, help="#candidate viewpoints")
    parser.add_argument("--probe_obj_len", type=int, default=6, help="#candidate objects")
    parser.add_argument("--probe_seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--probe_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for probing",
    )
    parser.add_argument(
        "--probe_out_dir",
        type=str,
        default="snap/layerwise_probe",
        help="Output directory for CSV/JSON logs",
    )
    parser.add_argument(
        "--probe_all_modules",
        action="store_true",
        help="Trace all modules (default traces leaf modules only)",
    )
    parser.add_argument(
        "--probe_max_print",
        type=int,
        default=40,
        help="Max rows printed per branch in stdout summary",
    )
    return parser


def _shape_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_tree(v) for k, v in x.items()}
    if x is None:
        return None
    return str(type(x).__name__)


def _format_shape(x: Any) -> str:
    return json.dumps(_shape_tree(x), ensure_ascii=True)


@dataclass
class TraceEvent:
    branch: str
    order: int
    module_name: str
    module_type: str
    input_shape: str
    output_shape: str
    own_params: int


class ShapeTracer:
    def __init__(self, leaf_only: bool = True):
        self.leaf_only = leaf_only
        self._events: List[TraceEvent] = []
        self._handles = []
        self._order = 0
        self._branch = "unknown"

    @property
    def events(self) -> List[TraceEvent]:
        return self._events

    def set_branch(self, branch: str) -> None:
        self._branch = branch

    def _hook(self, module_name: str, module: torch.nn.Module):
        def _fn(_module, inputs, outputs):
            self._order += 1
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            self._events.append(
                TraceEvent(
                    branch=self._branch,
                    order=self._order,
                    module_name=module_name,
                    module_type=module.__class__.__name__,
                    input_shape=_format_shape(inputs),
                    output_shape=_format_shape(outputs),
                    own_params=own_params,
                )
            )

        return _fn

    def attach(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if name == "":
                continue
            if self.leaf_only and any(module.children()):
                continue
            handle = module.register_forward_hook(self._hook(name, module))
            self._handles.append(handle)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _save_events(events: List[TraceEvent], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "branch": e.branch,
            "order": e.order,
            "module_name": e.module_name,
            "module_type": e.module_type,
            "input_shape": e.input_shape,
            "output_shape": e.output_shape,
            "own_params": e.own_params,
        }
        for e in events
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "branch",
                "order",
                "module_name",
                "module_type",
                "input_shape",
                "output_shape",
                "own_params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)


def _print_summary(events: List[TraceEvent], max_rows: int) -> None:
    by_branch: Dict[str, List[TraceEvent]] = {}
    for e in events:
        by_branch.setdefault(e.branch, []).append(e)

    for branch in ("language", "visual"):
        items = by_branch.get(branch, [])
        print(f"\n[{branch}] traced calls: {len(items)}")
        print("order | module_name | module_type | input_shape -> output_shape")
        for e in items[:max_rows]:
            print(
                f"{e.order:5d} | {e.module_name} | {e.module_type} | "
                f"{e.input_shape} -> {e.output_shape}"
            )
        if len(items) > max_rows:
            print(f"... ({len(items) - max_rows} more rows omitted)")


def main() -> None:
    probe_parser = _build_probe_parser()
    probe_args, remaining = probe_parser.parse_known_args()

    if probe_args.probe_seq_len < 2:
        raise ValueError("--probe_seq_len must be >= 2")

    # Let project parser consume the remaining args.
    orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + remaining
    from param import args  # noqa: WPS433
    import model_CA  # noqa: WPS433

    sys.argv = orig_argv

    seed = probe_args.probe_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args.vlnbert = "vilbert"

    if probe_args.probe_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(probe_args.probe_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--probe_device=cuda was requested but CUDA is not available.")

    model = model_CA.VLNBERT().to(device)
    model.eval()

    tracer = ShapeTracer(leaf_only=(not probe_args.probe_all_modules))
    tracer.attach(model)

    bsz = probe_args.probe_batch
    seq_len = probe_args.probe_seq_len
    cand_len = probe_args.probe_cand_len
    obj_len = probe_args.probe_obj_len

    vocab_size = int(getattr(model.vln_bert.config, "vocab_size", 30522))
    img_dim = int(model.vln_bert.config.img_feature_dim)
    obj_feat_dim = int(model.vln_bert.config.v_feature_size + 4)
    angle_dim = int(args.angle_feat_size)

    sentence = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.long, device=device)
    token_type_ids = torch.zeros_like(sentence)
    lang_masks = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)

    with torch.no_grad():
        tracer.set_branch("language")
        init_state, encoded_sentence = model(
            mode="language",
            sentence=sentence,
            token_type_ids=token_type_ids,
            lang_masks=lang_masks,
        )

        action_feats = torch.randn((bsz, angle_dim), dtype=torch.float32, device=device)
        cand_feats = torch.randn((bsz, cand_len, img_dim), dtype=torch.float32, device=device)
        obj_feats = torch.randn((bsz, obj_len, obj_feat_dim), dtype=torch.float32, device=device)
        obj_pos = torch.randn((bsz, obj_len, 5), dtype=torch.float32, device=device)

        cand_masks = torch.ones((bsz, cand_len), dtype=torch.bool, device=device)
        obj_masks = torch.ones((bsz, obj_len), dtype=torch.bool, device=device)

        tracer.set_branch("visual")
        _h_t, _logit, _logit_obj = model(
            mode="visual",
            sentence=encoded_sentence,
            token_type_ids=None,
            lang_masks=lang_masks[:, 1:],
            action_feats=action_feats,
            cand_feats=cand_feats,
            cand_masks=cand_masks,
            obj_feats=obj_feats,
            obj_pos=obj_pos,
            obj_masks=obj_masks,
            h_t=init_state,
            act_t=0,
        )

    tracer.detach()

    out_dir = Path(probe_args.probe_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = out_dir / "layer_shapes_all.csv"
    all_json = out_dir / "layer_shapes_all.json"
    _save_events(tracer.events, all_csv, all_json)

    lang_events = [e for e in tracer.events if e.branch == "language"]
    vis_events = [e for e in tracer.events if e.branch == "visual"]
    _save_events(lang_events, out_dir / "layer_shapes_language.csv", out_dir / "layer_shapes_language.json")
    _save_events(vis_events, out_dir / "layer_shapes_visual.csv", out_dir / "layer_shapes_visual.json")

    _print_summary(tracer.events, max_rows=probe_args.probe_max_print)

    print("\nSaved files:")
    print(f"- {all_csv}")
    print(f"- {all_json}")
    print(f"- {out_dir / 'layer_shapes_language.csv'}")
    print(f"- {out_dir / 'layer_shapes_visual.csv'}")


if __name__ == "__main__":
    main()

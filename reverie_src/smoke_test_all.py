#!/usr/bin/env python3
"""
Comprehensive smoke test for reverie_src prefix + gating pipeline.

Covers:
  P0: Basic runnability (syntax, 1-5 iter smoke, backward grad check)
  P1: Functional consistency (gating off, mode switching, branch toggles)
  P2: Stability & logging (NaN/Inf, log keys, RL/teacher feedback)

Usage (from repo root, conda activate m3d):
  python reverie_src/smoke_test_all.py

Loads data ONCE and reuses across all tests.
"""

import sys
import os

# ---- Override sys.argv BEFORE any other import (param.py parses at import time) ----
sys.argv = [
    'smoke_test',
    '--vlnbert', 'vilbert',
    '--init_bert_file', 'datasets/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin',
    '--features', 'img_features/ResNet-152-places365.tsv',
    '--scan_idx', '0',
    '--local_iters', '3',
    '--eval_every', '100',
    '--batchSize', '2',
    '--optim', 'adamW',
    '--name', 'smoke_test_tmp',
    '--maxAction', '6',
    # Prefix mode on
    '--prefix_mode',
    '--use_gating_prefix',
    '--attn_prefix_mode', 'fedperfix_add',
    '--feedback', 'sample',
    '--freeze_backbone',
]

import subprocess
import traceback
import math
import gc
import torch
import numpy as np
import pickle as pkl
from collections import defaultdict

from param import args
from utils import read_img_features
from env import R2RBatchScan
from vlnbert.vlnbert_init import get_tokenizer
from gating_policy import GatePolicyTwoTower
from agent import PrefixSeq2SeqAgent

# ======================================================================
# Colour helpers
# ======================================================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

pass_count = 0
fail_count = 0
skip_count = 0
results_log = []

def PASS(tag, msg=""):
    global pass_count
    pass_count += 1
    results_log.append(("PASS", tag, msg))
    print(f"  {GREEN}[PASS]{RESET} {tag} {msg}")
    sys.stdout.flush()

def FAIL(tag, msg=""):
    global fail_count
    fail_count += 1
    results_log.append(("FAIL", tag, msg))
    print(f"  {RED}[FAIL]{RESET} {tag} {msg}")
    sys.stdout.flush()

def SKIP(tag, msg=""):
    global skip_count
    skip_count += 1
    results_log.append(("SKIP", tag, msg))
    print(f"  {YELLOW}[SKIP]{RESET} {tag} {msg}")
    sys.stdout.flush()


# ======================================================================
# P0.1  Syntax check
# ======================================================================
def p0_1_syntax():
    print(f"\n{'='*60}")
    print("P0.1  Syntax check (py_compile)")
    print(f"{'='*60}")
    sys.stdout.flush()
    base = os.path.dirname(os.path.abspath(__file__))
    py_files = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'snap', '.git')]
        for f in files:
            if f.endswith('.py') and f != os.path.basename(__file__):
                py_files.append(os.path.join(root, f))

    all_ok = True
    for fpath in sorted(py_files):
        rel = os.path.relpath(fpath, os.path.dirname(base))
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', fpath],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, 'PYTHONPYCACHEPREFIX': '/tmp/pycache'}
            )
            if result.returncode != 0:
                FAIL(f"syntax:{rel}", result.stderr.strip())
                all_ok = False
        except Exception as e:
            FAIL(f"syntax:{rel}", str(e))
            all_ok = False

    if all_ok:
        PASS("P0.1 syntax", f"All {len(py_files)} files compile OK")
    return all_ok


# ======================================================================
# Shared data loading
# ======================================================================
_shared = {}

def load_data():
    if _shared:
        return _shared
    print("\n[shared] Loading features + tokenizer + env ...")
    sys.stdout.flush()

    tok = get_tokenizer(args)
    feat_dict = read_img_features(args.features, test_only=False)
    with open('img_features/REVERIE_obj_feats.pkl', 'rb') as f_obj:
        obj_feats = pkl.load(f_obj)

    train_env = R2RBatchScan(feat_dict, obj_feats, batch_size=args.batchSize,
                              splits=['train'], tokenizer=tok)
    scans_list = list(train_env.scans_list)
    scan_id = scans_list[0]

    _shared.update(dict(tok=tok, feat_dict=feat_dict, obj_feats=obj_feats,
                        train_env=train_env, scan_id=scan_id))
    print(f"[shared] Loaded. scan={scan_id}, nscans={len(scans_list)}, "
          f"data={len(train_env.data.get(scan_id, []))}")
    sys.stdout.flush()
    return _shared


def make_agent(freeze_backbone=True):
    d = load_data()
    d['train_env'].set_current_scan(d['scan_id'])
    agent = PrefixSeq2SeqAgent(
        d['train_env'], "", d['tok'], args.maxAction,
        prefix_len=getattr(args, 'prefix_len', 8),
        prefix_modules=getattr(args, 'prefix_modules', 'infer'),
        gate_hidden=getattr(args, 'gate_hidden', 256),
        freeze_backbone=freeze_backbone)
    agent.env = d['train_env']
    return agent


def cleanup(agent):
    del agent
    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================
# P0.2  Smoke test
# ======================================================================
def p0_2_smoke():
    print(f"\n{'='*60}")
    print("P0.2  Smoke test (prefix + gating, 2 iters, feedback=sample)")
    print(f"{'='*60}")
    sys.stdout.flush()
    try:
        agent = make_agent()
        agent.logs = defaultdict(list)
        agent.train(2, feedback='sample')

        if len(agent.losses) >= 2:
            PASS("P0.2 smoke",
                 f"losses={[round(l, 4) for l in agent.losses[:3]]}")
        else:
            FAIL("P0.2 smoke",
                 f"Expected >=2 losses, got {len(agent.losses)}")
        return agent
    except Exception as e:
        FAIL("P0.2 smoke", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ======================================================================
# P0.3  Backward grad check
# ======================================================================
def p0_3_grad(agent):
    print(f"\n{'='*60}")
    print("P0.3  Backward propagation grad check")
    print(f"{'='*60}")
    sys.stdout.flush()
    if agent is None:
        SKIP("P0.3 grad", "agent unavailable")
        return

    # gate_policy grads
    nz, total = 0, 0
    for name, p in agent.gate_policy.named_parameters():
        total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            nz += 1
    if nz > 0:
        PASS("P0.3 gate_policy grad", f"{nz}/{total} non-zero")
    else:
        FAIL("P0.3 gate_policy grad", f"0/{total} non-zero!")

    # attn_prefix_* grads
    nz, total, ok_names, zero_names = 0, 0, [], []
    for name, p in agent.vln_bert.named_parameters():
        if 'attn_prefix_' in name:
            total += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                nz += 1
                ok_names.append(name)
            else:
                zero_names.append(name)
    if nz > 0:
        PASS("P0.3 attn_prefix grad",
             f"{nz}/{total} non-zero (e.g. {ok_names[0]})")
    else:
        FAIL("P0.3 attn_prefix grad",
             f"0/{total} non-zero! examples: {zero_names[:3]}")


# ======================================================================
# P1.1  Gating off
# ======================================================================
def p1_1_gating_off():
    print(f"\n{'='*60}")
    print("P1.1  Gating off (use_gating_prefix=False)")
    print(f"{'='*60}")
    sys.stdout.flush()
    old = args.use_gating_prefix
    try:
        args.use_gating_prefix = False
        agent = make_agent()
        agent.logs = defaultdict(list)
        agent.train(2, feedback='teacher')
        args.use_gating_prefix = old

        gm = agent.logs.get('gate_mean', [])
        if gm:
            avg = sum(gm) / len(gm)
            if abs(avg - 1.0) < 1e-6:
                PASS("P1.1 gating off", f"gate_mean={avg:.6f} == 1.0")
            else:
                FAIL("P1.1 gating off", f"gate_mean={avg:.6f}, expected 1.0")
        else:
            FAIL("P1.1 gating off", "No gate_mean logged")
        cleanup(agent)
    except Exception as e:
        args.use_gating_prefix = old
        FAIL("P1.1 gating off", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ======================================================================
# P1.2  Mode switching
# ======================================================================
def p1_2_mode_switch():
    print(f"\n{'='*60}")
    print("P1.2  Mode switching (prefix_kv_concat)")
    print(f"{'='*60}")
    sys.stdout.flush()
    old = args.attn_prefix_mode
    try:
        args.attn_prefix_mode = 'prefix_kv_concat'
        agent = make_agent()
        agent.logs = defaultdict(list)
        agent.train(1, feedback='teacher')
        args.attn_prefix_mode = old

        if len(agent.losses) >= 1:
            PASS("P1.2 prefix_kv_concat", f"loss={agent.losses[0]:.4f}")
        else:
            FAIL("P1.2 prefix_kv_concat", "No losses")
        cleanup(agent)
    except Exception as e:
        args.attn_prefix_mode = old
        FAIL("P1.2 prefix_kv_concat", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ======================================================================
# P1.3  Branch toggles
# ======================================================================
def p1_3_branch(label, lang, vis, bi):
    old = (args.enable_lang_prefix, args.enable_vis_prefix, args.enable_bi_prefix)
    try:
        args.enable_lang_prefix = lang
        args.enable_vis_prefix = vis
        args.enable_bi_prefix = bi
        agent = make_agent()
        agent.logs = defaultdict(list)
        agent.train(1, feedback='teacher')
        args.enable_lang_prefix, args.enable_vis_prefix, args.enable_bi_prefix = old

        if len(agent.losses) >= 1:
            PASS(f"P1.3 {label}",
                 f"loss={agent.losses[0]:.4f}, "
                 f"modules={agent.vln_bert.num_prefix_modules}")
        else:
            FAIL(f"P1.3 {label}", "No losses")
        cleanup(agent)
    except Exception as e:
        args.enable_lang_prefix, args.enable_vis_prefix, args.enable_bi_prefix = old
        FAIL(f"P1.3 {label}", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def p1_3_all():
    print(f"\n{'='*60}")
    print("P1.3  Branch toggle tests")
    print(f"{'='*60}")
    sys.stdout.flush()
    p1_3_branch("lang_only", True, False, False)
    p1_3_branch("vis_only", False, True, False)
    p1_3_branch("bi_only", False, False, True)


# ======================================================================
# P2.1  Numerical stability
# ======================================================================
def p2_1_numerical(agent):
    print(f"\n{'='*60}")
    print("P2.1  Numerical stability (NaN/Inf)")
    print(f"{'='*60}")
    sys.stdout.flush()
    if agent is None:
        SKIP("P2.1", "agent unavailable")
        return

    keys = ['gate_mean', 'entropy', 'gate_entropy_corr']
    for i in range(30):
        k = f'gate_block_{i}'
        if k in agent.logs:
            keys.append(k)

    ok = True
    for key in keys:
        vals = agent.logs.get(key, [])
        for v in vals:
            if math.isnan(v) or math.isinf(v):
                FAIL(f"P2.1 {key}", f"bad value found, vals={vals[:5]}")
                ok = False
                break

    if ok:
        PASS("P2.1 numerical", "No NaN/Inf in monitored keys")


# ======================================================================
# P2.2  Logging completeness
# ======================================================================
def p2_2_logging(agent):
    print(f"\n{'='*60}")
    print("P2.2  Logging completeness")
    print(f"{'='*60}")
    sys.stdout.flush()
    if agent is None:
        SKIP("P2.2", "agent unavailable")
        return

    for key in ['gate_mean', 'gate_entropy_corr', 'gate_block_0']:
        vals = agent.logs.get(key, [])
        if vals:
            PASS(f"P2.2 log:{key}", f"{len(vals)} entries, sample={vals[0]:.4f}")
        else:
            FAIL(f"P2.2 log:{key}", "Missing/empty")


# ======================================================================
# P2.3  Feedback modes
# ======================================================================
def p2_3_feedback():
    print(f"\n{'='*60}")
    print("P2.3  RL/Teacher feedback modes")
    print(f"{'='*60}")
    sys.stdout.flush()

    for fb in ['teacher', 'sample']:
        try:
            agent = make_agent()
            agent.logs = defaultdict(list)
            agent.train(2, feedback=fb)
            if len(agent.losses) >= 2:
                PASS(f"P2.3 feedback={fb}",
                     f"losses={[round(l, 4) for l in agent.losses[:2]]}")
            else:
                FAIL(f"P2.3 feedback={fb}",
                     f"got {len(agent.losses)} losses")
            cleanup(agent)
        except Exception as e:
            FAIL(f"P2.3 feedback={fb}", f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 60)
    print("  REVERIE prefix+gating comprehensive smoke test")
    print("=" * 60)
    sys.stdout.flush()

    p0_1_syntax()
    agent = p0_2_smoke()
    p0_3_grad(agent)
    p2_1_numerical(agent)
    p2_2_logging(agent)
    if agent:
        cleanup(agent)

    p1_1_gating_off()
    p1_2_mode_switch()
    p1_3_all()
    p2_3_feedback()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for status, tag, msg in results_log:
        c = GREEN if status == "PASS" else (RED if status == "FAIL" else YELLOW)
        print(f"  {c}[{status}]{RESET} {tag} {msg}")
    print(f"\n  Total: {GREEN}{pass_count} PASS{RESET}, "
          f"{RED}{fail_count} FAIL{RESET}, "
          f"{YELLOW}{skip_count} SKIP{RESET}")

    if fail_count > 0:
        print(f"\n  {RED}⚠  {fail_count} test(s) FAILED{RESET}")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}✓  All tests passed!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()

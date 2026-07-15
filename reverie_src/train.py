import torch
import torch.distributed as dist

import os
import time
import json
import random
import math
import copy
import numpy as np
import pickle as pkl
from collections import defaultdict
import logging
from datetime import datetime
import re
from utils import read_vocab, write_vocab, build_vocab, padding_idx, timeSince, read_img_features, print_progress
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import gc
# import warnings
# warnings.filterwarnings("ignore")
import tracemalloc
try:
    import psutil
except Exception:
    psutil = None
from tensorboardX import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from vlnbert.vlnbert_init import get_tokenizer

# Setup logging
log_dir = 'logs/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

snap_dir = 'snap/%s' % args.name
if not os.path.exists(snap_dir):
    os.makedirs(snap_dir, exist_ok=True)

# Use safe filename (replace / with _)
safe_name = args.name.replace('/', '_')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=os.path.join(log_dir, f"{safe_name}_{timestamp}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console_handler)

PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

features = args.features
feedback_method = args.feedback  # teacher or sample

logger.info("===== Training Arguments =====")
for k, v in vars(args).items():
    logger.info(f"{k}: {v}")
logger.info("==============================")
import os
print("MEM_DEBUG =", os.environ.get("MEM_DEBUG"), flush=True)


class _AttrDataParallel(torch.nn.DataParallel):
    """DataParallel wrapper with transparent attribute/state_dict access.

    This keeps existing agent code unchanged (it accesses self.vln_bert.xxx
    fields directly) while still enabling multi-GPU batch splitting.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


class _AttrDDP(torch.nn.parallel.DistributedDataParallel):
    """DDP wrapper with transparent attribute/state_dict access."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def _dist_rank():
    return dist.get_rank() if _dist_is_initialized() else 0


def _dist_world_size():
    return dist.get_world_size() if _dist_is_initialized() else 1


def _dist_is_main():
    return _dist_rank() == 0


def _wandb_enabled(force=False):
    if not _dist_is_main():
        return False
    if force:
        return True
    return bool(getattr(args, 'use_wandb', False))


def _maybe_init_wandb(force=False):
    if not _wandb_enabled(force=force):
        return None
    if not WANDB_AVAILABLE:
        msg = '[wandb] wandb import failed.'
        if force:
            raise RuntimeError(f'{msg} REVERIE ours requires wandb logging.')
        logger.warning(f'{msg} skip wandb logging.')
        return None
    if wandb.run is not None:
        return wandb.run
    return wandb.init(
        project=getattr(args, 'wandb_project', 'fedvln'),
        entity=getattr(args, 'wandb_entity', None),
        name=args.name,
        job_type='reverie-ours',
        dir=log_dir,
        tags=['reverie', 'ours', str(args.vlnbert)],
        config=vars(args),
    )


def _maybe_wandb_log(metrics, step=None, force=False):
    if not _wandb_enabled(force=force):
        return
    if (not WANDB_AVAILABLE) or wandb.run is None:
        return
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)


def _maybe_finish_wandb(force=False):
    if not _wandb_enabled(force=force):
        return
    if (not WANDB_AVAILABLE) or wandb.run is None:
        return
    wandb.finish()


def _setup_ours_ddp():
    """Setup torch.distributed for ours mode. Returns runtime metadata."""
    use_ddp = bool(getattr(args, 'ours_ddp', False))
    if not use_ddp:
        return {
            'enabled': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
        }

    if not torch.cuda.is_available():
        logger.warning('[ours-ddp] CUDA unavailable; fallback to single process.')
        return {
            'enabled': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
        }

    if ('RANK' not in os.environ) or ('WORLD_SIZE' not in os.environ):
        logger.warning('[ours-ddp] RANK/WORLD_SIZE not found. Use torchrun. Fallback to single process.')
        return {
            'enabled': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
        }

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank % max(1, torch.cuda.device_count())))

    torch.cuda.set_device(local_rank)
    backend = getattr(args, 'ours_ddp_backend', 'nccl')
    if not _dist_is_initialized():
        dist.init_process_group(backend=backend, init_method='env://')

    logger.info(
        f'[ours-ddp] initialized: backend={backend}, rank={rank}, '
        f'local_rank={local_rank}, world_size={world_size}'
    )
    return {
        'enabled': True,
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
    }


def _maybe_wrap_ddp(module, local_rank, find_unused_parameters, name):
    """Wrap a module with DDP only when it has trainable parameters."""
    has_trainable = any(p.requires_grad for p in module.parameters())
    if not has_trainable:
        logger.info(f'[ours-ddp] skip wrapping {name}: no trainable params')
        return module
    return _AttrDDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=bool(find_unused_parameters),
    )


def _parse_dp_devices(device_str):
    """Parse comma-separated CUDA ids to logical ids for this process.

    Notes:
      - DataParallel expects logical ids in [0, cuda.device_count()).
      - If users pass physical ids while CUDA_VISIBLE_DEVICES is set,
        we map physical -> logical when possible.
    """
    if not torch.cuda.is_available():
        return []
    n_vis = int(torch.cuda.device_count())
    if not device_str or str(device_str).strip() == '':
        return list(range(n_vis))

    req_ids = []
    for tok in str(device_str).split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            req_ids.append(int(tok))
        except Exception:
            logger.warning(f"[ours-dp] invalid device id token: {tok}")

    vis_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    phys2logical = {}
    vis_phys = []
    if vis_env.strip():
        vis_tokens = [t.strip() for t in vis_env.split(',') if t.strip()]
        for logical_id, tok in enumerate(vis_tokens):
            try:
                phys = int(tok)
                phys2logical[phys] = logical_id
                vis_phys.append(phys)
            except Exception:
                pass
    req_as_physical = bool(vis_phys) and all((did in phys2logical) for did in req_ids)

    out = []
    seen = set()
    for did in req_ids:
        logical_id = did
        if did < 0:
            logger.warning(f"[ours-dp] skip negative device id: {did}")
            continue
        if req_as_physical:
            logical_id = phys2logical[did]
        elif did in phys2logical and did >= n_vis:
            logical_id = phys2logical[did]
        if did >= n_vis:
            if did in phys2logical:
                logical_id = phys2logical[did]
            else:
                logger.warning(
                    f"[ours-dp] device id {did} is invalid for visible_gpu_count={n_vis}; skip."
                )
                continue
        if logical_id < 0 or logical_id >= n_vis:
            logger.warning(
                f"[ours-dp] mapped logical id {logical_id} invalid for visible_gpu_count={n_vis}; skip."
            )
            continue
        if logical_id not in seen:
            seen.add(logical_id)
            out.append(logical_id)
    return out




def _update_best_val(best_val, env_name, score_summary, state):
    """Store all metrics and update a split's best checkpoint by success rate."""
    metrics = {
        key: float(value)
        for key, value in score_summary.items()
        if key != 'data_count' and isinstance(value, (int, float, np.number))
    }
    entry = best_val.setdefault(env_name, {
        'metrics': {},
        'success_rate': float('-inf'),
        'state': '',
        'update': False,
    })
    entry['metrics'] = metrics
    entry['update'] = False
    success_rate = metrics.get('success_rate', float('-inf'))
    if success_rate > entry['success_rate']:
        entry['success_rate'] = success_rate
        entry['state'] = state
        entry['update'] = True
    return entry['update']


''' train the listener '''


def train(train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    record_file = open(os.path.join(log_dir, 'train_log.txt'), 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None:
        start_iter = listner.load(os.path.join(args.load))
        logger.info("LOAD the model from {}, iteration {}".format(args.load, start_iter))

    start = time.time()
    logger.info('Listener training starts, start iteration: %s' % str(start_iter))

    best_val = {}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        listner.env = train_env
        listner.train(interval, feedback=feedback_method)
        

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        REF_loss = sum(listner.logs['REF_loss']) / max(len(listner.logs['REF_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("loss/REF_loss", REF_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        logger.info("total_actions %d, max_length %d" % (total, length))

        # Run validation
        loss_str = "iter %d IL_loss %.2f RL_loss %.2f REF_loss %.2f critic_loss %.2f entropy %.2f" % (iter,
            IL_loss, RL_loss, REF_loss, critic_loss, entropy)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if isinstance(val, (int, float, np.number)):
                    writer.add_scalar("metrics/%s/%s" % (env_name, metric), val, idx)
                loss_str += ', %s: %.4f' % (metric, val)
            _update_best_val(
                best_val,
                env_name,
                score_summary,
                'Iter %d %s' % (iter, loss_str),
            )

        record_file = open(os.path.join(log_dir, 'train_log.txt'), 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        logger.info('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        if iter % 1000 == 0:
            logger.info("BEST RESULT TILL NOW")
            for env_name in best_val:
                logger.info("%s %s" % (env_name, best_val[env_name]['state']))

                record_file = open(os.path.join(log_dir, 'train_log.txt'), 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

def _sd_to_cpu(sd):
    """Detach + move to CPU.
    Floating tensors -> float32 (stable aggregation).
    Non-floating tensors (e.g., position_ids) keep dtype.
    """
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            v_cpu = v.detach().cpu()
            if v_cpu.is_floating_point():
                out[k] = v_cpu.to(dtype=torch.float32).clone()
            else:
                out[k] = v_cpu.clone()  # keep int/bool dtype
        else:
            out[k] = copy.deepcopy(v)
    return out

def _reset_optim(agent):
    """Clear optimizer states to avoid cross-client carry-over."""
    agent.vln_bert_optimizer.state.clear()
    agent.critic_optimizer.state.clear()


def _weighted_average_metrics(results):
    """Compute data_count-weighted average of metric dicts."""
    if not results:
        return None
    total = float(sum(r.get('data_count', 0) for r in results))
    if total <= 0:
        return None
    metric_names = set()
    for r in results:
        metric_names.update([k for k in r.keys() if k != 'data_count'])
    out = {}
    for k in metric_names:
        out[k] = sum(float(r.get(k, 0.0)) * float(r.get('data_count', 0)) for r in results) / total
    out['data_count'] = int(total)
    return out

# =========================
# Memory debugging helpers
# =========================
_MEM_DEBUG = os.environ.get('MEM_DEBUG', '0') == '1'
_MEM_DEBUG_EVERY = int(os.environ.get('MEM_DEBUG_EVERY', '1') or '1')
_MEM_DEBUG_CLIENT = os.environ.get('MEM_DEBUG_CLIENT', '0') == '1'
_MEM_TRACEMALLOC = os.environ.get('MEM_TRACEMALLOC', '0') == '1'

if _MEM_DEBUG and _MEM_TRACEMALLOC:
    try:
        tracemalloc.start(25)
    except Exception:
        pass


def _read_first_existing(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    return f.read().strip()
        except Exception:
            continue
    return None


def _cgroup_mem_bytes():
    """Return (current_bytes, max_bytes_or_None). Works for cgroup v2; best-effort."""
    cur_s = _read_first_existing([
        '/sys/fs/cgroup/memory.current',
    ])
    max_s = _read_first_existing([
        '/sys/fs/cgroup/memory.max',
    ])

    def _parse(v):
        if v is None:
            return None
        v = v.strip()
        if v == 'max':
            return None
        try:
            return int(v)
        except Exception:
            return None

    cur_b = _parse(cur_s)
    max_b = _parse(max_s)
    return cur_b, max_b


def _bytes_to_gib(x):
    if x is None:
        return None
    return float(x) / (1024.0 ** 3)


def _proc_rss_bytes():
    """Return process RSS in bytes (best-effort)."""
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    # Fallback: parse /proc/self/status
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    # VmRSS: <kB> kB
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _mem_snapshot(tag: str):
    """Log a compact memory snapshot. Safe to call frequently."""
    if not _MEM_DEBUG:
        return

    rss_b = _proc_rss_bytes()
    cg_cur_b, cg_max_b = _cgroup_mem_bytes()

    rss_g = _bytes_to_gib(rss_b)
    cg_cur_g = _bytes_to_gib(cg_cur_b)
    cg_max_g = _bytes_to_gib(cg_max_b)

    gpu_alloc_g = None
    gpu_resv_g = None
    if torch.cuda.is_available():
        try:
            gpu_alloc_g = torch.cuda.memory_allocated() / (1024.0 ** 3)
            gpu_resv_g = torch.cuda.memory_reserved() / (1024.0 ** 3)
        except Exception:
            pass

    msg = f"[MEM] {tag}"
    if rss_g is not None:
        msg += f" | RSS={rss_g:.2f}GiB"
    if cg_cur_g is not None:
        msg += f" | cgroup_cur={cg_cur_g:.2f}GiB"
    if cg_max_b is None and _read_first_existing(['/sys/fs/cgroup/memory.max']) == 'max':
        msg += " | cgroup_max=max"
    elif cg_max_g is not None:
        msg += f" | cgroup_max={cg_max_g:.2f}GiB"
    if gpu_alloc_g is not None:
        msg += f" | gpu_alloc={gpu_alloc_g:.2f}GiB"
    if gpu_resv_g is not None:
        msg += f" | gpu_resv={gpu_resv_g:.2f}GiB"

    logger.info(msg)

    if _MEM_TRACEMALLOC:
        try:
            snap = tracemalloc.take_snapshot()
            top = snap.statistics('lineno')[:10]
            logger.info("[PYMEM] top allocations:")
            for st in top:
                logger.info(f"[PYMEM] {st}")
        except Exception:
            pass



def _reset_gpu_peak_stats():
    """Reset CUDA peak stats for per-client memory monitoring."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _log_gpu_mem_client(tag: str):
    """Always log GPU memory after each client local training."""
    if not torch.cuda.is_available():
        logger.info(f"[GPU-MEM] {tag} | cuda=unavailable")
        return
    try:
        alloc_g = torch.cuda.memory_allocated() / (1024.0 ** 3)
        resv_g = torch.cuda.memory_reserved() / (1024.0 ** 3)
        peak_alloc_g = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        peak_resv_g = torch.cuda.max_memory_reserved() / (1024.0 ** 3)
        logger.info(
            f"[GPU-MEM] {tag} | "
            f"alloc={alloc_g:.2f}GiB, reserved={resv_g:.2f}GiB, "
            f"peak_alloc={peak_alloc_g:.2f}GiB, peak_reserved={peak_resv_g:.2f}GiB"
        )
    except Exception as e:
        logger.warning(f"[GPU-MEM] {tag} | read_failed: {e}")


def _eval_one_scan(agent, val_env_scan, scan_id, evaluator):
    """Evaluate one client's model on one scan within a scan-based env."""
    if scan_id not in getattr(val_env_scan, 'data', {}):
        return None
    if len(val_env_scan.data.get(scan_id, [])) == 0:
        return None
    val_env_scan.set_current_scan(scan_id)
    agent.logs = defaultdict(list)
    agent.env = val_env_scan
    agent.test(use_dropout=False, feedback='argmax', iters=None)
    result = agent.get_results()
    score_summary, _ = evaluator.score(result)
    score_summary['data_count'] = len(val_env_scan.data[scan_id])
    return score_summary


def _opt_state_to_cpu(opt_sd):
    """Deep-copy an optimizer state_dict with all tensors moved to CPU."""
    cpu_sd = {'state': {}, 'param_groups': copy.deepcopy(opt_sd['param_groups'])}
    for k, v in opt_sd['state'].items():
        cpu_sd['state'][k] = {
            sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv
            for sk, sv in v.items()
        }
    return cpu_sd


# ===================================================================
# Helpers for "ours" mode — prefix / backbone key classification
# ===================================================================

def _is_prefix_key(key):
    """Return True if *key* belongs to a prefix-related (local) parameter
    inside a ``PrefixVLNBERT`` state_dict.

    Prefix keys:
      • ``prefix_layers.*``  — KV-concat prefix in the wrapper
      • keys containing ``attn_prefix_`` — additive QKV adapters inside
        backbone attention layers

    Everything else (embeddings, encoder layers, poolers, action proj,
    LayerNorm, lang_last_adapter, …) is treated as backbone / shared.
    """
    return key.startswith('prefix_layers.') or 'attn_prefix_' in key


def _extract_prefix_state(vln_bert_sd):
    """Return only prefix-related entries from a PrefixVLNBERT state_dict."""
    return {k: v for k, v in vln_bert_sd.items() if _is_prefix_key(k)}


def _extract_backbone_state(vln_bert_sd):
    """Return only backbone (non-prefix) entries from a PrefixVLNBERT state_dict."""
    return {k: v for k, v in vln_bert_sd.items() if not _is_prefix_key(k)}


def _load_backbone_only(vln_bert_model, backbone_sd_cpu):
    """Load backbone parameters into a PrefixVLNBERT model without touching
    any prefix-related parameters.

    Uses ``strict=False`` so that prefix keys (absent from *backbone_sd_cpu*)
    are simply left at their current values.
    """
    filtered = {k: v for k, v in backbone_sd_cpu.items()
                if not _is_prefix_key(k)}
    vln_bert_model.load_state_dict(filtered, strict=False)


def _load_prefix_only(vln_bert_model, prefix_sd_cpu):
    """Load prefix parameters into a PrefixVLNBERT model without touching
    backbone parameters.
    """
    vln_bert_model.load_state_dict(prefix_sd_cpu, strict=False)




# ===================================================================
# train_ours — federated prefix personalization (fl_mode='ours')
# ===================================================================

def train_ours(train_env, tok, n_iters, log_every=10, val_env=None,
               val_split_name='val_seen'):
    
    from agent import PrefixSeq2SeqAgent

    # REMOVE: DDP and DataParallel are infrastructure variants, not part of the
    # retained algorithm. Ours now runs its canonical single-process path.
    ddp_enabled = False
    ddp_rank = 0
    ddp_world = 1
    ddp_local_rank = 0
    writer = SummaryWriter(log_dir=log_dir)
    # REMOVE: W&B is optional experiment review/logging, not a training dependency.
    # _maybe_init_wandb(force=True)

    

    # ---- Client / scan setup ------------------------------------------------
    scans_list = list(getattr(train_env, 'scans_list', []))
    if args.n_parties is not None:
        n_parties = min(args.n_parties, len(scans_list))
        scans_list = scans_list[:n_parties]
    else:
        n_parties = len(scans_list)
    if n_parties == 0:
        raise ValueError('No scans found in train_env for ours mode')
    if args.disk_n_parties is None:
        disk_n_parties = n_parties
    else:
        disk_n_parties = max(0, min(args.disk_n_parties, n_parties))
    disk_scans_list = list(scans_list[:disk_n_parties])
    memory_scans_list = list(scans_list[disk_n_parties:])
    disk_scan_set = set(disk_scans_list)

    party_list = list(range(n_parties))
    n_party_per_round = max(1, int(n_parties * args.sample_fraction))

    # ---- Shared agent container (PrefixSeq2SeqAgent on GPU) -----------------
    #  freeze_backbone=False → backbone + prefix + gate + critic ALL trainable
    agent = PrefixSeq2SeqAgent(
        train_env, "", tok, args.maxAction,
        prefix_len=getattr(args, 'prefix_len', 8),
        prefix_modules=getattr(args, 'prefix_modules', 'infer'),
        gate_hidden=getattr(args, 'gate_hidden', 256),
        freeze_backbone=False,
    )

    # Optional pretrained backbone initialization is retained through --load.
    if args.load is not None:
        logger.info(f'[ours] Loading backbone from: {args.load}')
        agent.load_backbone_from_ckpt(args.load)

    

    # ---- Global state on CPU ------------------------------------------------
    full_vln_sd         = _sd_to_cpu(agent.vln_bert.state_dict())
    global_backbone_cpu = _extract_backbone_state(full_vln_sd)
    init_prefix_cpu     = _extract_prefix_state(full_vln_sd)
    del full_vln_sd

    # ---- Per-client local state on disk -------------------------------------
    # Each client persists: prefix, gate, critic (+ all optimizer states).
    # Backbone weights are NOT persisted locally.
    CLIENT_STATE_DISK_KEYS = (
        'prefix',
        'gate',
        'critic',
        'opt_vln_bert',
        'opt_gate',
        'opt_critic',
    )
    client_state_dir = os.path.join('snap', 'REVERIE_client_state')
    client_state_partition = {
        'run_name': args.name,
        'n_parties': n_parties,
        'disk_n_parties': disk_n_parties,
        'memory_n_parties': len(memory_scans_list),
        'all_scans': list(scans_list),
        'disk_scans': list(disk_scans_list),
        'memory_scans': list(memory_scans_list),
        'disk_state_dir': client_state_dir,
    }
    client_state_partition_paths = [
        os.path.join(client_state_dir, 'scan_partition.json'),
        os.path.join(args.log_dir, 'scan_partition.json'),
    ]
    client_state_paths = {
        scan: os.path.join(
            client_state_dir,
            f"{re.sub(r'[^A-Za-z0-9_.-]', '_', scan)}.pt",
        )
        for scan in disk_scans_list
    }
    client_state_memory = {}

    def _client_state_uses_disk(scan_id):
        return scan_id in disk_scan_set

    def _write_client_state_partition():
        for manifest_path in client_state_partition_paths:
            manifest_dir = os.path.dirname(manifest_path)
            if manifest_dir:
                os.makedirs(manifest_dir, exist_ok=True)
            with open(manifest_path, 'w') as manifest_file:
                json.dump(client_state_partition, manifest_file,
                          sort_keys=True, indent=2)

    def _cleanup_stale_client_states():
        if not os.path.isdir(client_state_dir):
            return
        keep_paths = set(os.path.abspath(p) for p in client_state_paths.values())
        for entry in os.listdir(client_state_dir):
            path = os.path.join(client_state_dir, entry)
            if (not os.path.isfile(path)) or (not entry.endswith('.pt')):
                continue
            if os.path.abspath(path) in keep_paths:
                continue
            try:
                os.remove(path)
                logger.info(f'[ours] removed stale client cache: {path}')
            except Exception as exc:
                logger.warning(f'[ours] failed to remove stale client cache {path}: {exc}')

    def _save_client_state(scan_id, state):
        payload = {
            key: state[key]
            for key in CLIENT_STATE_DISK_KEYS
            if key in state
        }
        if not _client_state_uses_disk(scan_id):
            client_state_memory[scan_id] = copy.deepcopy(payload)
            return
        os.makedirs(client_state_dir, exist_ok=True)
        path = client_state_paths[scan_id]
        tmp_path = f'{path}.tmp'
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    init_gate_cpu   = _sd_to_cpu(agent.gate_policy.state_dict())
    init_critic_cpu = _sd_to_cpu(agent.critic.state_dict())
    init_client_state_cpu = {
        'prefix': init_prefix_cpu,
        'gate': init_gate_cpu,
        'critic': init_critic_cpu,
    }
    client_state_memory = {
        scan: copy.deepcopy(init_client_state_cpu)
        for scan in memory_scans_list
    }

    def _load_client_state(scan_id):
        if not _client_state_uses_disk(scan_id):
            if scan_id not in client_state_memory:
                return copy.deepcopy(init_client_state_cpu)
            return copy.deepcopy(client_state_memory[scan_id])
        path = client_state_paths[scan_id]
        if not os.path.exists(path):
            return copy.deepcopy(init_client_state_cpu)
        payload = torch.load(path, map_location='cpu')
        return {
            key: payload[key]
            for key in CLIENT_STATE_DISK_KEYS
            if key in payload
        }

    _cleanup_stale_client_states()
    _write_client_state_partition()
    for scan in disk_scans_list:
        _save_client_state(scan, copy.deepcopy(init_client_state_cpu))

    # ---- Communication-round calculation ------------------------------------
    total_data = train_env.size()
    if args.comm_round is not None:
        comm_round = args.comm_round
    else:
        comm_round = int(math.ceil(
            n_iters / max(1.0, total_data / args.batchSize)))

    total_data_points = max(1, sum(len(train_env.data[s]) for s in scans_list))
    fed_avg_freqs = [len(train_env.data[scans_list[k]]) / total_data_points
                     for k in range(n_parties)]

    

    start = time.time()
    iter_total = 0
    best_val = {}

    # Per-scan evaluators
    evaluators_by_scan = {}
    if val_env is not None:
        for scan_id in scans_list:
            if scan_id in getattr(val_env, 'data', {}) and \
               len(val_env.data.get(scan_id, [])) > 0:
                evaluators_by_scan[scan_id] = Evaluation(
                    [val_split_name], {scan_id}, tok)

    # ==================================================================
    # Main communication-round loop
    # ==================================================================
    for round_idx in range(comm_round):
        # ---- Client sampling ----
        if n_party_per_round < n_parties:
            party_list_this_round = random.sample(party_list, n_party_per_round)
        else:
            party_list_this_round = list(party_list)

        client_sizes = [len(train_env.data[scans_list[k]])
                        for k in party_list_this_round]
        total_size = float(max(1, sum(client_sizes)))
        freq_this_round = [c / total_size for c in client_sizes]

        # Snapshot global backbone for delta aggregation
        old_global_backbone_cpu = copy.deepcopy(global_backbone_cpu)
        new_global_backbone_cpu = copy.deepcopy(global_backbone_cpu)

        # Per-round loss accumulators (across all clients in this round)
        round_IL_losses = []
        round_RL_losses = []
        round_REF_losses = []
        round_critic_losses = []
        round_gate_means = []
        round_gate_smooth_losses = []
        round_gate_block_means = defaultdict(list)   # {block_idx: [values]}

        for cid, k in enumerate(party_list_this_round):
            scan_id = scans_list[k]
            train_env.set_current_scan(scan_id)
            client_data_count = len(train_env.data[scan_id])
            num_step = max(1, int(args.local_epoches * client_data_count
                                  / args.batchSize))
            iter_total += num_step

            logger.info(
                f'  [Round {round_idx}] Client {cid+1}/'
                f'{len(party_list_this_round)}: scan={scan_id}, '
                f'data={client_data_count}, steps={num_step}')
            

            # 1. Load global backbone (server download) ─────────────────
            _load_backbone_only(agent.vln_bert, global_backbone_cpu)

            # 2. Load local prefix (personal, NOT aggregated) ──────────
            cs = _load_client_state(scan_id)
            _load_prefix_only(agent.vln_bert, cs['prefix'])

            # 3. Restore local gate + critic ────────────────────────────
            agent.gate_policy.load_state_dict(cs['gate'])
            agent.critic.load_state_dict(cs['critic'])

            # 4. Restore ALL optimizer states (accumulate across rounds)
            #    opt_vln_bert covers BOTH backbone+prefix Adam moments
            if 'opt_vln_bert' in cs:
                agent.vln_bert_optimizer.load_state_dict(cs['opt_vln_bert'])
            else:
                agent.vln_bert_optimizer.state.clear()
            if 'opt_gate' in cs:
                agent.gate_optimizer.load_state_dict(cs['opt_gate'])
            else:
                agent.gate_optimizer.state.clear()
            if 'opt_critic' in cs:
                agent.critic_optimizer.load_state_dict(cs['opt_critic'])
            else:
                agent.critic_optimizer.state.clear()

            # 5. Local training (backbone NOT frozen — all train jointly)
            agent.env = train_env
            agent.logs = defaultdict(list)
            agent.train(num_step, feedback=feedback_method)

            # Collect per-client losses from agent.logs
            if agent.logs.get('IL_loss'):
                avg_il = sum(agent.logs['IL_loss']) / max(1, len(agent.logs['IL_loss']))
                round_IL_losses.append(avg_il)
            if agent.logs.get('RL_loss'):
                avg_rl = sum(agent.logs['RL_loss']) / max(1, len(agent.logs['RL_loss']))
                round_RL_losses.append(avg_rl)
            if agent.logs.get('REF_loss'):
                avg_ref = sum(agent.logs['REF_loss']) / max(1, len(agent.logs['REF_loss']))
                round_REF_losses.append(avg_ref)
            if agent.logs.get('critic_loss'):
                avg_cri = sum(agent.logs['critic_loss']) / max(1, len(agent.logs['critic_loss']))
                round_critic_losses.append(avg_cri)
            if agent.logs.get('gate_mean'):
                avg_gate = sum(agent.logs['gate_mean']) / max(1, len(agent.logs['gate_mean']))
                round_gate_means.append(avg_gate)
            if agent.logs.get('gate_smooth_loss'):
                avg_gs = sum(agent.logs['gate_smooth_loss']) / max(1, len(agent.logs['gate_smooth_loss']))
                round_gate_smooth_losses.append(avg_gs)
            for bkey in list(agent.logs.keys()):
                if bkey.startswith('gate_block_'):
                    bi = bkey  # e.g. 'gate_block_0'
                    round_gate_block_means[bi].append(
                        sum(agent.logs[bkey]) / max(1, len(agent.logs[bkey])))

            # 6. Extract trained backbone → aggregate (upload) ──────────
            trained_vln_sd = _sd_to_cpu(agent.vln_bert.state_dict())
            trained_backbone_cpu = _extract_backbone_state(trained_vln_sd)
            trained_prefix_cpu   = _extract_prefix_state(trained_vln_sd)
            del trained_vln_sd

            w = float(freq_this_round[cid])
            with torch.no_grad():
                for key in new_global_backbone_cpu:
                    t = new_global_backbone_cpu[key]
                    if torch.is_tensor(t) and t.is_floating_point():
                        new_global_backbone_cpu[key].add_(
                            trained_backbone_cpu[key]
                            - old_global_backbone_cpu[key],
                            alpha=args.global_lr * w)

            # 7. Save client local state ────────────────────────────────
            #    NO backbone weights; YES all optimizer states
            #    (opt_vln_bert includes backbone+prefix Adam moments)
            _save_client_state(scan_id, {
                'prefix':       trained_prefix_cpu,
                'gate':         _sd_to_cpu(agent.gate_policy.state_dict()),
                'critic':       _sd_to_cpu(agent.critic.state_dict()),
                'opt_vln_bert': _opt_state_to_cpu(
                                    agent.vln_bert_optimizer.state_dict()),
                'opt_gate':     _opt_state_to_cpu(
                                    agent.gate_optimizer.state_dict()),
                'opt_critic':   _opt_state_to_cpu(
                                    agent.critic_optimizer.state_dict()),
            })
            del cs, trained_backbone_cpu, trained_prefix_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # REMOVE: per-client GPU memory logging is review-only.

        # ---- Update global backbone ----
        global_backbone_cpu = new_global_backbone_cpu

        # ---- Per-round loss summary ----
        avg_IL  = float(np.mean(round_IL_losses))   if round_IL_losses   else 0.0
        avg_RL  = float(np.mean(round_RL_losses))   if round_RL_losses   else 0.0
        avg_REF = float(np.mean(round_REF_losses))  if round_REF_losses  else 0.0
        avg_cri = float(np.mean(round_critic_losses)) if round_critic_losses else 0.0
        avg_gate = float(np.mean(round_gate_means)) if round_gate_means  else 0.0
        avg_gate_smooth = float(np.mean(round_gate_smooth_losses)) if round_gate_smooth_losses else 0.0
        avg_total = avg_IL + avg_RL + avg_REF

        loss_msg = (f'[ours Rd {round_idx}] '
                    f'IL={avg_IL:.4f}, RL={avg_RL:.4f}, REF={avg_REF:.4f}, '
                    f'total={avg_total:.4f} | '
                    f'critic={avg_cri:.4f}, gate_mean={avg_gate:.4f}, '
                    f'gate_smooth={avg_gate_smooth:.6f}')
        # Per-block gate means
        gate_block_str = ''
        for bkey in sorted(round_gate_block_means.keys()):
            bval = float(np.mean(round_gate_block_means[bkey]))
            gate_block_str += f', {bkey}={bval:.4f}'
        if gate_block_str:
            loss_msg += gate_block_str
        logger.info(loss_msg)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('loss/IL_loss', avg_IL, round_idx)
            writer.add_scalar('loss/RL_loss', avg_RL, round_idx)
            writer.add_scalar('loss/REF_loss', avg_REF, round_idx)
            writer.add_scalar('loss/total_loss', avg_total, round_idx)
            writer.add_scalar('loss/critic_loss', avg_cri, round_idx)
            writer.add_scalar('gate/mean', avg_gate, round_idx)
            writer.add_scalar('gate/smooth_loss', avg_gate_smooth, round_idx)
            for bkey in sorted(round_gate_block_means.keys()):
                bval = float(np.mean(round_gate_block_means[bkey]))
                writer.add_scalar(f'gate/{bkey}', bval, round_idx)

        

        # ==============================================================
        # Evaluation: each client's personal model on its own val scan
        # ==============================================================
        if (val_env is not None) and (round_idx % log_every == 0 or round_idx > 200):
            # REMOVE: DDP barriers and memory snapshots belong to excluded
            # distributed/profiling workflows.
            if True:
                _rng_py  = random.getstate()
                _rng_np  = np.random.get_state()
                _rng_cpu = torch.random.get_rng_state()
                _rng_gpu = (torch.cuda.get_rng_state()
                            if torch.cuda.is_available() else None)

                per_scan_scores = {}
                for scan_id in scans_list:
                    evaluator = evaluators_by_scan.get(scan_id)
                    if evaluator is None:
                        continue
                    if scan_id not in val_env.data or \
                       len(val_env.data.get(scan_id, [])) == 0:
                        continue

                    # Assemble full model for this client's eval:
                    #   backbone from global + prefix/gate/critic from local
                    _load_backbone_only(agent.vln_bert, global_backbone_cpu)
                    cs = _load_client_state(scan_id)
                    if 'prefix' in cs:
                        _load_prefix_only(agent.vln_bert, cs['prefix'])
                    if 'gate' in cs:
                        agent.gate_policy.load_state_dict(cs['gate'])
                    if 'critic' in cs:
                        agent.critic.load_state_dict(cs['critic'])

                    score = _eval_one_scan(agent, val_env, scan_id, evaluator)
                    if score is not None:
                        per_scan_scores[scan_id] = score
                        scan_str = f'  scan={scan_id}'
                        for mk in ['success_rate', 'spl', 'rgs', 'rgspl',
                                    'oracle_rate']:
                            if mk in score:
                                scan_str += f', {mk}: {score[mk]:.3f}'
                        logger.info(scan_str)

                if per_scan_scores:
                    metric_names = set()
                    for s in per_scan_scores.values():
                        metric_names.update(kk for kk in s if kk != 'data_count')
                    metric_avg = {}
                    for kk in metric_names:
                        metric_avg[kk] = np.mean(
                            [float(s.get(kk, 0.0))
                             for s in per_scan_scores.values()])

                    loss_str = (f'round {round_idx} {val_split_name} '
                                f'(ours, {len(per_scan_scores)} scans)')
                    for kk in ['success_rate', 'spl', 'rgs', 'rgspl',
                                'oracle_rate']:
                        if kk in metric_avg:
                            loss_str += f', {kk}: {metric_avg[kk]:.3f}'
                    logger.info(
                        f'{timeSince(start, float(round_idx+1)/comm_round)} '
                        f'({round_idx+1} '
                        f'{float(round_idx+1)/comm_round*100:.1f}%) {loss_str}')

                    # REMOVE: W&B metric export is experiment-review logging.

                    if _update_best_val(
                        best_val,
                        val_split_name,
                        metric_avg,
                        f'ours Rd {round_idx}, {loss_str}',
                    ):
                        logger.info(
                            f'  [new best] success_rate='
                            f'{best_val[val_split_name]["success_rate"]:.4f}')

                random.setstate(_rng_py)
                np.random.set_state(_rng_np)
                torch.random.set_rng_state(_rng_cpu)
                if _rng_gpu is not None:
                    torch.cuda.set_rng_state(_rng_gpu)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---- Periodic global checkpoint (backbone only) ----
        if round_idx % 100 == 0 and round_idx > 0:
            ckpt_path = os.path.join(
                "snap", args.name, "state_dict",
                f"ours_round_{round_idx}")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'global_backbone': global_backbone_cpu,
                'round':           round_idx,
            }, ckpt_path)
            logger.info(f'  [ours] Checkpoint saved: {ckpt_path}')

    # ---- Final save ----
    final_round = round_idx if comm_round > 0 else 0
    if True:
        ckpt_path = os.path.join(
            "snap", args.name, "state_dict",
            f"ours_LAST_round{final_round}")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            'global_backbone': global_backbone_cpu,
            'round':           final_round,
        }, ckpt_path)
        logger.info(f'[ours] Training finished! Final ckpt: {ckpt_path}')
        for env_name in best_val:
            logger.info(f"Best {env_name}: {best_val[env_name]['state']}")
        # REMOVE: final W&B reporting belongs to experiment-review logging.


def train_fedavg(train_env, tok, n_iters, log_every=10, val_envs={}, aug_env=None):
    """
    Federated Averaging training for REVERIE.
    Each scan is treated as a separate client.
    """
    writer = SummaryWriter(log_dir=log_dir)
    
    # Setup client list from scans
    scans_list = list(train_env.scans_list)
    if args.n_parties is not None:
        n_parties = min(args.n_parties, len(scans_list))
        scans_list = scans_list[:n_parties]
    else:
        n_parties = len(scans_list)
    
    party_list = list(range(n_parties))
    n_party_per_round = max(1, int(n_parties * args.sample_fraction))
    
    # Global agent
    listner_global = Seq2SeqAgent(train_env, "", tok, args.maxAction) #
    listner_local  = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    
    # Calculate communication rounds
    total_data = train_env.size()
    if args.comm_round is not None:
        comm_round = args.comm_round
    else:
        comm_round = int(math.ceil(n_iters / (total_data / args.batchSize)))
    
    # Calculate data frequency for weighted averaging
    total_data_points = sum(len(train_env.data[s]) for s in scans_list)
    fed_avg_freqs = [len(train_env.data[scans_list[k]]) / total_data_points for k in range(n_parties)]
    
    logger.info(f'FedAvg Training: {n_parties} clients, sample={n_party_per_round}/round, '
                f'local_epoches={args.local_epoches}, comm_rounds={comm_round}')
    
    start = time.time()
    iter_total = 0
    start_iter = 0
    if args.load is not None:
        start_iter = listner_global.load(os.path.join(args.load))
        logger.info("FedAvg init global model from {}, iteration {}".format(args.load, start_iter))
  
    best_val = {}
    
    record_file = open(os.path.join(log_dir, 'train_log.txt'), 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()
    
    for round_idx in range(comm_round):
        #listner_global.logs = defaultdict(list) #
        
        # Sample clients for this round
        if n_party_per_round < n_parties:
            party_list_this_round = random.sample(party_list, n_party_per_round)
        else:
            party_list_this_round = list(party_list)
        
        # Calculate frequency for this round
        total_freq_round = sum(fed_avg_freqs[k] for k in party_list_this_round)
        freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]
        
        # Store global weights
        # global_vln_bert_w = copy.deepcopy(listner_global.vln_bert.state_dict())
        # global_critic_w = copy.deepcopy(listner_global.critic.state_dict())
        global_vln_cpu = _sd_to_cpu(listner_global.vln_bert.state_dict())
        global_cri_cpu = _sd_to_cpu(listner_global.critic.state_dict())

        
        # Initialize new global weights for aggregation
        # new_vln_bert_w = copy.deepcopy(global_vln_bert_w)
        # new_critic_w = copy.deepcopy(global_critic_w)
        new_vln_cpu = copy.deepcopy(global_vln_cpu)
        new_cri_cpu = copy.deepcopy(global_cri_cpu)
        
        train_loss_list = []
        
        for cid, k in enumerate(party_list_this_round):
            scan_id = scans_list[k]
            train_env.set_current_scan(scan_id)
            client_data_count = len(train_env.data[scan_id])
            num_step = max(1, int(args.local_epoches * client_data_count / args.batchSize))
            iter_total += num_step
            
            logger.info(f'  [Round {round_idx}] Client {cid+1}/{len(party_list_this_round)}: '
                       f'scan={scan_id}, data={client_data_count}, steps={num_step}')
            
            # Copy global model to local
            
            
            # Local training
            # listner_local = Seq2SeqAgent(train_env, "", tok, args.maxAction)
            listner_local.vln_bert.load_state_dict(global_vln_cpu, strict=True)
            listner_local.critic.load_state_dict(global_cri_cpu, strict=True)
            _reset_optim(listner_local)
            listner_local.env = train_env
            listner_local.logs = defaultdict(list)
            listner_local.train(num_step, feedback=feedback_method)
            
            # Collect local model weights
            # local_vln_bert_w = listner_local.vln_bert.state_dict()
            # local_critic_w = listner_local.critic.state_dict()
            local_vln_cpu = _sd_to_cpu(listner_local.vln_bert.state_dict())
            local_cri_cpu = _sd_to_cpu(listner_local.critic.state_dict())
            
            # Weighted aggregation
            w = float(freq_this_round[cid])
            glr = float(args.global_lr)
            with torch.no_grad():
                # for key in new_vln_bert_w:
                #     delta = (local_vln_bert_w[key] - global_vln_bert_w[key]) * freq_this_round[cid]
                #     new_vln_bert_w[key] = new_vln_bert_w[key].float() + args.global_lr * delta.float()
                # for key in new_critic_w:
                #     delta = (local_critic_w[key] - global_critic_w[key]) * freq_this_round[cid]
                #     new_critic_w[key] = new_critic_w[key].float() + args.global_lr * delta.float()
                for key in new_vln_cpu:
                    if torch.is_tensor(new_vln_cpu[key]) and new_vln_cpu[key].is_floating_point():
                        new_vln_cpu[key].add_(local_vln_cpu[key] - global_vln_cpu[key], alpha=args.global_lr * w)
                    # if torch.is_tensor(new_vln_cpu[key]):
                    #     new_vln_cpu[key].add_(local_vln_cpu[key] - global_vln_cpu[key], alpha=args.global_lr * w)

                for key in new_cri_cpu:
                    if torch.is_tensor(new_cri_cpu[key]) and new_cri_cpu[key].is_floating_point():
                        new_cri_cpu[key].add_(local_cri_cpu[key] - global_cri_cpu[key], alpha=args.global_lr * w)
            
            # Track loss
            # if listner_global.logs['IL_loss']:
            #     train_loss_list.append(sum(listner_local.logs['IL_loss']) / len(listner_local.logs['IL_loss']))
            if listner_local.logs.get('IL_loss'):
                train_loss_list.append(
                    sum(listner_local.logs['IL_loss']) / max(1, len(listner_local.logs['IL_loss']))
                )
            
        # Update global model
        # listner_global.vln_bert.load_state_dict(new_vln_bert_w)
        # listner_global.critic.load_state_dict(new_critic_w)
        listner_global.vln_bert.load_state_dict(new_vln_cpu, strict=True)
        listner_global.critic.load_state_dict(new_cri_cpu, strict=True)

        
        avg_loss = np.mean(train_loss_list) if train_loss_list else 0
        logger.info(f'Round {round_idx} avg loss: {avg_loss:.4f}')
        logger.info(f'{timeSince(start, float(round_idx+1)/comm_round)} '
                    f'{(iter_total, float(iter_total)*args.batchSize/n_iters/4)}')
        
        # Validation every log_every rounds
        print(f'{log_every}')
        if (round_idx) % log_every == 0:
            loss_str = f"round {round_idx}, iter {iter_total}, IL_loss {avg_loss:.4f}"
            
            for env_name, (env, evaluator) in val_envs.items():
                listner_global.env = env
                listner_global.test(use_dropout=False, feedback='argmax', iters=None)
                result = listner_global.get_results()
                score_summary, _ = evaluator.score(result)
                
                loss_str += f", {env_name}"
                for metric, val in score_summary.items():
                    if isinstance(val, (int, float, np.number)):
                        writer.add_scalar(f"metrics/{env_name}/{metric}", val, round_idx)
                    loss_str += f', {metric}: {val:.4f}'
                _update_best_val(
                    best_val,
                    env_name,
                    score_summary,
                    f'Round {round_idx} {loss_str}',
                )
            
            logger.info(f'{timeSince(start, float(round_idx+1)/comm_round)} '
                       f'({round_idx+1} {float(round_idx+1)/comm_round*100:.1f}%) {loss_str}')
            
            record_file = open(os.path.join(log_dir, 'train_log.txt'), 'a')
            record_file.write(loss_str + '\n')
            record_file.close()
            
            # Save best model
            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['update'] = False
                    listner_global.save(round_idx, os.path.join("snap", args.name, "state_dict", f"best_{env_name}"))
        
        # Periodic save
        if (round_idx+1) % 100 == 0 and round_idx > 0:
            listner_global.save(round_idx, os.path.join("snap", args.name, "state_dict", f"round_{round_idx}"))
    
    listner_global.save(round_idx, os.path.join("snap", args.name, "state_dict", f"LAST_round{round_idx}_{(timestamp)}"))
    logger.info("Training finished!")
    for env_name in best_val:
        logger.info(f"Best {env_name}: {best_val[env_name]['state']}")

# def state_dict_to_cpu(sd):
#     return {k: v.detach().cpu().clone() for k, v in sd.items()}

# def zeros_like_state_dict(sd_cpu):
#     return {k: torch.zeros_like(v) for k, v in sd_cpu.items()}

# @torch.no_grad()
# def apply_cpu_delta_to_model(model, delta_cpu, scale=1.0):
#     # delta_cpu 是 CPU tensor dict
#     sd = model.state_dict()
#     for k in sd:
#         sd[k].add_(delta_cpu[k].to(sd[k].device), alpha=scale)
#     model.load_state_dict(sd, strict=True)



def setup():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    feat_dict = read_img_features(features, test_only=test_only)
    # load object feature
    with open('img_features/REVERIE_obj_feats.pkl', 'rb') as f_obj:
        obj_feats = pkl.load(f_obj)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    # Ours evaluates each personalized client on its own validation scan.
    if args.fl_mode == 'ours':
        val_env_names = ['val_seen']
    else:
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    # Federated modes use scan-local client environments.
    if args.fl_mode in ['fedavg', 'ours']:
        train_env = R2RBatchScan(feat_dict, obj_feats, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
        logger.info(f'data size: {train_env.size()}')
    else:
        train_env = R2RBatch(feat_dict, obj_feats, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
        print(f'data size:{train_env.size()}')
    from collections import OrderedDict

   

    # Validation envs
    if args.fl_mode == 'ours':
        # Scan-based validation env for per-client evaluation
        val_envs = OrderedDict(
            ((split,
              (R2RBatchScan(feat_dict, obj_feats, batch_size=args.batchSize, splits=[split], tokenizer=tok),
               None)
              )
             for split in val_env_names)
        )
        val_envs = {k: v for k, v in val_envs.items() if len(v[0].scans_list) > 0}
    else:
        val_envs = OrderedDict(
            ((split,
              (R2RBatch(feat_dict, obj_feats, batch_size=args.batchSize, splits=[split], tokenizer=tok),
               Evaluation([split], featurized_scans, tok))
              )
             for split in val_env_names)
        )
        val_envs = {key: value for key, value in val_envs.items() if len(value[0].data) > 0}

    if args.fl_mode == 'fedavg':
        train_fedavg(train_env, tok, args.iters, log_every=args.log_every, val_envs=val_envs)
    elif args.fl_mode == 'ours':
        val_env = val_envs.get('val_seen', (None, None))[0]
        train_ours(train_env, tok, args.iters, log_every=args.log_every,
                   val_env=val_env, val_split_name='val_seen')
    else:
        # Centralized listener training.
        train(train_env, tok, args.iters, log_every=args.log_every, val_envs=val_envs)




if __name__ == "__main__":
    train_val()

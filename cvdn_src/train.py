import argparse
import json

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch, R2RBatchScan
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent
from eval import Evaluation
import logging
from datetime import datetime
import math
import random
import copy
# TRAIN_VOCAB = 'tasks/NDH/data/train_vocab.txt'
# TRAINVAL_VOCAB = 'tasks/NDH/data/trainval_vocab.txt'
# RESULT_DIR = 'tasks/NDH/results/'
# SNAPSHOT_DIR = 'snap/%s' % args.name
# PLOT_DIR = 'tasks/NDH/plots/'

# IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'

# # Training settings.
# agent_type = 'seq2seq'

# # Fixed params from MP.
# features = IMAGENET_FEATURES
# batch_size = 100
# word_embedding_size = 256
# action_embedding_size = 32
# target_embedding_size = 32
# hidden_size = 512
# bidirectional = False
# dropout_ratio = 0.5
# learning_rate = 0.0001
# weight_decay = 0.0005




def train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
    log_every=50, val_envs=None, fl_mode=None, args=None, _vocab_size=None, enc_hidden_size=None):
    ''' Train on training set, validating on both seen and unseen. '''
    log_every = args.log_every
    if val_envs is None:
        val_envs = {}

    if agent_type == 'seq2seq':
        agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
        # print(f'scans:{agent.scans}')
    else:
        sys.exit("Unrecognized agent_type '%s'" % agent_type)
    print('Training a %s agent with %s feedback' % (agent_type, feedback_method))
    if fl_mode == 'c':
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay) 

        data_log = defaultdict(list)
        start = time.time()

        for idx in range(0, n_iters, log_every):

            interval = min(log_every,n_iters-idx)
            iter = idx + interval
            data_log['iteration'].append(iter)

            # Train for log_every interval
            agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)
            data_log['train loss'].append(train_loss_avg)
            loss_str = 'train loss: %.4f' % train_loss_avg

            # Run validation
            for env_name, (env, evaluator) in val_envs.items():
                agent.env = env
                agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
                # Get validation loss under the same conditions as training
                agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
                val_losses = np.array(agent.losses)
                val_loss_avg = np.average(val_losses)
                data_log['%s loss' % env_name].append(val_loss_avg)
                # Get validation distance from goal under test evaluation conditions
                agent.test(use_dropout=False, feedback='argmax')
                agent.write_results()
                score_summary, _ = evaluator.score(agent.results_path)
                loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
                for metric, val in score_summary.items():
                    data_log['%s %s' % (env_name, metric)].append(val)
                    if metric in ['success_rate', 'oracle success_rate', 'oracle path_success_rate', 'dist_to_end_reduction']:
                        loss_str += ', %s: %.3f' % (metric, val)

            agent.env = train_env

            print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                                iter, float(iter)/n_iters*100, loss_str))
            df = pd.DataFrame(data_log)
            df.set_index('iteration')
            df_path = '%s%s-log.csv' % (PLOT_DIR, model_prefix)
            df.to_csv(df_path)
            
            split_string = "-".join(train_env.splits)
            enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
            dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
            agent.save(enc_path, dec_path)

    elif fl_mode == 'fedavg':
        if not hasattr(train_env, 'scans_list') or not hasattr(train_env, 'set_current_scan'):
            raise AttributeError('train_env must support scans_list/set_current_scan; please patch cvdn_env.py')
        # Setup client list
        scans_list = list(train_env.scans_list)
        if args.n_parties is None:
            n_parties = len(scans_list)
        else:
            n_parties = min(args.n_parties, len(scans_list))
            scans_list = scans_list[:n_parties]

        party_list = [i for i in range(n_parties)]
        n_party_per_round = max(1, int(n_parties * args.sample_fraction))

        agent_global = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
        encoder_global = copy.deepcopy(encoder)
        decoder_global = copy.deepcopy(decoder)
        print('Federated training (FedAvg): %d clients, sample=%d/round, local_epoches=%.3f' % (
            n_parties, n_party_per_round, args.local_epoches))

        data_log = defaultdict(list)
        start = time.time()
        iter_steps = 0  # local optimization steps processed so far
        round_id = 0
        best_val = {'val_seen': {'spl': 0.0, 'success_rate': 0.0, 'nav_error': 0.0,
                               'oracle success_rate': 0.0, 'oracle path_success_rate': 0.0,
                               'dist_to_end_reduction': 0.0,
         'state': '', 'update': False}}
        start_iter = 0
        # FedAvg and ours share --rounds; keeping this configurable prevents
        # the FedAvg script from silently using a different training budget.
        comm_round = args.rounds
        logger.info(f'The totol round is : {comm_round}')
        iter = start_iter

        # Per-scan evaluators on val_seen (aligned with pFL eval style).
        val_env = None
        if val_envs and 'val_seen' in val_envs:
            val_env = val_envs['val_seen'][0]
        evaluators_by_scan = {}
        if val_env is not None and hasattr(val_env, 'data'):
            for scan_id in scans_list:
                if scan_id in val_env.data and len(val_env.data.get(scan_id, [])) > 0:
                    evaluators_by_scan[scan_id] = Evaluation(
                        ['val_seen'], path_type=path_type, scans={scan_id}
                    )

        for round_idx in range(comm_round):
            print(f'Round:{round_idx}')
            if n_party_per_round < n_parties:
                party_list_this_round = random.sample(party_list, n_party_per_round)
            else:
                party_list_this_round = list(party_list)
            
            client_sizes = []
            for k in party_list_this_round:
                client_sizes.append(len(train_env.data[scans_list[k]]))
            total_size = float(sum(client_sizes)) if sum(client_sizes) > 0 else 1.0
            freq_this_round = [c / total_size for c in client_sizes]
            new_encoder_global = copy.deepcopy(agent_global.encoder.state_dict())
            new_decoder_global = copy.deepcopy(agent_global.decoder.state_dict())
            train_loss_list = []

            for cid, k in enumerate(party_list_this_round):
                scan_id = scans_list[k]
                train_env.set_current_scan(scan_id)
                client_count = len(train_env.data[scan_id])
                num_step = int(args.local_epoches * client_count / args.bs)
                logger.info('  [Round %d] client %d/%d: scan=%s, data=%d' % (
                    round_idx, cid + 1, len(party_list_this_round), scan_id, client_count))
                iter += num_step
                encoder_local = copy.deepcopy(agent_global.encoder)
                decoder_local = copy.deepcopy(agent_global.decoder)
                encoder_local = encoder_local.cuda() if torch.cuda.is_available() else encoder_local
                decoder_local = decoder_local.cuda() if torch.cuda.is_available() else decoder_local

                agent_local = Seq2SeqAgent(train_env, "", encoder_local, decoder_local, max_episode_len)
                encoder_optimizer = optim.Adam(encoder_local.parameters(), lr=learning_rate, weight_decay=weight_decay)
                decoder_optimizer = optim.Adam(decoder_local.parameters(), lr=learning_rate, weight_decay=weight_decay)

                agent_local.train(encoder_optimizer, decoder_optimizer, num_step, feedback=feedback_method)

                train_losses = np.array(agent_local.losses)
                assert len(train_losses) == num_step
                train_loss_avg = np.average(train_losses)
                # logger.info(f'Loss: {train_loss_avg}')
                train_loss_list.append(train_loss_avg)

                encoder_local_w = agent_local.encoder.state_dict()
                decoder_local_w = agent_local.decoder.state_dict()
                for key in new_encoder_global.keys():
                    new_encoder_global[key] += args.global_lr*(encoder_local_w[key]-agent_global.encoder.state_dict()[key])* (freq_this_round[cid])
                for key in new_decoder_global.keys():
                    new_decoder_global[key] += args.global_lr*(decoder_local_w[key]-agent_global.decoder.state_dict()[key])* (freq_this_round[cid])
                
                del agent_local, encoder_local, decoder_local, encoder_optimizer, decoder_optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            agent_global.encoder.load_state_dict(new_encoder_global)
            agent_global.decoder.load_state_dict(new_decoder_global)
            avg_loss = float(np.mean(train_loss_list)) if train_loss_list else 0.0
            loss_msg = f'Rd {round_idx}] global_loss={avg_loss:.4f}'

            logger.info(loss_msg)
            logger.info(f'Round {round_idx} client training loss: {np.mean(train_loss_list)}')

            if round_idx % log_every == 0:
                loss_str = f'round {round_idx}'
                per_scan_scores = {}

                if val_env is not None:
                    # Preserve training RNG stream.
                    _rng_py = random.getstate()
                    _rng_np = np.random.get_state()
                    _rng_cpu = torch.random.get_rng_state()
                    _rng_gpu = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

                    for scan_id in scans_list:
                        evaluator = evaluators_by_scan.get(scan_id)
                        if evaluator is None:
                            continue
                        if scan_id not in val_env.data or len(val_env.data.get(scan_id, [])) == 0:
                            continue

                        val_env.set_current_scan(scan_id)
                        agent_global.env = val_env
                        agent_global.results_path = os.path.join(
                            RESULT_DIR, f'{model_prefix}_val_seen_scan_{scan_id}_round_{round_idx}.json'
                        )
                        agent_global.test(use_dropout=False, feedback='argmax')
                        agent_global.write_results()
                        score_summary, _ = evaluator.score(agent_global.results_path)
                        if os.path.exists(agent_global.results_path):
                            os.remove(agent_global.results_path)

                        score_summary['data_count'] = len(val_env.data[scan_id])
                        per_scan_scores[scan_id] = score_summary

                        scan_str = f'  scan={scan_id} | datasize={len(val_env.data[scan_id])}'
                        for mk in ['success_rate', 'spl', 'nav_error', 'oracle success_rate',
                                   'oracle path_success_rate', 'dist_to_end_reduction', 'length']:
                            if mk in score_summary:
                                scan_str += f', {mk}: {score_summary[mk]:.3f}'
                        logger.info(scan_str)

                    random.setstate(_rng_py)
                    np.random.set_state(_rng_np)
                    torch.random.set_rng_state(_rng_cpu)
                    if _rng_gpu is not None:
                        torch.cuda.set_rng_state(_rng_gpu)

                if per_scan_scores:
                    metric_names = set()
                    for s in per_scan_scores.values():
                        metric_names.update(kk for kk in s if kk != 'data_count')
                    metric_avg = {}
                    for kk in metric_names:
                        metric_avg[kk] = np.mean(
                            [float(s.get(kk, 0.0)) for s in per_scan_scores.values()]
                        )
                    loss_str = f'round {round_idx} val_seen (fedavg, {len(per_scan_scores)} scans)'
                    for kk in ['success_rate', 'spl', 'nav_error',
                               'oracle success_rate', 'oracle path_success_rate',
                               'dist_to_end_reduction', 'length']:
                        if kk in metric_avg:
                            loss_str += f', {kk}: {metric_avg[kk]:.3f}'
                            data_log[f'val_seen {kk}'].append(metric_avg[kk])

                    spl_val = metric_avg.get('spl', 0.0)
                    sr_val = metric_avg.get('success_rate', 0.0)
                    if sr_val > best_val['val_seen']['success_rate']:
                        best_val['val_seen']['spl'] = spl_val
                        best_val['val_seen']['success_rate'] = sr_val
                        best_val['val_seen']['nav_error'] = metric_avg.get('nav_error', 0.0)
                        best_val['val_seen']['oracle success_rate'] = metric_avg.get('oracle success_rate', 0.0)
                        best_val['val_seen']['oracle path_success_rate'] = metric_avg.get('oracle path_success_rate', 0.0)
                        best_val['val_seen']['dist_to_end_reduction'] = metric_avg.get('dist_to_end_reduction', 0.0)
                        best_val['val_seen']['length'] = metric_avg.get('length', 0.0)
                        best_val['val_seen']['state'] = f'fedavg Rd {round_idx}, iter {iter}'
                        best_val['val_seen']['update'] = True
                else:
                    for kk in ['success_rate', 'spl', 'nav_error',
                               'oracle success_rate', 'oracle path_success_rate',
                               'dist_to_end_reduction', 'length']:
                        data_log[f'val_seen {kk}'].append(np.nan)

                agent_global.env = train_env

                logger.info('%s (%d %d%%) %s' % (
                    timeSince(start, float(round_idx+1)/comm_round),
                    iter, float(round_idx)/comm_round*100, loss_str
                ))
                data_log['iteration'].append(iter)
                df = pd.DataFrame(data_log)
                df.set_index('iteration')
                df_path = '%s%s-log.csv' % (PLOT_DIR, model_prefix)
                df.to_csv(df_path)
            if iter % 5000 == 0 and iter > 10:
                split_string = "-".join(train_env.splits)
                enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
                dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        
                agent_global.save(enc_path, dec_path)



    elif fl_mode == 'ours':
        # ==================================================================
        # Our method: Prefix + State Adapter + Gate  (personalized FL)
        #   encoder + decoder backbone  → aggregated via FedAvg
        #   decoder prefix/adapter + gate → personal per client
        # ==================================================================
        from model_prefix import is_personal_key
        from gating_policy import GatePolicyTwoTower
        from agent import PrefixSeq2SeqAgent

        if not hasattr(train_env, 'scans_list') or not hasattr(train_env, 'set_current_scan'):
            raise AttributeError('train_env must support scans_list/set_current_scan; use R2RBatchScan')

        scans_list = list(train_env.scans_list)
        if args.n_parties is not None:
            scans_list = scans_list[:min(args.n_parties, len(scans_list))]
        n_parties = len(scans_list)
        if n_parties == 0:
            raise ValueError('No scans found for ours')
        party_list = list(range(n_parties))
        n_party_per_round = max(1, int(n_parties * args.sample_fraction))

        # Helpers ----------------------------------------------------------
        def _extract_backbone_dec(sd):
            return {k: v.cpu().clone() for k, v in sd.items() if not is_personal_key(k)}

        def _extract_personal_dec(sd):
            return {k: v.cpu().clone() for k, v in sd.items() if is_personal_key(k)}

        def _opt_state_to_cpu(opt_sd):
            cpu_sd = {'state': {}, 'param_groups': copy.deepcopy(opt_sd['param_groups'])}
            for k, v in opt_sd['state'].items():
                cpu_sd['state'][k] = {
                    sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv
                    for sk, sv in v.items()
                }
            return cpu_sd

        # Gate policy ------------------------------------------------------
        gate_policy = GatePolicyTwoTower(
            hidden_size=enc_hidden_size,
            num_blocks=2,
            stats_dim=2,
            tower_dim=args.gate_hidden,
            obs_dim=2048,            # ResNet feature dim
        ).cuda()
   

        # Global state (CPU) -----------------------------------------------
        global_enc = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        global_dec_backbone = _extract_backbone_dec(decoder.state_dict())

        # Per-client personal states: {scan_id: (personal_dec_cpu, gate_cpu)}
        init_personal_dec = _extract_personal_dec(decoder.state_dict())
        init_gate_sd = {k: v.cpu().clone() for k, v in gate_policy.state_dict().items()}
        personal_states: dict = {}
        for _scan in scans_list:
            personal_states[_scan] = (
                copy.deepcopy(init_personal_dec), copy.deepcopy(init_gate_sd))

        # Per-client optimizer states: {scan_id: (prefix_opt_cpu, gate_opt_cpu)}
        personal_opt_states: dict = {}

        # Agent (shared GPU objects) ----------------------------------------
        agent = PrefixSeq2SeqAgent(
            train_env, '', encoder, decoder, gate_policy,
            episode_len=max_episode_len,
            lambda_smooth=args.lambda_smooth)
        

        # Evaluation setup --------------------------------------------------
        val_env = None
        if val_envs and 'val_seen' in val_envs:
            val_env = val_envs['val_seen'][0]
        evaluators_by_scan = {}
        if val_env is not None and hasattr(val_env, 'data'):
            for scan_id in scans_list:
                if scan_id in val_env.data and len(val_env.data.get(scan_id, [])) > 0:
                    evaluators_by_scan[scan_id] = Evaluation(
                        ['val_seen'], path_type=path_type, scans={scan_id})

        comm_round = args.rounds
        
        start = time.time()
        data_log = defaultdict(list)
        iter_total = 0
        # best_val = {'val_seen': {'spl': 0.0, 'sr': 0.0,

        #  'state': ''}}
        best_val = {'val_seen': {'spl': 0.0, 'success_rate': 0.0, 'nav_error': 0.0,
                               'oracle success_rate': 0.0, 'oracle path_success_rate': 0.0,
                               'dist_to_end_reduction': 0.0,
         'state': ''}}

        logger.info(
            f'Ours Training (prefix+gate): {n_parties} clients, '
            f'sample={n_party_per_round}/round, local_epoches={args.local_epoches}, '
            f'prefix_len={args.prefix_len}, adapter_mid={args.adapter_mid_dim}, '
            f'gate_hidden={args.gate_hidden}, lambda_smooth={args.lambda_smooth}, '
            f'prefix_lr={args.prefix_lr}, gate_lr={args.gate_lr}')

        # ==============================================================
        # Main communication-round loop
        # ==============================================================
        for rd in range(comm_round):
            # Client sampling
            if n_party_per_round < n_parties:
                party_this = random.sample(party_list, n_party_per_round)
            else:
                party_this = list(party_list)

            client_sizes = [len(train_env.data[scans_list[k]]) for k in party_this]
            total_size = float(sum(client_sizes)) if sum(client_sizes) > 0 else 1.0
            freq_this = [c / total_size for c in client_sizes]

            new_enc_global = copy.deepcopy(global_enc)
            new_dec_bb_global = copy.deepcopy(global_dec_backbone)

            round_losses = []
            round_gate_prefix = []
            round_gate_adapter = []
            

            for cid, k in enumerate(party_this):
                scan_id = scans_list[k]
                train_env.set_current_scan(scan_id)
                client_count = len(train_env.data[scan_id])
                num_step = args.local_epoches

                # ---- Load weights into shared GPU model ----
                # Encoder: global
                encoder.load_state_dict(global_enc)
                # Decoder: merge global backbone + personal prefix/adapter
                dec_sd = decoder.state_dict()
                for kn, v in global_dec_backbone.items():
                    dec_sd[kn].copy_(v)
                p_dec_sd, p_gate_sd = personal_states[scan_id]
                for kn, v in p_dec_sd.items():
                    dec_sd[kn].copy_(v)
                decoder.load_state_dict(dec_sd)
                # Gate: personal
                gate_policy.load_state_dict(p_gate_sd)

                encoder.train()
                decoder.train()
                gate_policy.train()

                # ---- Optimizers (3 groups) ----
                backbone_params = (
                    list(encoder.parameters())
                    + [p for n, p in decoder.named_parameters() if not is_personal_key(n)])
                prefix_params = [p for n, p in decoder.named_parameters() if is_personal_key(n)]

                backbone_opt = optim.Adam(backbone_params, lr=learning_rate, weight_decay=weight_decay)
                prefix_opt = optim.Adam(prefix_params, lr=args.prefix_lr, weight_decay=weight_decay)
                gate_opt = optim.Adam(gate_policy.parameters(), lr=args.gate_lr, weight_decay=weight_decay)

                # Restore personal optimizer states
                if scan_id in personal_opt_states:
                    prefix_opt.load_state_dict(personal_opt_states[scan_id][0])
                    gate_opt.load_state_dict(personal_opt_states[scan_id][1])

                # ---- Train ----
                agent.env = train_env
                agent.train(backbone_opt, prefix_opt, gate_opt, num_step,
                            feedback=feedback_method)

                round_losses.append(np.mean(agent.losses) if agent.losses else 0.0)
                round_gate_prefix.append(getattr(agent, 'gate_prefix_mean', 0.0))
                round_gate_adapter.append(getattr(agent, 'gate_adapter_mean', 0.0))

                

                # ---- Extract backbone deltas for aggregation ----
                enc_w_cpu = {kk: vv.cpu() for kk, vv in encoder.state_dict().items()}
                dec_bb_cpu = _extract_backbone_dec(decoder.state_dict())

                for key in new_enc_global.keys():
                    new_enc_global[key] += args.global_lr * (
                        enc_w_cpu[key] - global_enc[key]) * freq_this[cid]
                for key in new_dec_bb_global.keys():
                    new_dec_bb_global[key] += args.global_lr * (
                        dec_bb_cpu[key] - global_dec_backbone[key]) * freq_this[cid]

                # ---- Save personal states ----
                personal_states[scan_id] = (
                    _extract_personal_dec(decoder.state_dict()),
                    {kk: vv.cpu().clone() for kk, vv in gate_policy.state_dict().items()},
                )
                personal_opt_states[scan_id] = (
                    _opt_state_to_cpu(prefix_opt.state_dict()),
                    _opt_state_to_cpu(gate_opt.state_dict()),
                )

                iter_total += num_step

                logger.info(
                    f'  [Rd {rd}] client {cid+1}/{len(party_this)}: scan={scan_id}, '
                    f'data={client_count}, steps={num_step}, '
                    f'loss={round_losses[-1]:.4f}, '
                    f'g_prefix={round_gate_prefix[-1]:.3f}, '
                    f'g_adapter={round_gate_adapter[-1]:.3f}')

                del backbone_opt, prefix_opt, gate_opt, enc_w_cpu, dec_bb_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---- Aggregate global model ----
            global_enc = new_enc_global
            global_dec_backbone = new_dec_bb_global

            

            avg_loss = float(np.mean(round_losses)) if round_losses else 0.0
            logger.info(
                f'[ours Rd {rd}] avg_loss={avg_loss:.4f}, '
                f'gate_prefix={np.mean(round_gate_prefix):.3f}, '
                f'gate_adapter={np.mean(round_gate_adapter):.3f}')

            

            # ==============================================================
            # Per-scan evaluation (personal models)
            # ==============================================================
            if val_env is not None and (rd % log_every == 0 or rd > 1800):
                _rng_py  = random.getstate()
                _rng_np  = np.random.get_state()
                _rng_cpu = torch.random.get_rng_state()
                _rng_gpu = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

                eval_enc = copy.deepcopy(encoder).cuda()
                eval_dec = copy.deepcopy(decoder).cuda()
                eval_gate = copy.deepcopy(gate_policy).cuda()
                eval_agent = PrefixSeq2SeqAgent(
                    val_env, '', eval_enc, eval_dec, eval_gate, max_episode_len)
                
                per_scan_scores = {}
                for scan_id in scans_list:
                    if scan_id not in personal_states:
                        continue
                    evaluator = evaluators_by_scan.get(scan_id)
                    if evaluator is None:
                        continue
                    if scan_id not in val_env.data or len(val_env.data.get(scan_id, [])) == 0:
                        continue

                    # Load global enc + merged dec + personal gate
                    eval_enc.load_state_dict(copy.deepcopy(global_enc))
                    eval_dec_sd = eval_dec.state_dict()
                    for kn, v in global_dec_backbone.items():
                        eval_dec_sd[kn].copy_(v)
                    p_dec_sd, p_gate_sd = personal_states[scan_id]
                    for kn, v in p_dec_sd.items():
                        eval_dec_sd[kn].copy_(v)
                    eval_dec.load_state_dict(eval_dec_sd)
                    eval_gate.load_state_dict(copy.deepcopy(p_gate_sd))

                    val_env.set_current_scan(scan_id)
                    eval_agent.env = val_env
                    eval_agent.results_path = os.path.join(
                        RESULT_DIR,
                        f'{model_prefix}_val_seen_scan_{scan_id}_round_{rd}.json')
                    eval_agent.test(use_dropout=False, feedback='argmax')
                    eval_agent.write_results()
                    score_summary, _ = evaluator.score(eval_agent.results_path)
                    if os.path.exists(eval_agent.results_path):
                        os.remove(eval_agent.results_path)
                    score_summary['data_count'] = len(val_env.data[scan_id])
                    per_scan_scores[scan_id] = score_summary

                    scan_str = f'  scan={scan_id} | datasize={len(val_env.data[scan_id])}'
                    data_log['round'].append(rd)
                    data_log['iteration'].append(iter_total)
                    data_log['scan'].append(scan_id)
                    for mk in ['success_rate', 'spl', 'nav_error',
                               'oracle success_rate', 'oracle path_success_rate',
                               'dist_to_end_reduction', 'length']:
                        if mk in score_summary:
                            scan_str += f', {mk}: {score_summary[mk]:.3f}'
                            data_log['%s' % mk].append(score_summary[mk])
                    logger.info(scan_str)

                if per_scan_scores:
                    metric_names = set()
                    for s in per_scan_scores.values():
                        metric_names.update(kk for kk in s if kk != 'data_count')
                    metric_avg = {}
                    for kk in metric_names:
                        metric_avg[kk] = np.mean(
                            [float(s.get(kk, 0.0)) for s in per_scan_scores.values()])

                    loss_str = f'round {rd} val_seen (ours, {len(per_scan_scores)} scans)'
                    for kk in ['success_rate', 'spl', 'nav_error',
                               'oracle success_rate', 'oracle path_success_rate',
                               'dist_to_end_reduction', 'length']:
                        if kk in metric_avg:
                            loss_str += f', {kk}: {metric_avg[kk]:.3f}'
                    logger.info('%s (%d %d%%) %s' % (
                        timeSince(start, float(rd + 1) / comm_round),
                        iter_total, float(rd) / comm_round * 100, loss_str))

                    spl_val = metric_avg.get('spl', 0.0)
                    sr_val = metric_avg.get('success_rate', 0.0)
                    if sr_val > best_val['val_seen']['success_rate']:
                        best_val['val_seen']['spl'] = spl_val
                        best_val['val_seen']['success_rate'] = sr_val
                        best_val['val_seen']['nav_error'] = metric_avg.get('nav_error', 0.0)
                        best_val['val_seen']['oracle success_rate'] = metric_avg.get('oracle success_rate', 0.0)
                        best_val['val_seen']['oracle path_success_rate'] = metric_avg.get('oracle path_success_rate', 0.0)
                        best_val['val_seen']['dist_to_end_reduction'] = metric_avg.get('dist_to_end_reduction', 0.0)
                        best_val['val_seen']['length'] = metric_avg.get('length', 0.0)
                        best_val['val_seen']['state'] = f'fedavg Rd {round_idx}, iter {iter}'
                        # logger.info(f'  [new best] spl={spl_val:.4f}')

                del eval_agent, eval_enc, eval_dec, eval_gate
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                random.setstate(_rng_py)
                np.random.set_state(_rng_np)
                torch.random.set_rng_state(_rng_cpu)
                if _rng_gpu is not None:
                    torch.cuda.set_rng_state(_rng_gpu)

        logger.info(
            f'Ours training complete. BEST val_seen: '
            f'{best_val["val_seen"]["state"]}, spl={best_val["val_seen"]["spl"]:.4f}')


def setup():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)





def train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind, fl_mode):
    ''' Train on the training set, and validate according to mode. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    # Federated modes train one client/scan at a time; centralized training
    # uses the original mixed-scan environment.
    if fl_mode in ['fedavg', 'ours']:
        train_env = R2RBatchScan(features, batch_size=batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)
    else:
        train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)
        print('data size: ',train_env.size())

    # Create validation environments
    if fl_mode in ['fedavg', 'ours']:
        # FL modes evaluate on per-scan val_seen; unseen split is not required.
        val_envs = {
            'val_seen': (
                R2RBatchScan(features, batch_size=batch_size, splits=['val_seen'], tokenizer=tok,
                            path_type=path_type, history=history, blind=blind),
                None,
            )
        }
    else:
        val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                    tokenizer=tok, path_type=path_type, history=history, blind=blind),
                    Evaluation([split], path_type=path_type)) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    if fl_mode == 'ours':
        from model_prefix import PrefixAttnDecoderLSTM
        decoder = PrefixAttnDecoderLSTM(
            Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
            action_embedding_size, hidden_size, dropout_ratio,
            prefix_len=args.prefix_len,
            adapter_mid_dim=args.adapter_mid_dim).cuda()
    else:
        decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                      action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters,
          path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
          val_envs=val_envs,fl_mode=args.fl_mode, args=args,
          _vocab_size=len(vocab), enc_hidden_size=enc_hidden_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_type', type=str, required=True,
                        help='planner_path, player_path, or trusted_path')
    parser.add_argument('--history', type=str, required=True,
                        help='none, target, oracle_ans, nav_q_oracle_ans, or all')
    parser.add_argument('--feedback', type=str, required=True,
                        help='teacher or sample')
    # REMOVE: --eval_type exposed the removed test/submission code path.
    # parser.add_argument('--eval_type', type=str, required=True)
    parser.add_argument('--blind', action='store_true', required=False,
                        help='whether to replace the ResNet encodings with zero vectors at inference time')
    parser.add_argument('--name',type=str)
    parser.add_argument('--fl_mode', type=str, default='c',
                        choices=['c', 'fedavg', 'ours'],
                        help='Training method')
    parser.add_argument('--local_epoches',type=int, default=3)
    parser.add_argument('--seed',type=int, default=1)
    parser.add_argument('--bs',type=int, default=100)# sample_fraction
    parser.add_argument('--sample_fraction',type=float, default=0.2) # n_parties
    parser.add_argument('--n_parties',type=int, default=58)
    parser.add_argument('--rounds',type=int, default=1000)
    parser.add_argument('--global_lr', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=50)
    

    # ---- Ours (prefix + gate) arguments ----
    parser.add_argument('--prefix_len', type=int, default=4,
                        help='Number of learnable prefix tokens prepended to encoder context')
    parser.add_argument('--adapter_mid_dim', type=int, default=64,
                        help='Bottleneck dimension for the state adapter MLP')
    parser.add_argument('--prefix_lr', type=float, default=1e-3,
                        help='Learning rate for prefix + adapter parameters')
    parser.add_argument('--gate_lr', type=float, default=1e-3,
                        help='Learning rate for gate policy')
    parser.add_argument('--gate_hidden', type=int, default=128,
                        help='Tower dimension inside GatePolicyTwoTower')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                        help='Weight for gate temporal smoothness loss')
    args = parser.parse_args()


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SEED = args.seed
    TRAIN_VOCAB = 'datasets/CVDN/train_vocab.txt'
    TRAINVAL_VOCAB = 'datasets/CVDN/trainval_vocab.txt'
    RESULT_DIR = 'logs/%s' % args.name + '/' + timestamp
    SNAPSHOT_DIR = 'snap/%s' % args.name + '/' + timestamp
    PLOT_DIR = 'plots/%s' % args.name + '/' + timestamp
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)


    IMAGENET_FEATURES = 'img_features/ResNet-152-places365.tsv'
    safe_name = args.name.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=os.path.join(RESULT_DIR, f"{safe_name}_{timestamp}.log"),
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
    logger.info("===== Training Arguments =====")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("==============================")

    # Training settings.
    agent_type = 'seq2seq'

    # Fixed params from MP.
    features = IMAGENET_FEATURES
    batch_size = 100
    word_embedding_size = 256
    action_embedding_size = 32
    target_embedding_size = 32
    hidden_size = 512
    bidirectional = False
    dropout_ratio = 0.5
    learning_rate = 0.0001
    weight_decay = 0.0005

    assert args.path_type in ['planner_path', 'player_path', 'trusted_path']
    assert args.history in ['none', 'target', 'oracle_ans', 'nav_q_oracle_ans', 'all']
    assert args.feedback in ['sample', 'teacher']
    # REMOVE: validation training is the only supported entrypoint.
    # assert args.eval_type in ['val', 'test']

    blind = args.blind

    # Set default args.
    path_type = args.path_type
    # In MP, max_episode_len = 20 while average hop range [4, 7], e.g. ~3x max.
    # max_episode_len has to account for turns; this heuristically allowed for about 1 turn per hop.
    if path_type == 'planner_path':
        max_episode_len = 20  # [1, 6], e.g., ~3x max
    else:
        max_episode_len = 80  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]

    # Input settings.
    history = args.history
    # In MP, MAX_INPUT_LEN = 80 while average utt len is 29, e.g., a bit less than 3x avg.
    if history == 'none':
        MAX_INPUT_LENGTH = 1  # [<EOS>] fixed length.
    elif history == 'target':
        MAX_INPUT_LENGTH = 3  # [<TAR> target <EOS>] fixed length.
    elif history == 'oracle_ans':
        MAX_INPUT_LENGTH = 70  # 16.16+/-9.67 ora utt len, 35.5 at x2 stddevs. 71 is double that.
    elif history == 'nav_q_oracle_ans':
        MAX_INPUT_LENGTH = 120  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
    else:  # i.e., 'all'
        MAX_INPUT_LENGTH = 120 * 6  # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A. 5.67 at x2 std. Call it 6 (real max 13).

    # Training settings.
    feedback_method = args.feedback
    n_iters = 20000 if feedback_method == 'teacher' else 40000

    # Model prefix to uniquely id this instance.
    model_prefix = 'val-seq2seq-%s-%s-%d-%s-imagenet' % (
        history, path_type, max_episode_len, feedback_method)
    if blind:
        model_prefix += '-blind'

    train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH,
              feedback_method, n_iters, model_prefix, blind, args.fl_mode)


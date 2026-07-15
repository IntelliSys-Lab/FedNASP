#!/bin/bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR" || exit 1

# CUDA_VISIBLE_DEVICES=1 python reverie_src/train.py 
# DEBUG_GATE_MODE=gate_fixed:1.0 \
# name=REVERIE-ours/test
# mkdir -p snap/$name
# mkdir -p logs/$name
# MEM_DEBUG=1 MEM_DEBUG_EVERY=1 \
# CUDA_VISIBLE_DEVICES=0 python reverie_src/train.py \
#       --vlnbert vilbert \
#       --train listener \
#       --test_only 0 \
#       --init_bert_file datasets/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin \
#       --features img_features/ResNet-152-places365.tsv \
#       --maxAction 15 \
#       --maxInput 50 \
#       --batchSize 8 \
#       --feedback sample \
#       --lr 1e-5 \
#       --iters 300000 \
#       --optim adamW \
#       --mlWeight 0.20 \
#       --angleFeatSize 128 \
#       --featdropout 0.4 \
#       --dropout 0.5 \
#       --fl_mode ours \
#       --comm_round 300 \
#       --local_epoches 3 \
#       --sample_fraction 0.2 \
#       --global_lr 1.0 \
#       --log_every 10 \
#       --attn_prefix_mode fedperfix_add \
#       --prefix_mid_dim 256 \
#       --prefix_scale 1.0 \
#       --prefix_lr 1e-5 \
#       --gate_lr 1e-5 \
#       --name $name \
      # --hybrid_global_ckpt snap/REVERIE-FedAvg-airbert/train-test/state_dict/round_299 \
    # --ours_data_parallel \
  # --ours_dp_devices 0,2 \
  #       --ours_ddp \
  # --ours_ddp_backend nccl \
  # --ours_ddp_find_unused \
  # --ours_dp_safe_nccl \

for seed in 10 
do
  name=REVERIE-ours/test_seed${seed}
  log_file=logs/$name/train.log

  mkdir -p snap/$name
  mkdir -p logs/$name

  MEM_DEBUG=1 MEM_DEBUG_EVERY=1 \
  CUDA_VISIBLE_DEVICES=0 python -u reverie_src/train.py \
      --vlnbert vilbert \
      --train listener \
      --test_only 0 \
      --init_bert_file datasets/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin \
      --features img_features/ResNet-152-places365.tsv \
      --maxAction 15 \
      --maxInput 50 \
      --batchSize 8 \
      --feedback sample \
      --lr 1e-5 \
      --iters 300000 \
      --optim adamW \
      --mlWeight 0.20 \
      --angleFeatSize 128 \
      --featdropout 0.4 \
      --dropout 0.5 \
      --fl_mode ours \
      --comm_round 400 \
      --local_epoches 3 \
      --sample_fraction 0.2 \
      --global_lr 1.0 \
      --log_every 10 \
      --attn_prefix_mode fedperfix_add \
      --prefix_mid_dim 256 \
      --prefix_scale 1.0 \
      --prefix_lr 1e-5 \
      --gate_lr 1e-5 \
      --seed ${seed} \
      --name $name \
      --disk_n_parties 15
      2>&1 | tee "$log_file"
done
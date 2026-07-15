name=cvdn/ours/prefix_gate

flag="
      --path_type=trusted_path
      --history=all
      --feedback=sample
      --fl_mode ours
      --sample_fraction 0.2
      --local_epoches 3
      --rounds 2000
      --log_every 10
      --prefix_len 4
      --adapter_mid_dim 64
      --prefix_lr 1e-4
      --gate_lr 1e-4
      --gate_hidden 128
      --lambda_smooth 0
      "

mkdir -p snap/$name

CUDA_VISIBLE_DEVICES=0 python cvdn_src/train.py $flag --name $name
# for seed in 10 33 465 74 235
# do
#   name=cvdn/ours/prefix_gate_seed${seed}

#   flag="
#       --path_type=trusted_path
#       --history=all
#       --feedback=sample
#       --fl_mode ours
#       --sample_fraction 0.2
#       --local_epoches 3
#       --rounds 2000
#       --log_every 10
#       --prefix_len 4
#       --adapter_mid_dim 64
#       --prefix_lr 1e-4
#       --gate_lr 1e-4
#       --gate_hidden 128
#       --lambda_smooth 0
#       --seed ${seed}
#       "

#   mkdir -p snap/$name

#   CUDA_VISIBLE_DEVICES=0 python cvdn_src/train.py $flag --name $name
# done
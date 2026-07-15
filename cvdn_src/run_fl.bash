name=cvdn/fl

flag="
      --path_type=trusted_path
      --history=all
      --feedback=sample
      --fl_mode fedavg
      --log_every 100
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=4 python cvdn_src/train.py $flag --name $name


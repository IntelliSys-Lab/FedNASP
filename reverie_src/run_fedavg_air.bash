# export AIRBERT_ROOT=$(pwd)
# export PYTHONPATH=${PYTHONPATH}:${AIRBERT_ROOT}/build

name=REVERIE-FedAvg-airbert/train-test

flag="--vlnbert vilbert

      --train listener
      --test_only 0
      
      --fl_mode fedavg
      --local_epoches 3
      --sample_fraction 0.05
      --global_lr 1.0
      --comm_round 500
      --init_bert_file datasets/vln-bert/r2rM_bnbMS_2capt.pth1.4.bin
      --features img_features/ResNet-152-places365.tsv
      --maxAction 15
      --maxInput 50
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 200000
      --log_every 1
      --optim adamW
      --load snap/REVERIE-FedAvg-airbert/train-test/state_dict/round_299
      --mlWeight 0.20
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
mkdir -p logs/$name
CUDA_VISIBLE_DEVICES=5 python reverie_src/train.py $flag --name $name

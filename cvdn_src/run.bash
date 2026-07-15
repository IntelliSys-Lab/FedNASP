name=cvdn/centralized

flag="
	--path_type=trusted_path
	--history=all
	--feedback=sample
	--fl_mode c
	--log_every 100
	"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=2 python cvdn_src/train.py $flag --name $name


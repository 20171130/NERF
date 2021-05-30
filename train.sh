tb_port=6018
port=1$tb_port
n_gpu=2
name=tmp
OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py --world_size $n_gpu --train \
--vae \
--batch_size 64 --dropout=0.1 --depth 6 --dim 256 \
--prefix inc4 --name $name --epochs 100 
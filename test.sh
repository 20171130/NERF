tb_port=6018
port=1$tb_port
n_gpu=1 
name=tmp
OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=6
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py --world_size $n_gpu --test \
--vae \
--batch_size 256 --dropout=0.1 --depth 6 --dim 256 \
--prefix inc4 --name $name --epochs 100 --checkpoint tmp.pt --temperature 0.0
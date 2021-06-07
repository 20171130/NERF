Code for our paper "Non-Autoregressive Electron Redistribution Modeling for Reaction Prediction" (ICML 2021)

This repository contains a legacy version of our implementation, which we used to produce the results in our paper.
Although the code style may be improved, we leave it unchanged for reproducibility.

# Dependencies
The code has been tested using the following environment:
* python 3.7.6
* numpy 1.18.5
* torch 1.8.1
* rdkit 2018.03.4

# Examples
## Data Preprocessing
For each atom, we assume it has at most 6 bonds, and at most 4 bonds around it are broken/formed during reactions.
These constraints can be easily relaxed by changing the hyperparameters in `preprocess.py`, but we leave them unchanged for reproducibility.
Notice that about 0.3% of reactions in USPTO-MIT do not satisfy the criteria and we subtract all our accuracy by 0.3% for a fair comparison with other methods.

If the training set is in data/train.txt, call 
`process("data/train")`
defined in `preprocess.py`. 


## Training

```bash
tb_port=6018
port=1$tb_port
n_gpu=2
name=tmp
OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py --world_size $n_gpu --train \
--vae \
--batch_size 128 --dropout=0.1 --depth 6 --dim 256 \
--prefix data --name $name --epochs 250
```

## Testing
For top-k sampling, we use temperature = 0.7*1.3**n, n = 0, 1,...and use the first k different valid predictions at the lowest temperature as the top-k predictions.

```bash
tb_port=6018
port=1$tb_port
n_gpu=1 
name=tmp
OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py --world_size $n_gpu --test \
--vae \
--batch_size 256 --dropout=0.1 --depth 6 --dim 256 \
--prefix data --name $name --epochs 100 --checkpoint tmp.pt --temperature 0.7
```
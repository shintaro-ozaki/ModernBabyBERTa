#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --account=is-nlp
#SBATCH --job-name=👶.pretrain
#SBATCH -o logs/slurm-%x-%j.log

set -eu
project=$(pwd)
source .venv/bin/activate

python $project/modernbabyberta/train_model.py \
    --corpus_dir $project/data/train_10M \
    --output_dir ./checkpoints/modernbabyberta-pretrain \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_seq_length 128 \
    --mlm_probability 0.15 \
    --tokenizer_path $project/tokenizer_babylm_10M \
    --save_steps 2000

echo Done

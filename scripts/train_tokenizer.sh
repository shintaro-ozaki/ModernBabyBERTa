#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 8
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=👶.tokenizer
#SBATCH -o logs/slurm-%x-%j.log

set -eu
project=$(pwd)
source .venv/bin/activate

python $project/modernbabyberta/train_tokenizer.py \
  --corpus_dir $project/data/train_10M \
  --output_dir $project/tokenizer_babylm_10M \
  --vocab_size 8192

echo Done

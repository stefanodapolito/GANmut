#!/bin/bash
#SBATCH  --output=log/lin_2d_v2_%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
source  /home/paudeld/apps/miniconda2/etc/profile.d/conda.sh
conda activate pytcu99

python -u main.py --mode train --image_size 128 --architecture_v2 1.\
               --c_dim 7  --lambda_g_strength=4  \
               --sample_dir samples/samples_linear_2d --log_dir GANmut/logs_linear_2d \
               --model_save_dir GANmut/models_linear_2d --result_dir GANmut/results_linear_2d "$@"

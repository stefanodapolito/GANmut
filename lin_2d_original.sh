#!/bin/bash
#SBATCH  --output=log/lin_2d_original%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
source  /home/paudeld/apps/miniconda2/etc/profile.d/conda.sh
conda activate pytcu99

python -u main.py --mode train --image_size 128 \
               --c_dim 7  --lambda_g_strength=4 --cycle_loss original  \
               --sample_dir samples/samples_linear_2d_cl_orig --log_dir GANmut/logs_linear_2d_cl_orig \
               --model_save_dir GANmut/models_linear_2d_cl_orig --result_dir GANmut/results_linear_2d_cl_orig "$@"

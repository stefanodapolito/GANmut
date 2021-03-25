#!/bin/bash
#SBATCH  --output=log/gaus_2d_%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
source  /home/paudeld/apps/miniconda2/etc/profile.d/conda.sh
conda activate pytcu99

python -u main.py --mode train --image_size 128  --parametrization gaussian \
               --c_dim 7  --lambda_cls=2.   \
               --sample_dir samples/samples_gaussian_2d_2 --log_dir GANmut/logs_gaussian_2d_2 \
               --model_save_dir GANmut/models_gaussian_2d_2 --result_dir GANmut/results_gaussian_2d_2 "$@"

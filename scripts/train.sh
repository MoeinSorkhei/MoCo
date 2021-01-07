#!/bin/bash

cd /home/sorkhei/MoCo || { echo "Could not change directory, aborting..."; exit 1; }
source /opt/anaconda/etc/profile.d/conda.sh
conda activate detect

python main_moco.py \
  --arch resnet34 \
  --lr 0.001 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --devices 0,1,2,3,4,5,6,7 \
  --checkpoint_dir /raid/sorkhei/MoCo/checkpoints/conf_2 \
  --image_names_file all_others/data_registry/train_images.txt --imread_mode 2 \
  --resume /raid/sorkhei/MoCo/checkpoints/conf_2/checkpoint_0161.pth.tar \
  /raid/sorkhei/MoCo/data/data_1910_256


# NOTE: this has --resume option right now
# to append both stdout and stdin to file: bash scripts/train.sh >> all_others/logs/conf_1.txt 2>&1 &
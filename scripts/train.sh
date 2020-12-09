python main_moco.py \
  --arch resnet34 \
  --lr 0.0001 \
  --batch-size 128 \
  --moco-t 0.2 --moco-k 8192 --mlp --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --checkpoint_dir /storage/sorkhei/MoCo \
  /storage/sorkhei/Detectability/data/all_png/train_val/resize_2
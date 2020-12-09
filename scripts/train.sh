python main_moco.py \
  --arch resnet50 \
  --lr 0.015 \
  --batch-size 128 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /storage/sorkhei/Detectability/data/all_png/train_val/resize_2

#  --comet \  # it seems there is a probelm with comet at this point

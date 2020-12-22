python main_moco.py \
  --arch resnet34 \
  --lr 0.0001 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --devices 0,1,2,3,4,5,6,7 \
  --checkpoint_dir /raid/sorkhei/MoCo/checkpoints/conf_1 \
  --image_names_file data_registry/train_images.txt --imread_mode 2 \
  /raid/sorkhei/MoCo/data/data_1910_256

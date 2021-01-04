import os
import glob
import logging
import sys
import torch
import matplotlib.pyplot as plt

from .file_io import *


def set_gpu_devices(devices):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    print(f'Setting GPU devices is done. '
          f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}, '
          f'device count: {torch.cuda.device_count()}')


def show_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model total params: {total_params:,} - trainable params: {trainable_params:,}')


def files_with_suffix(directory, suffix, pure=False):
    """
    retrieving all files with the given suffix from a folder
    :param suffix: -
    :param directory: -
    :param pure: if set to True, only filenames are returned (as opposed to absolute paths)
    """
    files = [os.path.abspath(path) for path in glob.glob(os.path.join(directory, '**', f'*{suffix}'), recursive=True)]
    if pure:
        files = [os.path.split(file)[-1] for file in files]
    return files


def get_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(filename)s line %(lineno)d] %(message)s"  # also get the function name
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    return root


def waited_print(string):
    print(string)
    print('====== Waiting for input')
    input()


def plot_log_file(file, metrics, title):
    lines = read_file_to_list(file)
    lines = [line for line in lines if line.startswith('Epoch')]  # remove initial lines

    acc_at1_list, acc_at1_avg_list = [], []
    acc_at5_list, acc_at5_avg_list = [], []
    loss_list, loss_avg_list = [], []

    for line in lines:
        the_list = line.split('\t')
        loss_part = the_list[3]
        loss, loss_avg = float(loss_part.split(' ')[1]), float(loss_part.split(' ')[2][1:-1])

        acc_at1_part = the_list[4]
        acc_at5_part = the_list[5]

        acc_at_1, acc_at1_avg = float(acc_at1_part[6:12].strip()), float(acc_at1_part[14:20].strip())
        acc_at_5, acc_at5_avg = float(acc_at5_part[6:12].strip()), float(acc_at5_part[14:20].strip())

        loss_list.append(loss)
        loss_avg_list.append(loss_avg)

        acc_at1_list.append(acc_at_1)
        acc_at1_avg_list.append(acc_at1_avg)
        acc_at5_list.append(acc_at_5)
        acc_at5_avg_list.append(acc_at5_avg)

    plt.title(title)
    if 'loss' in metrics:
        plt.plot(loss_list, label='loss')
        plt.plot(loss_avg_list, label='loss_avg')

    if 'acc_at1' in metrics:
        plt.plot(acc_at1_list, label='acc_at1')
        plt.plot(acc_at1_avg_list, label='acc_at1_avg')

    if 'acc_at5' in metrics:
        plt.plot(acc_at5_list, label='acc_at5')
        plt.plot(acc_at5_avg_list, label='acc_at5_avg')

    plt.legend()
    plt.show()



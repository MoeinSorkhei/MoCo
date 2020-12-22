# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import cv2
from PIL import ImageFilter, Image
import random
from torch.utils import data
import os

import helper


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MammoDataset(data.Dataset):
    def __init__(self, image_names, data_folder, transformations, imread_mode):
        self.data_folder = data_folder
        self.transformations = transformations
        self.image_names = image_names
        self.imread_mode = imread_mode
        print(f'MammoDataset created with imread_mode: {self.imread_mode}, len image_names: {len(self.image_names):,}')

    def __len__(self):
        return len(self.image_names)  # number of png images in data folder

    def __getitem__(self, index):
        image_path = os.path.join(self.data_folder, self.image_names[index])

        if self.imread_mode == 1:  # read with PIL
            image = Image.open(image_path)
        else:
            image_array = cv2.imread(image_path)  # read with cv2 as RGB with 3 channels - for 16-bit images that PIL cannot understand
            image = Image.fromarray(image_array)  # convert to PIL image
        return self.transformations(image)

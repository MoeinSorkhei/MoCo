# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
    def __init__(self, data_folder, transformations):
        self.data_folder = data_folder
        self.transformations = transformations
        self.image_names = helper.files_with_suffix(directory=self.data_folder, suffix='.png', pure=True)

    def __len__(self):
        return len(self.image_names)  # number of png images in data folder

    def __getitem__(self, index):
        image_path = os.path.join(self.data_folder, self.image_names[index])
        image = Image.open(image_path)
        return self.transformations(image)

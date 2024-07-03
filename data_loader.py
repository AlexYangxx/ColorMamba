import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.utils import data


def get_data_paths():
    path_train_nir = './datasets/NIR'
    path_train_rgb = './datasets/RGB-Registered'

    path_val = './datasets/Validation'
    nir_files = glob.glob(path_train_nir + '/*.png')
    nir_files.sort()
    nir_files_val = glob.glob(path_val + '/*_nir*.png')
    nir_files_val.sort()

    rgb_files = glob.glob(path_train_rgb + '/*.png')
    rgb_files.sort()
    rgb_files_val = glob.glob(path_val + '/*_rgb_reg.png')
    rgb_files_val.sort()

    train_files = np.stack([nir_files, rgb_files], axis=1)
    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return train_files, val_files


def get_test_paths():
    path_val = './datasets/Testing'
    nir_files_val = glob.glob(path_val + '/*_nir*.png')
    nir_files_val.sort()

    rgb_files_val = glob.glob(path_val + '/*_rgb_reg.png')
    rgb_files_val.sort()

    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return val_files


def to_hsv(img):
    hsv_images = []
    for img_ in img:
        hsv_image = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)
        hsv_images.append(hsv_image)
    hsv_images = np.stack(hsv_images, axis=0)

    return hsv_images


def nor(img):
    img -= np.min(img)
    img = img / (np.max(img) + 1e-3)
    return img


def randomCrop(img, width, height):
    assert img.size[0] >= height
    assert img.size[1] >= width

    x = random.randint(0, img.size[0] - width)
    y = random.randint(0, img.size[1] - height)
    img = img.crop((x, y, x + width, y + height))

    return img, [x, y, x + width, y + height]


def crop_resize(img, position, resize_size):
    img = img.crop(position)  # 根据给定的位置坐标进行裁剪
    img = img.resize(resize_size, Image.BICUBIC)  # 使用双三次插值法调整图像大小

    return img


class Dataset(data.Dataset):
    def __init__(self, files, shape=(256, 256), return_name=False):
        self.files = files
        self.return_name = return_name
        self.input_shape = shape
        self.up_pix = 30

    def __len__(self):
        return len(self.files)

    def read_data(self, img_path, position=None, rad=None, rad2=None, factor_contrast=None):
        img_rgb = Image.open(img_path).convert('RGB')

        img_gray = Image.open(img_path).convert('L')
        if rad is None:
            rad = random.uniform(0, 1)
        if rad2 is None:
            rad2 = random.uniform(0, 1)
        if rad2 < .5:
            img_rgb = ImageOps.mirror(img_rgb)
            img_gray = ImageOps.mirror(img_gray)
        if rad < .5:
            if factor_contrast is None:
                factor_contrast = random.uniform(0.5, 1.5)
            enhancer_contrast = ImageEnhance.Contrast(img_rgb)
            img_rgb = enhancer_contrast.enhance(factor_contrast)
        else:
            factor_contrast = None
        if position is None:
            img_rgb, position = randomCrop(img_rgb, 200, 200)
            img_rgb = img_rgb.resize([256, 256], Image.BICUBIC)
        else:
            img_rgb = crop_resize(img_rgb, position, [256, 256])

        img_gray = crop_resize(img_gray, position, [256, 256])

        img_hsv = img_rgb.copy().convert('HSV')
        img_rgb = np.array(img_rgb)
        img_rgb = nor(img_rgb)

        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)

        img_hsv = np.array(img_hsv)
        img_hsv = nor(img_hsv)

        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).type(torch.FloatTensor)
        img_gray = torch.from_numpy(img_gray[None]).type(torch.FloatTensor)
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).type(torch.FloatTensor)

        return img_gray, img_rgb, img_hsv, position, rad, rad2, factor_contrast

    def __getitem__(self, index):
        nir_gray, nir_rgb, nir_hsv, position, rad, rad2, factor_contrast = self.read_data(self.files[index][0])
        rgb_gray, rgb_rgb, rgb_hsv, *_ = self.read_data(self.files[index][1], position, rad, rad2, factor_contrast)
        return {
            'nir_gray': nir_gray, 'nir_rgb': nir_rgb, 'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray, 'rgb_rgb': rgb_rgb, 'rgb_hsv': rgb_hsv,
            'nir_path': self.files[index][0], 'rgb_path': self.files[index][1]
        }


class Dataset_test(data.Dataset):
    def __init__(self, files, shape=(256, 256), return_name=False):
        self.files = files
        self.return_name = return_name
        self.input_shape = shape
        self.up_pix = 30

    def __len__(self):
        return len(self.files)

    def read_data(self, img_path):
        img_rgb = Image.open(img_path).convert('RGB').resize(self.input_shape)

        img_gray = Image.open(img_path).convert('L').resize(self.input_shape)

        img_hsv = img_rgb.copy().convert('HSV')
        img_rgb = np.array(img_rgb)
        img_rgb = nor(img_rgb)

        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)

        img_hsv = np.array(img_hsv)
        img_hsv = nor(img_hsv)

        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).type(torch.FloatTensor)
        img_gray = torch.from_numpy(img_gray[None]).type(torch.FloatTensor)
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).type(torch.FloatTensor)

        return img_gray, img_rgb, img_hsv

    def __getitem__(self, index):
        nir_gray, nir_rgb, nir_hsv = self.read_data(self.files[index][0])
        rgb_gray, rgb_rgb, rgb_hsv = self.read_data(self.files[index][1])
        return {
            'nir_gray': nir_gray, 'nir_rgb': nir_rgb, 'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray, 'rgb_rgb': rgb_rgb, 'rgb_hsv': rgb_hsv,
            'nir_path': self.files[index][0], 'rgb_path': self.files[index][1]
        }


if __name__ == '__main__':
    get_data_paths()

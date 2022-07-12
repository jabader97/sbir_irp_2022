#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system and other libraries
import os
import time

import numpy as np
import torch.utils.data as data
import pickle
from PIL import Image, ImageOps
from sklearn import utils as sk_utils
from skimage.transform import warp, AffineTransform


class DataGeneratorPaired(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd, fls_sk, fls_im, clss_sk, clss_im,
                 transforms_sketch=None, transforms_image=None, int2str='', zero_version='', match_class=True,
                 aug=False):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        shuffle = sk_utils.shuffle(fls_im, clss_im)
        self.fls_im = shuffle[0]
        self.clss_im = shuffle[1]
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image
        self.match_class = match_class
        self.aug = aug
        if len(int2str) > 0:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = {}
            for key in int_cid_matrix.keys():
                self.cid_matrix[int2str[key]] = int_cid_matrix[key]
        else:
            self.cid_matrix = ''

    def __getitem__(self, item):
        get_item_time = time.time()
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.match_class:
            sk_options = np.where(self.clss_sk == cls_im)[0]
        else:
            sk_options = np.arange(len(self.clss_sk))
        sk_id = np.random.choice(sk_options)
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd,
                                                     self.fls_sk[sk_id]))).convert(mode='RGB')
        cls_sk = self.clss_sk[sk_id]
        if self.aug:
            im = self.random_transform(im)
            sk = self.random_transform(sk)
        if self.transforms_image is not None:
            im = self.transforms_image(im)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
        if len(self.cid_matrix) > 0:
            mask_im = self.cid_matrix[cls_im]
            mask_sk = self.cid_matrix[cls_sk]
            return sk, im, cls_sk, cls_im, mask_sk, mask_im, (time.time() - get_item_time)
        return sk, im, cls_sk, cls_im, (time.time() - get_item_time)

    def __len__(self):
        return len(self.clss_im)

    def get_weights(self):
        weights = np.zeros(self.clss_im.shape[0])
        uniq_clss = np.unique(self.clss_im)
        for cls in uniq_clss:
            idx = np.where(self.clss_im == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights

    def random_transform(self, img):
        img = np.array(img)
        if np.random.random() < 0.5:
            img = img[:, ::-1, :]

        if np.random.random() < 0.5:
            sx = np.random.uniform(0.7, 1.3)
            sy = np.random.uniform(0.7, 1.3)
        else:
            sx = 1.0
            sy = 1.0

        if np.random.random() < 0.5:
            rx = np.random.uniform(-30.0 * 2.0 * np.pi / 360.0, +30.0 * 2.0 * np.pi / 360.0)
        else:
            rx = 0.0

        if np.random.random() < 0.5:
            tx = np.random.uniform(-10, 10)
            ty = np.random.uniform(-10, 10)
        else:
            tx = 0.0
            ty = 0.0

        aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx, ty))
        img_aug = warp(img, aftrans.inverse, preserve_range=True).astype('uint8')

        return Image.fromarray(img_aug)


class DataGeneratorSketch(data.Dataset):
    def __init__(self, dataset, root, sketch_dir, sketch_sd, fls_sk, clss_sk, transforms=None, int2str='',
                 zero_version=''):
        self.dataset = dataset
        self.root = root
        self.sketch_dir = sketch_dir
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        self.transforms = transforms
        if len(int2str) > 0:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = {}
            for key in int_cid_matrix.keys():
                self.cid_matrix[int2str[key]] = int_cid_matrix[key]
        else:
            self.cid_matrix = ''

    def __getitem__(self, item):
        get_item_time = time.time()
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        cls_sk = self.clss_sk[item]
        if self.transforms is not None:
            sk = self.transforms(sk)
        if len(self.cid_matrix) > 0:
            mask = self.cid_matrix[cls_sk]
            return sk, cls_sk, mask, (time.time() - get_item_time)

        return sk, cls_sk, (time.time() - get_item_time)

    def __len__(self):
        return len(self.fls_sk)

    def get_weights(self):
        weights = np.zeros(self.clss_sk.shape[0])
        uniq_clss = np.unique(self.clss_sk)
        for cls in uniq_clss:
            idx = np.where(self.clss_sk == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


class DataGeneratorImage(data.Dataset):
    def __init__(self, dataset, root, photo_dir, photo_sd, fls_im, clss_im, transforms=None, int2str='',
                 zero_version=''):

        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.photo_sd = photo_sd
        self.fls_im = fls_im
        self.clss_im = clss_im
        self.transforms = transforms
        if len(int2str) > 0:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = {}
            for key in int_cid_matrix.keys():
                self.cid_matrix[int2str[key]] = int_cid_matrix[key]
        else:
            self.cid_matrix = ''

    def __getitem__(self, item):
        get_item_time = time.time()
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.transforms is not None:
            im = self.transforms(im)
        if len(self.cid_matrix) > 0:
            mask = self.cid_matrix[cls_im]
            return im, cls_im, mask, (time.time() - get_item_time)
        return im, cls_im, (time.time() - get_item_time)

    def __len__(self):
        return len(self.fls_im)

    def get_weights(self):
        weights = np.zeros(self.clss_im.shape[0])
        uniq_clss = np.unique(self.clss_im)
        for cls in uniq_clss:
            idx = np.where(self.clss_im == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights

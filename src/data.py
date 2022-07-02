#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system and other libraries
import os
import time

import numpy as np
import torch.utils.data as data
import pickle
from PIL import Image, ImageOps


class DataGeneratorPaired(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd, fls_sk, fls_im, clss,
                 transforms_sketch=None, transforms_image=None, cid_mask=False, zero_version=''):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.fls_im = fls_im
        self.clss = clss
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = self.get_class_name_from_int(int_cid_matrix, zero_version)

    def __getitem__(self, item):
        get_item_time = time.time()
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls = self.clss[item]
        if self.transforms_image is not None:
            im = self.transforms_image(im)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
        if self.cid_mask:
            mask = self.cid_matrix[cls]
            return sk, im, cls, mask, (time.time() - get_item_time)
        return sk, im, cls, (time.time() - get_item_time)

    def __len__(self):
        return len(self.clss)

    def get_weights(self):
        weights = np.zeros(self.clss.shape[0])
        uniq_clss = np.unique(self.clss)
        for cls in uniq_clss:
            idx = np.where(self.clss == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights

    def get_class_name_from_int(self, int_cid_matrix, zero_version):
        # load the conversion file
        int_to_class_path = os.path.join(self.root, zero_version, 'cname_cid.txt')
        int_to_class_dict = {}
        with open(int_to_class_path) as f:
            for line in f:
                key, val = self.process_str(line)
                int_to_class_dict[int(key)] = val
        # convert the classes
        class_cid_matrix = {}
        for key in int_cid_matrix.keys():
            class_cid_matrix[int_to_class_dict[key]] = int_cid_matrix[key]
        return class_cid_matrix

    @staticmethod
    def process_str(line):
        contents = line.split()
        key = contents[-1]
        name = []
        for c in contents[0:-1]:
            name += c.split('-')
        join_symbol = '_'
        val = join_symbol.join(name)
        return key, val


class DataGeneratorSketch(data.Dataset):
    def __init__(self, dataset, root, sketch_dir, sketch_sd, fls_sk, clss_sk, transforms=None, cid_mask=False, zero_version=''):
        self.dataset = dataset
        self.root = root
        self.sketch_dir = sketch_dir
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        self.transforms = transforms
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = self.get_class_name_from_int(int_cid_matrix, zero_version)

    def __getitem__(self, item):
        get_item_time = time.time()
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        cls_sk = self.clss_sk[item]
        if self.transforms is not None:
            sk = self.transforms(sk)
        if self.cid_mask:
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

    def get_class_name_from_int(self, int_cid_matrix, zero_version):
        # load the conversion file
        int_to_class_path = os.path.join(self.root, zero_version, 'cname_cid.txt')
        int_to_class_dict = {}
        with open(int_to_class_path) as f:
            for line in f:
                key, val = self.process_str(line)
                int_to_class_dict[int(key)] = val
        # convert the classes
        class_cid_matrix = {}
        for key in int_cid_matrix.keys():
            class_cid_matrix[int_to_class_dict[key]] = int_cid_matrix[key]
        return class_cid_matrix

    @staticmethod
    def process_str(line):
        contents = line.split()
        key = contents[-1]
        name = []
        for c in contents[0:-1]:
            name += c.split('-')
        join_symbol = '_'
        val = join_symbol.join(name)
        return key, val


class DataGeneratorImage(data.Dataset):
    def __init__(self, dataset, root, photo_dir, photo_sd, fls_im, clss_im, transforms=None, cid_mask=False, zero_version=''):

        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.photo_sd = photo_sd
        self.fls_im = fls_im
        self.clss_im = clss_im
        self.transforms = transforms
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                int_cid_matrix = pickle.load(fh)
            self.cid_matrix = self.get_class_name_from_int(int_cid_matrix, zero_version)

    def __getitem__(self, item):
        get_item_time = time.time()
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.transforms is not None:
            im = self.transforms(im)
        if self.cid_mask:
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

    def get_class_name_from_int(self, int_cid_matrix, zero_version):
        # load the conversion file
        int_to_class_path = os.path.join(self.root, zero_version, 'cname_cid.txt')
        int_to_class_dict = {}
        with open(int_to_class_path) as f:
            for line in f:
                key, val = self.process_str(line)
                int_to_class_dict[int(key)] = val
        # convert the classes
        class_cid_matrix = {}
        for key in int_cid_matrix.keys():
            class_cid_matrix[int_to_class_dict[key]] = int_cid_matrix[key]
        return class_cid_matrix

    @staticmethod
    def process_str(line):
        contents = line.split()
        key = contents[-1]
        name = []
        for c in contents[0:-1]:
            name += c.split('-')
        join_symbol = '_'
        val = join_symbol.join(name)
        return key, val


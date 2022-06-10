#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import argparse
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from options import Options
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from data import DataGeneratorPaired, DataGeneratorSketch, DataGeneratorImage
from PIL import Image

import torch

np.random.seed(0)


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def create_dict_texts(texts):
    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def read_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config.read(os.path.join(cur_path, 'config.ini'))
    host = socket.gethostname()
    return config[host]


def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)


def prec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if k is not None:
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    return pr


def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    return re


def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 10)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 10)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    return aps


def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def get_coarse_grained_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True):

    idx_im_ret = np.array([], dtype=np.int)
    idx_sk_ret = np.array([], dtype=np.int)
    clss_im = np.array([f.split('/')[-2] for f in fls_im])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])
    names_sk = np.array([f.split('-')[0] for f in fls_sk])
    for i, c in enumerate(classes):
        idx1 = np.where(clss_im == c)[0]
        idx2 = np.where(clss_sk == c)[0]
        if set_type == 'train':
            idx_cp = list(itertools.product(idx1, idx2))
            if len(idx_cp) > 100000:
                random.seed(i)
                idx_cp = random.sample(idx_cp, 100000)
            idx1, idx2 = zip(*idx_cp)
        else:
            # remove duplicate sketches
            if filter_sketch:
                names_sk_tmp = names_sk[idx2]
                idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                idx2 = idx2[idx_tmp]
        idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
        idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)

    return idx_im_ret, idx_sk_ret


def load_files_sketchy_zeroshot(root_path, split_eccv_2018=False, filter_sketch=False, photo_dir='photo',
                                sketch_dir='sketch', photo_sd='tx_000000000000', sketch_sd='tx_000000000000_ready',
                                model='', zero_version=''):
    # paths of sketch and image
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # all the image and sketch files together with classes and core names
    fls_sk = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_sk, '*/*.png'))])
    fls_im = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_im, '*/*.jpg'))])

    # classes for image and sketch
    clss_sk = np.array([f.split('/')[0] for f in fls_sk])
    clss_im = np.array([f.split('/')[0] for f in fls_im])

    # all the unique classes
    classes = sorted(os.listdir(path_sk))
    if classes[0] == '.DS_Store':
        classes.pop(0)
    # get test classes from SAKE split
    test_path = os.path.join(root_path, zero_version, 'cname_cid_zero.txt')
    te_classes = []
    with open(test_path) as f:
        for line in f:
            join_symbol = '_'
            class_orig = line.split()[0]
            class_reformatted = join_symbol.join(class_orig.split('-'))
            if not class_reformatted in te_classes:
                te_classes.append(class_reformatted)
    te_classes = np.asarray(te_classes)
    # get train and validation classes
    if model == 'sake':
        # SAKE uses splits from zeroshot files. Note this implementation is a bit different
        # from the original paper: this splits the classes into train/validation, the
        # original paper split the images in each class
        train_path = os.path.join(root_path, zero_version, 'cname_cid.txt')
        tr_and_va_classes = []
        with open(train_path) as f:
            for line in f:
                join_symbol = '_'
                class_orig = line.split()[0]
                class_reformatted = join_symbol.join(class_orig.split('-'))
                if class_reformatted not in tr_and_va_classes:
                    tr_and_va_classes.append(class_reformatted)
        tr_classes = np.random.choice(tr_and_va_classes, int(0.93 * len(tr_and_va_classes)), replace=False)
        va_classes = np.setdiff1d(tr_and_va_classes, tr_classes)

        tr_classes = np.asarray(tr_classes)
        va_classes = np.asarray(va_classes)
    elif split_eccv_2018:
        # According to Yelamarthi et al., "A Zero-Shot Framework for Sketch Based Image Retrieval", ECCV 2018.
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(cur_path, "test_classes_eccv_2018.txt")) as fp:
            te_classes = fp.read().splitlines()
            va_classes = te_classes
            tr_classes = np.setdiff1d(classes, np.union1d(te_classes, va_classes))
    else:
        # Make train and validation splits randomly
        tr_classes = np.random.choice(np.setdiff1d(classes, te_classes), int(0.75 * len(classes)), replace=False)
        va_classes = np.setdiff1d(classes, np.union1d(tr_classes, te_classes))
        # tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
        # va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.1 * len(classes)), replace=False)
        # te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train',
                                                      filter_sketch=filter_sketch)
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid',
                                                      filter_sketch=filter_sketch)
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test',
                                                      filter_sketch=filter_sketch)

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]

    return splits


def load_files_tuberlin_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd='',
                                 model='', zero_version=''):

    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg')) + glob.glob(os.path.join(path_im, '*', '*.JPEG'))
    fls_im = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im])
    clss_im = np.array([f.split('/')[-2] for f in fls_im])

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])

    # all the unique classes
    classes = np.unique(clss_im)
    join_symbol = '_'
    for i, c in enumerate(classes):
        classes[i] = join_symbol.join(c.split('-'))

    np.random.seed(0)
    # get test classes from SAKE split
    test_path = os.path.join(root_path, zero_version, 'png_ready_filelist_zero.txt')
    te_classes = []
    with open(test_path) as f:
        for line in f:
            content = line.split()[0].split('/')
            if not content[1] in te_classes:
                te_classes.append(content[1])
    te_classes = np.asarray(te_classes)
    # get train and validation classes
    if model == 'sake':
        # SAKE uses splits from zeroshot files. Note this implementation is a bit different
        # from the original paper: this splits the classes into train/validation, the
        # original paper split the images in each class

        train_path = os.path.join(root_path, zero_version, 'png_ready_filelist_train.txt')
        tr_and_va_classes = []
        with open(train_path) as f:
            for line in f:
                cur_class = line.split('/')[1]
                join_symbol = '_'
                class_reformatted = join_symbol.join(cur_class.split('-'))
                class_reformatted = join_symbol.join(class_reformatted.split())
                if class_reformatted not in tr_and_va_classes:
                    tr_and_va_classes.append(class_reformatted)
        tr_classes = np.random.choice(tr_and_va_classes, int(0.93 * len(tr_and_va_classes)), replace=False)
        va_classes = np.setdiff1d(tr_and_va_classes, tr_classes)
        tr_classes = np.asarray(tr_classes)
        va_classes = np.asarray(va_classes)
    else:
        # make train/validation splits randomly
        tr_classes = np.random.choice(np.setdiff1d(classes, te_classes), int(0.80 * len(classes)), replace=False)
        va_classes = np.setdiff1d(classes, np.union1d(tr_classes, te_classes))
        # tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False)
        # va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.06 * len(classes)), replace=False)
        # te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train')
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid')
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test')

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]

    return splits


def save_qualitative_results(root, sketch_dir, sketch_sd, photo_dir, photo_sd, fls_sk, fls_im, dir_op, aps, sim,
                             str_sim, nq=50, nim=20, im_sz=(256, 256), best=False, save_image=False):

    # Set directories according to dataset
    dir_sk = os.path.join(root, sketch_dir, sketch_sd)
    dir_im = os.path.join(root, photo_dir, photo_sd)

    if not os.path.isdir(dir_op):
        os.makedirs(dir_op)
    else:
        clean_folder(dir_op)

    if best:
        ind_sk = np.argsort(-aps)[:nq]
    else:
        np.random.seed(0)
        temp = len(aps)
        ind_sk = np.random.choice(len(aps), nq, replace=False)

    # create a text file for results
    fp = open(os.path.join(dir_op, "Results.txt"), "w")

    for i, isk in enumerate(ind_sk):
        fp.write("{0}, ".format(fls_sk[isk]))
        if save_image:
            sdir_op = os.path.join(dir_op, str(i + 1))
            if not os.path.isdir(sdir_op):
                os.makedirs(sdir_op)
            sk = Image.open(os.path.join(dir_sk, fls_sk[isk])).convert(mode='RGB').resize(im_sz)
            sk.save(os.path.join(sdir_op, fls_sk[isk].split('/')[0] + '.png'))
        ind_im = np.argsort(-sim[isk])[:nim]
        for j, iim in enumerate(ind_im):
            if j < len(ind_im)-1:
                fp.write("{0} {1}, ".format(fls_im[iim], str_sim[isk][iim]))
            else:
                fp.write("{0} {1}".format(fls_im[iim], str_sim[isk][iim]))
            if save_image:
                im = Image.open(os.path.join(dir_im, fls_im[iim])).convert(mode='RGB').resize(im_sz)
                im.save(os.path.join(sdir_op, str(j + 1) + '_' + str(str_sim[isk][iim]) + '.png'))
        fp.write("\n")
    fp.close()


def clean_folder(folder):

    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:
            if os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(e)


def get_args():
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    if args.filter_sketch:
        assert args.dataset == 'Sketchy'
    if args.split_eccv_2018:
        assert args.dataset == 'Sketchy_extended' or args.dataset == 'Sketchy'
    if args.gzs_sbir:
        args.test = True
    # modify the log and check point paths
    args.ds_var = None
    if '_' in args.dataset:
        token = args.dataset.split('_')
        args.dataset = token[0]
        args.ds_var = token[1]
    # sort semantic models
    args.semantic_models = sorted(args.semantic_models)
    return args


def get_paths(args):
    # get paths
    str_aux = ''
    if args.split_eccv_2018:
        str_aux = 'split_eccv_2018'
    if args.gzs_sbir:
        str_aux = os.path.join(str_aux, 'generalized')
    model_name = '+'.join(args.semantic_models)
    root_path = os.path.join(args.path_dataset, args.dataset)
    path_cp = os.path.join(args.path_aux, 'CheckPoints', args.dataset, str_aux, model_name, str(args.dim_out), args.savename)
    path_log = os.path.join(args.path_aux, 'LogFiles', args.dataset, str_aux, model_name, str(args.dim_out), args.model)
    path_results = os.path.join(args.path_aux, 'Results', args.dataset, str_aux, model_name, str(args.dim_out), args.model)
    files_semantic_labels = []
    sem_dim = 0
    for f in args.semantic_models:
        fi = os.path.join(args.path_aux, 'Semantic', args.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]
    args.files_semantic_labels = files_semantic_labels
    args.sem_dim = sem_dim
    args.root_path = root_path
    return path_cp, path_log, path_results


def get_datasets(args):
    # Parameters for transforming the images
    transform_image_list = [transforms.Resize((args.im_sz, args.im_sz)), transforms.ToTensor()]
    transform_sketch_list = [transforms.Resize((args.sk_sz, args.sk_sz)), transforms.ToTensor()]
    if args.normalize:
        immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
        imstd = [0.229, 0.224, 0.225]
        transform_image_list.append(transforms.Normalize(immean, imstd))
        transform_sketch_list.append(transforms.Normalize(immean, imstd))
    transform_image = transforms.Compose(transform_image_list)
    transform_sketch = transforms.Compose(transform_sketch_list)

    # Load the dataset
    print('Loading data...', end='')

    if args.dataset == 'Sketchy':
        if args.ds_var == 'extended':
            photo_dir = 'extended_photo'  # photo or extended_photo
            photo_sd = ''
        else:
            photo_dir = 'photo'
            photo_sd = 'tx_000000000000_ready'
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000_ready'
        splits = load_files_sketchy_zeroshot(root_path=args.root_path, split_eccv_2018=args.split_eccv_2018,
                                             photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
                                             sketch_sd=sketch_sd, model=args.model,
                                             zero_version=args.zero_version)
    elif args.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        splits = load_files_tuberlin_zeroshot(root_path=args.root_path, photo_dir=photo_dir, sketch_dir=sketch_dir,
                                              photo_sd=photo_sd, sketch_sd=sketch_sd, model=args.model,
                                              zero_version=args.zero_version)
    else:
        raise Exception('Wrong dataset.')

    # Combine the valid and test set into test set
    # splits['te_fls_sk'] = np.concatenate((splits['va_fls_sk'], splits['te_fls_sk']), axis=0)
    # splits['te_clss_sk'] = np.concatenate((splits['va_clss_sk'], splits['te_clss_sk']), axis=0)
    # splits['te_fls_im'] = np.concatenate((splits['va_fls_im'], splits['te_fls_im']), axis=0)
    # splits['te_clss_im'] = np.concatenate((splits['va_clss_im'], splits['te_clss_im']), axis=0)

    if args.gzs_sbir:
        perc = 0.2
        _, idx_sk = np.unique(splits['tr_fls_sk'], return_index=True)
        tr_fls_sk_ = splits['tr_fls_sk'][idx_sk]
        tr_clss_sk_ = splits['tr_clss_sk'][idx_sk]
        _, idx_im = np.unique(splits['tr_fls_im'], return_index=True)
        tr_fls_im_ = splits['tr_fls_im'][idx_im]
        tr_clss_im_ = splits['tr_clss_im'][idx_im]
        if args.dataset == 'Sketchy' and args.filter_sketch:
            _, idx_sk = np.unique([f.split('-')[0] for f in tr_fls_sk_], return_index=True)
            tr_fls_sk_ = tr_fls_sk_[idx_sk]
            tr_clss_sk_ = tr_clss_sk_[idx_sk]
        idx_sk = np.sort(np.random.choice(tr_fls_sk_.shape[0], int(perc * splits['te_fls_sk'].shape[0]), replace=False))
        idx_im = np.sort(np.random.choice(tr_fls_im_.shape[0], int(perc * splits['te_fls_im'].shape[0]), replace=False))
        splits['te_fls_sk'] = np.concatenate((tr_fls_sk_[idx_sk], splits['te_fls_sk']), axis=0)
        splits['te_clss_sk'] = np.concatenate((tr_clss_sk_[idx_sk], splits['te_clss_sk']), axis=0)
        splits['te_fls_im'] = np.concatenate((tr_fls_im_[idx_im], splits['te_fls_im']), axis=0)
        splits['te_clss_im'] = np.concatenate((tr_clss_im_[idx_im], splits['te_clss_im']), axis=0)

    # class dictionary
    args.dict_clss = create_dict_texts(splits['tr_clss_im'])
    cid_mask = True if 'sake' in args.model else False
    if 'sake' in args.model:
        data_train_sketch = DataGeneratorSketch(args.dataset, args.root_path, sketch_dir, sketch_sd, splits['tr_fls_sk'],
                                            splits['tr_clss_sk'], transforms=transform_sketch, cid_mask=cid_mask,
                                            zero_version=args.zero_version)
        data_train_image = DataGeneratorImage(args.dataset, args.root_path, photo_dir, photo_sd, splits['tr_fls_im'],
                                          splits['tr_clss_im'], transforms=transform_image, cid_mask=cid_mask,
                                          zero_version=args.zero_version)
        data_train = (data_train_image, data_train_sketch)
    else:
        data_train = DataGeneratorPaired(args.dataset, args.root_path, photo_dir, sketch_dir, photo_sd, sketch_sd,
                                         splits['tr_fls_sk'], splits['tr_fls_im'], splits['tr_clss_im'],
                                         transforms_sketch=transform_sketch, transforms_image=transform_image)
    data_valid_sketch = DataGeneratorSketch(args.dataset, args.root_path, sketch_dir, sketch_sd, splits['va_fls_sk'],
                                            splits['va_clss_sk'], transforms=transform_sketch, cid_mask=False,
                                            zero_version=args.zero_version)
    data_valid_image = DataGeneratorImage(args.dataset, args.root_path, photo_dir, photo_sd, splits['va_fls_im'],
                                          splits['va_clss_im'], transforms=transform_image, cid_mask=False,
                                          zero_version=args.zero_version)
    data_test_sketch = DataGeneratorSketch(args.dataset, args.root_path, sketch_dir, sketch_sd, splits['te_fls_sk'],
                                           splits['te_clss_sk'], transforms=transform_sketch, cid_mask=False,
                                           zero_version=args.zero_version)
    data_test_image = DataGeneratorImage(args.dataset, args.root_path, photo_dir, photo_sd, splits['te_fls_im'],
                                         splits['te_clss_im'], transforms=transform_image, cid_mask=False,
                                         zero_version=args.zero_version)
    print('Done')

    if not isinstance(data_train, DataGeneratorPaired):
        num_samples = args.epoch_size * args.batch_size
        train_sampler_image = WeightedRandomSampler(data_train[0].get_weights(),  num_samples=num_samples,
                                                    replacement=True)
        train_sampler_sketch = WeightedRandomSampler(data_train[1].get_weights(), num_samples=num_samples,
                                                     replacement=True)
        train_loader_image = DataLoader(dataset=data_train[0], batch_size=args.batch_size, sampler=train_sampler_image,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        train_loader_sketch = DataLoader(dataset=data_train[1], batch_size=args.batch_size, sampler=train_sampler_sketch,
                                         num_workers=args.num_workers, pin_memory=True)
        train_loader = (train_loader_image, train_loader_sketch)
    else:
        train_sampler = WeightedRandomSampler(data_train.get_weights(), num_samples=args.epoch_size * args.batch_size,
                                              replacement=True)
        train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True)

    # PyTorch valid loader for sketch
    valid_loader_sketch = DataLoader(dataset=data_valid_sketch, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
    # PyTorch valid loader for image
    valid_loader_image = DataLoader(dataset=data_valid_image, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader_sketch, valid_loader_image, test_loader_sketch, test_loader_image, \
           photo_dir, sketch_dir, splits, photo_sd, sketch_sd


def get_params(args):
    # Model parameters
    params_model = dict()
    params_model['model'] = args.model
    params_model['epochs'] = args.epochs
    # Paths to pre-trained sketch and image models
    path_sketch_model = os.path.join(args.path_aux, 'CheckPoints', args.dataset, 'sketch')
    path_image_model = os.path.join(args.path_aux, 'CheckPoints', args.dataset, 'image')
    params_model['path_sketch_model'] = path_sketch_model
    params_model['path_image_model'] = path_image_model
    params_model['root_path'] = args.root_path
    # Dimensions
    params_model['dim_out'] = args.dim_out
    params_model['sem_dim'] = args.sem_dim
    # Number of classes
    params_model['num_clss'] = len(args.dict_clss)
    # Weight (on losses) parameters
    params_model['lambda_se'] = args.lambda_se
    params_model['lambda_im'] = args.lambda_im
    params_model['lambda_sk'] = args.lambda_sk
    params_model['lambda_gen_cyc'] = args.lambda_gen_cyc
    params_model['lambda_gen_adv'] = args.lambda_gen_adv
    params_model['lambda_gen_cls'] = args.lambda_gen_cls
    params_model['lambda_gen_reg'] = args.lambda_gen_reg
    params_model['lambda_disc_se'] = args.lambda_disc_se
    params_model['lambda_disc_sk'] = args.lambda_disc_sk
    params_model['lambda_disc_im'] = args.lambda_disc_im
    params_model['lambda_regular'] = args.lambda_regular
    # Optimizers' parameters
    params_model['lr'] = args.lr
    params_model['momentum'] = args.momentum
    params_model['milestones'] = args.milestones
    params_model['gamma'] = args.gamma
    # Files with semantic labels
    params_model['files_semantic_labels'] = args.files_semantic_labels
    # Class dictionary
    params_model['dict_clss'] = args.dict_clss
    # SAKE specific parameters
    if "sake" in args.model:
        params_model['arch'] = args.arch
        params_model['num_hashing'] = args.num_hashing
        params_model['num_classes'] = args.num_classes
        params_model['freeze_features'] = args.freeze_features
        params_model['ems_loss'] = args.ems_loss
        params_model['kd_lambda'] = args.kd_lambda
        params_model['kdneg_lambda'] = args.kdneg_lambda
        params_model['sake_lambda'] = args.sake_lambda
        params_model['weight_decay'] = args.weight_decay
        params_model['zero_version'] = args.zero_version
    return params_model

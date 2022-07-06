#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np
from scipy.spatial.distance import cdist

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn

# user defined
import itq
import utils
from logger import AverageMeter
from models import get_model
from torch.utils.data import DataLoader


def main():

    # Parse options
    args = utils.get_args()
    np.random.seed(args.seed)
    args.test = True
    path_cp, _, path_results = utils.get_paths(args)

    print('Checkpoint path: {}'.format(path_cp))
    print('Result path: {}'.format(path_results))

    _, _, _, test_loader_sketch, test_loader_image, photo_dir, sketch_dir, splits, photo_sd, \
        sketch_sd = utils.get_datasets(args)

    params_model = utils.get_params(args)

    # Model
    model = get_model(params_model)

    cudnn.benchmark = True

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        model = model.cuda()
    print('Done')

    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f})".format(best_model_file, epoch, best_map))
        print('***Test***')
        valid_data = validate(test_loader_sketch, test_loader_image, model, epoch, args)
        print('Results on test set: mAP@all = {1:.4f}, Prec@100 = {0:.4f}, mAP@200 = {3:.4f}, Prec@200 = {2:.4f}, '
              'Time = {4:.6f} || mAP@all (binary) = {6:.4f}, Prec@100 (binary) = {5:.4f}, mAP@200 (binary) = {8:.4f}, '
              'Prec@200 (binary) = {7:.4f}, Time (binary) = {9:.6f} '
              .format(valid_data['prec@100'], np.mean(valid_data['aps@all']), valid_data['prec@200'],
                      np.mean(valid_data['aps@200']), valid_data['time_euc'], valid_data['prec@100_bin'],
                      np.mean(valid_data['aps@all_bin']), valid_data['prec@200_bin'], np.mean(valid_data['aps@200_bin'])
                      , valid_data['time_bin']))
        print('Saving qualitative results...', end='')
        path_qualitative_results = os.path.join(path_results, 'qualitative_results')
        utils.save_qualitative_results(args.root_path, sketch_dir, sketch_sd, photo_dir, photo_sd, splits['te_fls_sk'],
                                       splits['te_fls_im'], path_qualitative_results, valid_data['aps@all'],
                                       valid_data['sim_euc'], valid_data['str_sim'], save_image=args.save_image_results,
                                       nq=args.number_qualit_results, best=args.save_best_results)
        print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def get_topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(train_loader, model, epoch, args):
    acc_embedding_time = time.time()
    for i, info in enumerate(train_loader):
        if 'sake' in args.model:
            sk, im, cl_sk, cl_im, _, _, ti = info
        else:
            sk, im, cl_sk, cl_im, ti = info
        if torch.cuda.is_available():
            sk = sk.cuda()
        sk_pr = model.get_sketch_prediction(sk)
        if i == 0:
            acc_sk_pr = sk_pr.cpu().data.numpy()
            acc_cls_sk = cl_sk
        else:
            acc_sk_pr = np.concatenate((acc_sk_pr, sk_pr.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cl_sk), axis=0)

        if torch.cuda.is_available():
            im = im.cuda()
        im_pr = model.get_image_prediction(im)
        if i == 0:
            acc_im_pr = im_pr.cpu().data.numpy()
            acc_cls_im = cl_im
        else:
            acc_im_pr = np.concatenate((acc_im_pr, im_pr.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cl_im), axis=0)

    acc_embedding_time = time.time() - acc_embedding_time
    time_info = {'acc_embedding_time': acc_embedding_time}

    accuracy_time = time.time()
    predicted = np.concatenate((acc_im_pr, acc_sk_pr), axis=0)
    acc_cls = utils.numeric_classes(np.concatenate((acc_cls_im, acc_cls_sk), axis=0), args.dict_clss_str2int)
    acc1, acc5 = get_topk_accuracy(torch.from_numpy(predicted), torch.from_numpy(acc_cls), topk=(1, 5))
    accuracy_time = time.time() - accuracy_time
    stats = {'acc1': acc1, 'acc5': acc5}
    time_info['accuracy_time'] = accuracy_time

    return stats, time_info


def validate(valid_loader_sketch, valid_loader_image, model, epoch, args):
    validate_setup_time = time.time()
    valid_get_sketch_time = AverageMeter()
    valid_get_image_time = AverageMeter()

    # Switch to test mode
    model.eval()

    batch_time = AverageMeter()

    # Start counting time
    time_start = time.time()
    validate_setup_time = time.time() - validate_setup_time
    sketch_embedding_time = time.time()
    for i, (sk, cls_sk, ti) in enumerate(valid_loader_sketch):
        valid_get_sketch_time.update(ti.sum())
        if torch.cuda.is_available():
            sk = sk.cuda()

        # Sketch embedding into a semantic space
        sk_em = model.get_sketch_embeddings(sk)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
            acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Test][Sketch] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_sketch), batch_time=batch_time))
        # todo remove
        if i > 2:
            break
        # ==========

    sketch_embedding_time = time.time() - sketch_embedding_time
    image_embedding_time = time.time()
    for i, (im, cls_im, ti) in enumerate(valid_loader_image):
        valid_get_image_time.update(ti.sum())

        if torch.cuda.is_available():
            im = im.cuda()

        # Image embedding into a semantic space
        im_em = model.get_image_embeddings(im)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
            acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Test][Image] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_image), batch_time=batch_time))
        # todo remove
        if i > 2:
            break
        # ==========

    image_embedding_time = time.time() - image_embedding_time
    # Compute mAP
    print('Computing evaluation metrics...', end='')

    # Compute similarity
    similarity_time = time.time()
    t = time.time()
    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))
    time_euc = (time.time() - t) / acc_cls_sk.shape[0]
    similarity_time = time.time() - similarity_time

    # binary encoding with ITQ
    binary_encoding_time = time.time()
    acc_sk_em_bin, acc_im_em_bin = itq.compressITQ(acc_sk_em, acc_im_em)
    t = time.time()
    sim_bin = np.exp(-cdist(acc_sk_em_bin, acc_im_em_bin, metric='hamming'))
    time_bin = (time.time() - t) / acc_cls_sk.shape[0]
    binary_encoding_time = time.time() - binary_encoding_time

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall_time = time.time()
    apsall = utils.apsak(sim_euc, str_sim)
    apsall_time = time.time() - apsall_time
    aps200_time = time.time()
    aps200 = utils.apsak(sim_euc, str_sim, k=200)
    aps200_time = time.time() - aps200_time
    prec100_time = time.time()
    prec100, _ = utils.precak(sim_euc, str_sim, k=100)
    prec100_time = time.time() - prec100_time
    prec200_time = time.time()
    prec200, _ = utils.precak(sim_euc, str_sim, k=200)
    prec200_time = time.time() - prec200_time

    apsall_bin_time = time.time()
    apsall_bin = utils.apsak(sim_bin, str_sim)
    apsall_bin_time = time.time() - apsall_bin_time
    aps200_bin_time = time.time()
    aps200_bin = utils.apsak(sim_bin, str_sim, k=200)
    aps200_bin_time = time.time() - aps200_bin_time
    prec100_bin_time = time.time()
    prec100_bin, _ = utils.precak(sim_bin, str_sim, k=100)
    prec100_bin_time = time.time() - prec100_bin_time
    prec200_bin_time = time.time()
    prec200_bin, _ = utils.precak(sim_bin, str_sim, k=200)
    prec200_bin_time = time.time() - prec200_bin_time

    valid_data = {'aps@all': apsall, 'aps@200': aps200, 'prec@100': prec100, 'prec@200': prec200, 'sim_euc': sim_euc,
                  'time_euc': time_euc, 'aps@all_bin': apsall_bin, 'aps@200_bin': aps200_bin, 'prec@100_bin':
                      prec100_bin, 'prec@200_bin': prec200_bin, 'sim_bin': sim_bin, 'time_bin': time_bin, 'str_sim':
                      str_sim}
    time_info = {'sketch_embedding_time': sketch_embedding_time, 'image_embedding_time': image_embedding_time,
                 'similarity_time': similarity_time, 'binary_encoding_time': binary_encoding_time,
                 'apsall_time': apsall_time, 'aps200_time': aps200_time, 'prec100_time': prec100_time,
                 'prec200_time': prec200_time, 'apsall_bin_time': apsall_bin_time, 'aps200_bin_time': aps200_bin_time,
                 'prec100_bin_time': prec100_bin_time, 'prec200_bin_time': prec200_bin_time,
                 'validate_setup_time': validate_setup_time, 'valid_get_sketch_time': valid_get_sketch_time.avg,
                 'valid_get_image_time': valid_get_image_time.avg}

    print('Done')

    return valid_data, time_info


if __name__ == '__main__':
    main()
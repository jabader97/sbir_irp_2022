#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
import time

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn

# user defined
import utils
from logger import Logger, AverageMeter
from test import validate, accuracy
from models import get_model


def main():

    args = utils.get_args()
    np.random.seed(args.seed)
    if args.savename == 'group_plus_seed':
        if args.log_online:
            args.savename = args.group + '_s{}'.format(args.seed)
        else:
            args.savename = ''
    path_cp, path_log, path_results = utils.get_paths(args)

    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_results))

    dataset_creation_time = time.time()
    train_loader, train_loader_sketch, valid_loader_sketch, valid_loader_image, test_loader_sketch, test_loader_image,\
        photo_dir, sketch_dir, splits, photo_sd, sketch_sd = utils.get_datasets(args)
    dataset_creation_time = time.time() - dataset_creation_time
    params_model = utils.get_params(args)

    # If wandb-logging is turned on, initialize the wandb-run here:
    if args.log_online:
        import wandb
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        save_path = os.path.join(args.path_aux, 'CheckPoints', 'wandb')
        wandb.init(project=args.project, group=args.group, name=args.savename, dir=save_path,
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(params_model)

    # Model
    model_load_time = time.time()
    model = get_model(params_model)
    model_load_time = time.time() - model_load_time

    cudnn.benchmark = True

    # Logger
    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 and torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        model = model.cuda()
    print('Done')

    if args.log_online:
        wandb.log({'dataset_creation_time': dataset_creation_time, 'model_load_time': model_load_time})

    best_map = 0
    early_stop_counter = 0

    # Epoch for loop
    if not args.test:
        print('***Train***')
        for epoch in range(args.epochs):

            # train on training set
            loss_per_epoch_time = time.time()
            model.train()
            losses, time_info = model.train_once(train_loader, epoch, args)
            time_info['train_one_epoch_time'] = time.time() - loss_per_epoch_time

            model.scheduler_step(epoch)

            # evaluate on validation set, map_ since map is already there
            print('***Validation***')
            validation_per_epoch_time = time.time()
            valid_data, valid_time_info = validate(valid_loader_sketch, valid_loader_image, model, epoch, args)
            time_info['validation_per_epoch_time'] = time.time() - validation_per_epoch_time
            map_ = np.mean(valid_data['aps@all'])

            print('mAP@all on validation set after {0} epochs: {1:.4f} (real), {2:.4f} (binary)'
                .format(epoch + 1, map_, np.mean(valid_data['aps@all_bin'])))

            if args.accuracy:
                accuracy_per_epoch_time = time.time()
                model.eval()
                acc_data, acc_time_info = accuracy(train_loader_sketch, model, epoch, args)
                time_info['accuracy_per_epoch_time'] = time.time() - accuracy_per_epoch_time

            if args.log_online:
                for key in valid_data.keys():
                    valid_data[key] = np.mean(valid_data[key])
                if args.accuracy:
                    for key in acc_data.keys():
                        valid_data[key] = acc_data[key]
                wandb.log(valid_data)

            del valid_data

            if map_ > best_map:
                best_map = map_
                early_stop_counter = 0
                save_checkpoint_time = time.time()
                utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_map':
                    best_map}, directory=path_cp)
                time_info['save_checkpoint_time'] = time.time() - save_checkpoint_time
            else:
                if args.early_stop == early_stop_counter:
                    break
                early_stop_counter += 1

            # Logger step
            model.add_to_log(logger, losses)
            logger.add_scalar('mean average precision', map_)
            logger.step()

            if args.log_online:
                for key in valid_time_info.keys():
                    time_info[key] = valid_time_info[key]
                if args.accuracy:
                    for key in acc_time_info.keys():
                        time_info[key] = acc_time_info[key]
                wandb.log(time_info)

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
        model.eval()
        valid_data, time_info = validate(test_loader_sketch, test_loader_image, model, epoch, args)

        if args.log_online:
            valid_data_test = {}
            for key in valid_data.keys():
                valid_data_test["test_" + key] = np.mean(valid_data[key])
            wandb.log(valid_data_test)

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


if __name__ == '__main__':
    main()

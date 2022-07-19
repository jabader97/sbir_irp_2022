import torch.cuda
import torch.nn as nn
from torch import ones_like, mul, sum, randn, mm, transpose
import torch.nn.functional as F
import math
import torch.backends.cudnn as cudnn
import numpy as np
import os, time
from logger import AverageMeter
from utils import create_dict_texts
from architectures import get_model


class EMSLoss(nn.Module):
    def __init__(self, m=4):
        super(EMSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.m = m

    def forward(self, inputs, targets):
        mmatrix = ones_like(inputs)
        for ii in range(inputs.size()[0]):
            mmatrix[ii, int(targets[ii])] = self.m

        inputs_m = mul(inputs, mmatrix)
        return self.criterion(inputs_m, targets)


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask=None, mask_pos=None):
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)

        if mask_pos is not None:
            target_logits = target_logits + mask_pos

        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = sum(mul(log_likelihood, F.softmax(target_logits, dim=1))) / sample_num
        else:
            sample_num = sum(mask)
            loss = sum(mul(mul(log_likelihood, F.softmax(target_logits, dim=1)), mask)) / sample_num

        return loss


class SAKE(nn.Module):
    def __init__(self, params_model):
        super(SAKE, self).__init__()
        self.model = get_model(params_model['student_arch'] + "_kd", params_model['num_clss'], 'sake',
                               hashing_dim=params_model['num_hashing'], freeze_features=params_model['freeze_features'],
                               ems=params_model['ems_loss'])
        self.model = nn.DataParallel(self.model)
        self.model_t = get_model(params_model['teacher_arch'], params_model['num_clss'], 'sake',
                               hashing_dim=params_model['num_hashing'], freeze_features=params_model['freeze_features'],
                               ems=params_model['ems_loss'])
        self.model_t = nn.DataParallel(self.model_t)
        if params_model['ems_loss']:
            print("**************  Use EMS Loss!")
            self.curr_m = 1
            self.criterion_train = EMSLoss(self.curr_m)
        else:
            self.criterion_train = nn.CrossEntropyLoss()
        self.criterion_train_kd = SoftCrossEntropy()
        self.lr = params_model['lr']
        self.epochs = params_model['epochs']
        self.root = params_model['root_path']
        self.zero_version = params_model['zero_version']
        torch.manual_seed(params_model['seed'])

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model_t = self.model_t.cuda()
            self.criterion_train = self.criterion_train.cuda()
            self.criterion_train_kd = self.criterion_train_kd.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=params_model['weight_decay'])

        self.class_to_int_dict, _ = create_dict_texts(self.root, self.zero_version)
        self.sake_lambda = params_model['sake_lambda']
        cudnn.benchmark = True

    def scheduler_step(self, epoch):
        lr = self.lr * math.pow(0.001, float(epoch) / self.epochs)
        print('epoch: {}, lr: {}'.format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_once(self, train_loader, epoch, args):
        self.model_t.eval()
        train_setup_time = time.time()
        losses = AverageMeter()
        losses_kd = AverageMeter()
        get_item_time = AverageMeter()
        reformat_data_time = AverageMeter()
        forward_pass_time = AverageMeter()
        forward_pass_s_time = AverageMeter()
        forward_pass_t_time = AverageMeter()
        loss_time = AverageMeter()
        backward_pass_time = AverageMeter()

        if args.ems_loss:
            if epoch in [20, 25]:
                new_m = self.curr_m * 2
                print("update m at epoch {}: from {} to {}".format(epoch, self.curr_m, new_m))
                self.criterion_train = EMSLoss(new_m)
                if torch.cuda.is_available():
                    self.criterion_train = self.criterion_train.cuda()
                self.curr_m = new_m
        train_setup_time = time.time() - train_setup_time
        for i, (input_ext, input, target_ext, target, cid_mask_ext, cid_mask, ti) in enumerate(train_loader):
            get_item_time.update(ti.sum())
            one_loop_time_start = time.time()
            target = torch.Tensor([int(self.class_to_int_dict[t]) for t in target])
            target_ext = torch.Tensor([int(self.class_to_int_dict[t]) for t in target_ext])
            input_all = torch.cat([input, input_ext], dim=0)
            tag_zeros = torch.zeros(input.size()[0], 1)
            tag_ones = torch.ones(input_ext.size()[0], 1)
            tag_all = torch.cat([tag_zeros, tag_ones], dim=0)
            target_all = torch.cat([target, target_ext], dim=0)
            cid_mask_all = torch.cat([cid_mask, cid_mask_ext], dim=0)

            shuffle_idx = np.arange(input_all.size()[0])
            np.random.shuffle(shuffle_idx)
            input_all = input_all[shuffle_idx]
            tag_all = tag_all[shuffle_idx]
            target_all = target_all[shuffle_idx]
            cid_mask_all = cid_mask_all[shuffle_idx].float()

            target_all = target_all.type(torch.LongTensor).view(-1, )
            if torch.cuda.is_available():
                input_all = input_all.cuda()
                tag_all = tag_all.cuda()
                target_all = target_all.cuda()
                cid_mask_all = cid_mask_all.cuda()
            reformat_data_time.update(time.time() - one_loop_time_start)
            forward_pass_s_time_start = time.time()
            output, output_kd = self.model(input_all, tag_all)
            forward_pass_s_time.update(time.time() - forward_pass_s_time_start)
            with torch.no_grad():
                forward_pass_t_time_start = time.time()
                output_t = self.model_t(input_all, tag_all)
                forward_pass_t_time.update(time.time() - forward_pass_t_time_start)
                forward_pass_time.update(time.time() - forward_pass_s_time_start)
            loss_time_start = time.time()
            loss = self.criterion_train(output, target_all)
            loss_kd = self.criterion_train_kd(output_kd, output_t * args.kd_lambda, tag_all, cid_mask_all * args.kdneg_lambda)
            losses.update(loss.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))
            loss_time.update(time.time() - loss_time_start)
            # compute gradient and take step
            self.optimizer.zero_grad()
            loss_total = loss + self.sake_lambda * loss_kd
            backward_pass_time_start = time.time()
            loss_total.backward()
            backward_pass_time.update(time.time() - backward_pass_time_start)
            self.optimizer.step()

            if (i + 1) % args.log_interval == 0:
                print('[Train] Epoch: [{0}][{1}/{2}]\t'
                      'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      'KD Loss {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                      'Loss total {loss_total:.4f} ({loss_total:.4f})\t'
                      .format(epoch + 1, i + 1, len(train_loader), losses=losses,
                              losses_kd=losses_kd, loss_total=loss_total))

        loss_total = losses.avg + self.sake_lambda * losses_kd.avg
        loss_stats = {'losses': losses, 'losses_kd': losses_kd, 'loss_total': loss_total}
        time_stats = {'train_setup_time': train_setup_time, 'get_item_time': get_item_time.avg,
                      'reformat_data_time': reformat_data_time.avg, 'forward_pass_s_time': forward_pass_s_time.avg,
                      'forward_pass_t_time': forward_pass_t_time.avg, 'loss_time': loss_time.avg,
                      'backward_pass_time': backward_pass_time.avg, 'forward_pass_time': forward_pass_time.avg}
        return loss_stats, time_stats

    def get_sketch_embeddings(self, sk):
        tag = torch.zeros(sk.size()[0], 1)
        if torch.cuda.is_available():
            tag = tag.cuda()
        features = self.model.module.original_model.features(sk, tag)
        features = self.model.module.original_model.hashing(features)
        features = F.normalize(features)
        return features

    def get_image_embeddings(self, im):
        tag = torch.ones(im.size()[0], 1)
        if torch.cuda.is_available():
            tag = tag.cuda()
        features = self.model.module.original_model.features(im, tag)
        features = self.model.module.original_model.hashing(features)
        features = F.normalize(features)
        return features

    def get_sketch_prediction(self, sk):
        tag = torch.zeros(sk.size()[0], 1)
        if torch.cuda.is_available():
            tag = tag.cuda()
        prediction, _ = self.model(sk, tag)
        return prediction

    def get_image_prediction(self, im):
        tag = torch.ones(im.size()[0], 1)
        if torch.cuda.is_available():
            tag = tag.cuda()
        prediction, _ = self.model(im, tag)
        return prediction


    @staticmethod
    def add_to_log(logger, losses):
        logger.add_scalar('losses', losses['losses'].avg)
        logger.add_scalar('losses_kd', losses['losses_kd'].avg)

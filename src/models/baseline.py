import torch.cuda
import torch.nn as nn
import time
from torch import optim
from architectures import get_model
from logger import AverageMeter


class Baseline(nn.Module):
    def __init__(self, params_model):
        super(Baseline, self).__init__()
        # todo try with one model and with separate
        self.image_model = get_model(params_model['image_arch'], params_model['image_dim'], 'baseline')
        self.sketch_model = get_model(params_model['sketch_arch'], params_model['sketch_dim'], 'baseline')
        self.loss = nn.TripletMarginLoss(margin=params_model['triplet_margin'])
        self.sample_train_params = list(self.image_model.parameters()) + list(self.sketch_model.parameters())
        self.optimizer = optim.Adam(self.sample_train_params, params_model['lr'])
        torch.manual_seed(params_model['seed'])
        self.params_model = params_model

    def scheduler_step(self, epoch):
        pass

    def get_sketch_embeddings(self, sk):
        return self.sketch_model(sk)

    def get_image_embeddings(self, im):
        return self.image_model(im)

    def get_sketch_prediction(self, sk):
        raise ValueError("Accuracy not supported for Baseline model.")

    def get_image_prediction(self, im):
        raise ValueError("Accuracy not supported for Baseline model.")

    def train_once(self, train_loader, epoch, args):
        self.train()
        time_info = {'forward_pass_time': AverageMeter(), 'backward_pass_time': AverageMeter(),
                     'get_item_time': AverageMeter()}
        losses = AverageMeter()
        for i, (sk, im, neg_im, cls_sk, cls_im, cls_neg, ti) in enumerate(train_loader):
            time_info['get_item_time'].update(ti.sum())
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                im, sk, neg_im = im.cuda(), sk.cuda(), neg_im.cuda()
            forward_time = time.time()
            positive_feature = self.image_model(im)
            negative_feature = self.image_model(neg_im)
            sample_feature = self.sketch_model(sk)
            time_info['forward_pass_time'].update(time.time() - forward_time)
            backward_time = time.time()
            loss = self.loss(sample_feature, positive_feature, negative_feature)
            loss.backward()
            self.optimizer.step()
            time_info['backward_pass_time'].update(time.time() - backward_time)
            losses.update(loss.item())
            if (i + 1) % args.log_interval == 0:
                print('[Train] Epoch: [{0}][{1}/{2}]\t'
                      'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      .format(epoch + 1, i + 1, len(train_loader), losses=losses))

        time_info['forward_pass_time'] = time_info['forward_pass_time'].avg
        time_info['backward_pass_time'] = time_info['backward_pass_time'].avg
        time_info['get_item_time'] = time_info['get_item_time'].avg
        return {'loss': losses.avg}, time_info

    @staticmethod
    def add_to_log(logger, losses):
        return

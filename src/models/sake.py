import torch.cuda
import torch.nn as nn
from torch import ones_like, mul, sum, randn, mm, transpose, sqrt, sign
import torch.nn.functional as F
import torchvision.models as models
import math
import pretrainedmodels
import torch.backends.cudnn as cudnn
import numpy as np
import os, time
from logger import AverageMeter
from models.senet import cse_resnet50, cse_resnet50_hashing
from models.resnet import resnet50_hashing


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


class ResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False):
        super(ResnetModel, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        original_model = models.__dict__[arch](pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out


class ResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(ResnetModel_KDHashing, self).__init__()
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=False)

        self.ems = ems
        if self.ems:
            print('Error, no ems implementationin AlexnetModel_KDHashing')
            return None
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            print('Error, no freeze_features implementationin AlexnetModel_KDHashing')
            return None

    def forward(self, x):
        out_o = self.original_model.features(x)
        out_o = self.original_model.hashing(out_o)

        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)
        return out, out_kd


class SEResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)

        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out


class SEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel_KD, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)

        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.original_output = nn.Sequential(*list(original_model.children())[-1:])

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out_o = self.features(x)
        out_o = self.last_block(out_o)
        out_o = out_o.view(out_o.size()[0], -1)

        out = self.linear(out_o)
        out_kd = self.original_output(out_o)

        return out, out_kd


class CSEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KD, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = cse_resnet50()
        else:
            self.original_model = cse_resnet50(pretrained=None)

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x, y):
        out_o = self.original_model.features(x, y)
        out = nn.AdaptiveAvgPool2d(1)(out_o)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)

        out_kd = self.original_model.logits(out_o)
        return out, out_kd


class CSEResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KDHashing, self).__init__()

        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None)

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            layers = [self.original_model.layer1, self.original_model.layer2, self.original_model.layer3,
                      self.original_model.layer4]
            for ff in layers:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x, y):
        out_o = self.original_model.features(x, y)
        out_o = self.original_model.hashing(out_o)

        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)
        return out, out_kd


class EMSLayer(nn.Module):
    def __init__(self, num_classes, num_dimension):
        super(EMSLayer, self).__init__()
        self.cpars = nn.Parameter(randn(num_classes, num_dimension))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = pairwise_distances(x, self.cpars)
        out = - self.relu(out).sqrt()
        return out


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * mm(x, transpose(y, 0, 1))
    return dist


class HashingEncoder(nn.Module):
    def __init__(self, input_dim, one_dim, hash_dim):
        super(HashingEncoder, self).__init__()
        self.input_dim = input_dim
        self.one_dim = one_dim
        self.hash_dim = hash_dim

        self.en1 = nn.Linear(input_dim, one_dim)
        self.en2 = nn.Linear(one_dim, hash_dim)
        self.de1 = nn.Linear(hash_dim, one_dim)
        self.de2 = nn.Linear(one_dim, input_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e = self.en1(x)
        e = self.en2(self.relu(e))
        r = self.de1(e)
        r = self.de2(self.relu(r))
        r = self.relu(r)
        return e, r


class ScatterLoss(nn.Module):
    def __init__(self):
        super(ScatterLoss, self).__init__()

    def forward(self, e, y):
        sample_num = y.shape[0]
        e_norm = e / sqrt(sum(mul(e, e), dim=1, keepdim=True))
        cnter = 0
        loss = 0
        for i1 in range(sample_num - 1):
            e1 = e_norm[i1]
            y1 = y[i1]
            for i2 in range(i1 + 1, sample_num):
                e2 = e_norm[i2]
                y2 = y[i2]
                if y1 != y2:
                    cnter += 1
                    loss += sum(mul(e1, e2))

        return loss / cnter


class QuantizationLoss(nn.Module):
    def __init__(self):
        super(QuantizationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, e):
        return self.mse(e, sign(e))


class SAKE(nn.Module):
    def __init__(self, params_model):
        super(SAKE, self).__init__()
        self.model = CSEResnetModel_KDHashing(params_model['arch'], params_model['num_hashing'],
                                              params_model['num_clss'],
                                              freeze_features=params_model["freeze_features"],
                                              ems=params_model['ems_loss'])
        self.model = nn.DataParallel(self.model)
        self.model_t = cse_resnet50()
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

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model_t = self.model_t.cuda()
            self.criterion_train = self.criterion_train.cuda()
            self.criterion_train_kd = self.criterion_train_kd.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=params_model['weight_decay'])

        self.class_to_int_dict = self.get_class_int_from_str_dict()
        self.sake_lambda = params_model['sake_lambda']
        cudnn.benchmark = True

    def scheduler_step(self, epoch):
        lr = self.lr * math.pow(0.001, float(epoch) / self.epochs)
        print('epoch: {}, lr: {}'.format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_once(self, train_loader, epoch, args):
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
        self.model.train()
        self.model_t.train()
        train_loader_image = train_loader[0]
        train_loader_sketch = train_loader[1]
        train_setup_time = time.time() - train_setup_time
        for i, ((input, target, cid_mask, ti), (input_ext, target_ext, cid_mask_ext, ti_ext)) in enumerate(zip(train_loader_image, train_loader_sketch)):
            get_item_time.update(ti.sum() + ti_ext.sum())
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
                      .format(epoch + 1, i + 1, len(train_loader_image), losses=losses,
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

    def get_class_int_from_str_dict(self):
        class_to_int_path = os.path.join(self.root, self.zero_version, 'cname_cid.txt')
        class_to_int_dict = {}
        with open(class_to_int_path) as f:
            for line in f:
                key, val = self.process_str(line)
                class_to_int_dict[key] = val
        return class_to_int_dict

    @staticmethod
    def process_str(line):
        contents = line.split()
        val = contents[-1]
        name = []
        for c in contents[0:-1]:
            name += c.split('-')
        join_symbol = '_'
        key = join_symbol.join(name)
        return key, val

    @staticmethod
    def add_to_log(logger, losses):
        logger.add_scalar('losses', losses['losses'].avg)
        logger.add_scalar('losses_kd', losses['losses_kd'].avg)

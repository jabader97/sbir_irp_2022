#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import utils
import argparse
import pretrainedmodels


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Zero-Shot Sketch-based Image Retrieval.')
        # Seed
        parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility.')
        # Model
        parser.add_argument('--model', required=True, default='sem_pcyc', help='SBIR model')
        parser.add_argument('--sketch_arch', default='resnet50', help='sketch model architecture')
        parser.add_argument('--image_arch', default='resnet50', help='sketch model architecture')
        # Optional argument
        parser.add_argument('--dataset', required=True, default='Sketchy', help='Name of the dataset')
        parser.add_argument('--accuracy', default='None', type=str, help='Calculate the accuracy')
        # Different training test sets
        parser.add_argument('--split-eccv-2018', action='store_true', default=False,
                            help='Whether to use the splits of ECCV 2018 paper')
        parser.add_argument('--gzs-sbir', action='store_true', default=False,
                            help='Generalized zero-shot sketch based image retrieval')
        parser.add_argument('--filter-sketch', action='store_true', default=False, help='Allows only one sketch per '
                                                                                        'image (only for Sketchy)')
        # Size parameters
        parser.add_argument('--im-sz', default=224, type=int, help='Image size')
        parser.add_argument('--sk-sz', default=224, type=int, help='Sketch size')
        parser.add_argument('--dim-out', default=128, type=int, help='Output dimension of sketch and image')
        parser.add_argument('--image_dim', default=512, type=int, help='image embedding network output size')
        parser.add_argument('--sketch_dim', default=512, type=int, help='sketch embedding network output size')
        # Model parameters
        parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
        parser.add_argument('--epoch-size', default=100, type=int, help='Epoch size')
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in data loader')
        # Checkpoint parameters
        parser.add_argument('--test', action='store_true', default=False, help='Test only flag')
        parser.add_argument('--early-stop', type=int, default=20, help='Early stopping epochs.')
        # Optimization parameters
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='Number of epochs to train (default: 100)')
        parser.add_argument('--lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR',
                            help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--milestones', type=int, nargs='+', default=[], help='Milestones for scheduler')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule steps.')
        # I/O parameters
        parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--save-image-results', action='store_true', default=False, help='Whether to save image '
                                                                                             'results')
        parser.add_argument('--number-qualit-results', type=int, default=200, help='Number of qualitative results to be'
                                                                                   ' saved')
        parser.add_argument('--save-best-results', action='store_true', default=False, help='Whether to save the best '
                                                                                            'results')
        parser.add_argument('--normalize', action='store_true', default=False, help='Whether to normalize images')
        parser.add_argument('--path_dataset', type=str, default="", help='Dataset path')
        parser.add_argument('--path_aux', type=str, default="", help='Output path')
        parser = self.sem_pcyc_parse(parser)
        parser = self.sake_parser(parser)
        parser = self.baseline_parser(parser)
        parser = self.wandb_parse(parser)
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()

    def sem_pcyc_parse(self, parser):
        # Semantic models
        parser.add_argument('--semantic-models', nargs='+', default=['word2vec-google-news', 'hieremb-path'],
                            type=str, help='Semantic model')
        # Weight (on loss) parameters
        parser.add_argument('--lambda-se', default=10.0, type=float, help='Weight on the semantic model')
        parser.add_argument('--lambda-im', default=10.0, type=float, help='Weight on the image model')
        parser.add_argument('--lambda-sk', default=10.0, type=float, help='Weight on the sketch model')
        parser.add_argument('--lambda-gen-cyc', default=1.0, type=float, help='Weight on cycle consistency loss (gen)')
        parser.add_argument('--lambda-gen-adv', default=1.0, type=float, help='Weight on adversarial loss (gen)')
        parser.add_argument('--lambda-gen-cls', default=1.0, type=float, help='Weight on classification loss (gen)')
        parser.add_argument('--lambda-gen-reg', default=0.1, type=float, help='Weight on regression loss (gen)')
        parser.add_argument('--lambda-disc-se', default=0.25, type=float, help='Weight on semantic loss (disc)')
        parser.add_argument('--lambda-disc-sk', default=0.5, type=float, help='Weight on sketch loss (disc)')
        parser.add_argument('--lambda-disc-im', default=0.5, type=float, help='Weight on image loss (disc)')
        parser.add_argument('--lambda-regular', default=0.001, type=float, help='Weight on regularizer')
        return parser

    def sake_parser(self, parser):
        model_names = sorted(name for name in pretrainedmodels.__dict__
                             if name.islower() and not name.startswith("__"))
        parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                            choices=model_names,
                            help='model architecture: ' +
                                 ' | '.join(model_names) +
                                 ' (default: cse_resnet50)')
        parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                            help='number of hashing dimension (default: 64)')
        parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
                            help='freeze features of the base network')
        parser.add_argument('--ems_loss', dest='ems_loss', action='store_true',
                            help='use ems loss for the training')
        parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                            help='lambda for kd loss (default: 1)')
        parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                            help='lambda for semantic adjustment (default: 0.3)')
        parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                            help='lambda for total SAKE loss (default: 1)')
        parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str,
                            help='zeroshot version for training and testing (default: zeroshot)')
        parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)')
        parser.add_argument('--student_arch', default='cse_resnet50_hashing', help='student model architecture')
        parser.add_argument('--teacher_arch', default='cse_resnet50', help='teacher model architecture')
        return parser

    def baseline_parser(self, parser):
        parser.add_argument('--triplet_margin', default=0.2, type=float, help='Triplet loss margin for baseline.')
        return parser

    def wandb_parse(self, parser):
        # wandb args
        parser.add_argument('--log_online', action='store_true',
                            help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
        parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')
        parser.add_argument('--project', default='Sample_Project', type=str,
                            help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
        parser.add_argument('--group', default='Sample_Group', type=str, help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                                   In --savename default setting part of the savename.')
        parser.add_argument('--savename', default='group_plus_seed', type=str,
                            help='Run savename - if default, the savename will comprise the project and group name (see wandb_parameters()).')
        return parser

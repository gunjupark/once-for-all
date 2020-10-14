# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
import torch

from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.elastic_nn.networks import OFAMobileNetV3, OFAMobileNetV3_BP
from ofa.imagenet_codebase.run_manager import DistributedImageNetRunConfig
from ofa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from ofa.utils import download_url
from ofa.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()

parser.add_argument('--bp', action='store_true', help='train with aux')
parser.add_argument('--wc', type=float, default=2.0, help='aux loss factor')
parser.add_argument('--path', type=str, default='exp/temp', help='log path')

args = parser.parse_args()

print(args.bp)

if args.bp :
    print("bp training start... \n")

#args.path = 'exp/Aux_supernet'
args.dynamic_batch_size = 1
args.n_epochs = 200
# args.base_lr = 8e-3  #  OFA_mbv3
args.base_lr = 8e-4  #  OFA_mbv3_bp
args.warmup_epochs = 50
args.warmup_lr = 15e-4
args.ks_list = '7'
args.expand_list = '6'
args.depth_list = '4'

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 64
args.valid_size = 3925

args.opt_type = 'sgd'
#args.momentum = 0.95
args.momentum = 0.8
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'

#args.image_size = '128,160,192,224'
#args.continuous_size = True

#NOTE : for test(temporally)
args.image_size = '224'
args.continuous_size = False
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-6
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = -1
args.independent_distributed_sampling = False

args.kd_ratio = 0
args.kd_type = 'ce'


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    net = OFAMobileNetV3_BP(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    )
    # teacher model
    if args.kd_ratio > 0:
        args.teacher_model = OFAMobileNetV3_BP(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0, width_mult_list=1.0, ks_list=7, expand_ratio_list=6, depth_list=4,
        )
        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)


    
    # training
    from ofa.elastic_nn.training.progressive_shrinking import validate_bp, train

    validate_func_dict = {'image_size_list': {224} if isinstance(args.image_size, int) else sorted({160, 224}),
                          'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)})}
    validate_func_dict['ks_list'] = sorted(args.ks_list)
#    if distributed_run_manager.start_epoch == 0:
#        distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
#                validate(distributed_run_manager, **validate_func_dict), 'valid')
    train(distributed_run_manager, args,
            lambda _run_manager, epoch, is_test: validate_bp(_run_manager, epoch, is_test, **validate_func_dict))

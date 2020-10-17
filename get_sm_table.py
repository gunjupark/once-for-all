# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from tqdm import tqdm

import numpy as np

# (file-index) dictionary format
import json

from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig
from ofa.imagenet_codebase.run_manager import RunManager
from ofa.model_zoo import ofa_net

from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.elastic_nn.networks import OFAMobileNetV3_BP


parser = argparse.ArgumentParser()

parser.add_argument(
    '-a',
    '--alpha',
    help='The number of random subnet sample',
    type=int,
    default=1000)
parser.add_argument(
    '-p',
    '--path',
    help='The path of imagenet',
    type=str,
    default='/home/gunju/dataset/imagenette2')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')

parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)

parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=20)

'''
parser.add_argument(
    '-n',
    '--net',
    metavar='OFANET',
    default='ofa_mbv3_d234_e346_k357_w1.0',
    choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3'],
    help='OFA networks')
'''

args = parser.parse_args()
if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

# ofa_network = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

bp_net = OFAMobileNetV3_BP(
        dropout_rate=0, width_mult_list=1.0, ks_list=[3,5,7], expand_ratio_list=[3,4,6], depth_list=[2,3,4],
        )

# pretrained model load
init = torch.load('exp/bp_final/model_best.pth.tar')['state_dict']
bp_net.load_state_dict(init)


""" Randomly sample a sub-network,
    you can also manually set the sub-network using:
        ofa_network.set_active_subnet(ks=7, e=6, d=4)
"""

# (File:subnet) Index Dictionary
index_dict = {}

with tqdm(total=args.alpha, desc='Make sm table from subnet ...', disable=False) as t :
    for i in range(args.alpha):

        subnet_setting = bp_net.sample_active_subnet()
        subnet = bp_net.get_active_subnet(preserve_weight=True)
        index_dict[i] = subnet_setting


        """ Test sampled subnet """

        run_manager = RunManager('.tmp/eval_bp_subnet', subnet, run_config, init=False)
        # fixed image size: 224
        run_config.data_provider.assign_active_img_size(224)
        run_manager.reset_running_statistics(net=subnet)

        #print('Test random subnet:')
        #print(subnet.module_str)

        sm_res = run_manager.get_sm(net=subnet)

        #print('Results: sm = %s, sm_shape = %s' %(sm_res, sm_res.shape))
        np_fn = str(i) + ".npy"

        np.save(os.path.join("sm_tables",np_fn), sm_res)
        t.update(1)

json.dump(index_dict, open("sm_tables/file_subnet_dict.json","w"))


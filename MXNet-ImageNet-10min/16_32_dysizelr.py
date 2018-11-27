# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ctypes
ctypes.CDLL('./imagenet_10m/libimageaugdefault.so', ctypes.RTLD_LOCAL)
print("load libimageaug")
import os
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '16'
os.environ['MXNET_CPU_TEMP_COPY'] = '300'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = '24'
#############################################################################################
from moxing.mxnet.optimizer.contrib.XLRScheduler import XLRScheduler
from moxing.mxnet.optimizer.contrib.initializer_8k import Xavier_8k, MSRAPrelu_8k
from moxing.mxnet.optimizer.contrib.optimizer_8k import SGD_8k, NAG_8k
##############################################################################################
import mxnet as mx
import moxing.mxnet as mox
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Imagenet 10min')
    parser.add_argument('--network', dest='network', type=str, default='resnet_v1',
                        help='which network to use')
    parser.add_argument('--train_file1', dest='train_file1', type=str, default='ImageNet.rec',
                        help='dataset of the first stage')
    parser.add_argument('--train_idx1', dest='train_idx1', type=str, default='ImageNet.idx',
                        help='dataset of the first stage')
    parser.add_argument('--train_file2', dest='train_file2', type=str, default='ImageNet_original.rec',
                        help='second stage of train data')
    parser.add_argument('--train_idx2', dest='train_idx2', type=str, default='ImageNet_original.idx',
                        help='second stage of train data')
    parser.add_argument('--val_file', dest='val_file', type=str, default='ImageNet_val_original.rec',
                        help='eval data')
    parser.add_argument('--ckpt', dest='ckpt', help='save model', type=str)
    parser.add_argument('--save_frequency', dest='save_frequency', help='how many steps to save',
                        type=int, default=2)
    args, unkown = parser.parse_known_args()
    return args

def design_symbol(network):
    net = mox.get_model('classification', 'resnet_v1', num_classes=1000,
                        num_layers=50, image_shape='3,224,224', dtype='float16')
    return net


def get_model(network):
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None else [
        mx.gpu(int(i)) for i in range(num_gpus)]
    model = mx.mod.Module(
        context=devs,
        symbol=design_symbol(network)
    )
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = "Round"
    return model

def get_optimizer_params():
    lr = 0.24 * 8 
    sixteen_machines = [
        {'ep':0, 'lr':lr,  'sz':128, 'bs':64},
        {'ep':6, 'lr':lr*2, 'sz':128,  'bs':128, 'mode':'node'},
        {'ep':11, 'lr':lr*2, 'mode':'node'},
        {'ep':14, 'lr':lr*2, 'sz':224, 'bs':128, 'mode':'linear'},
        {'ep':17, 'lr':lr, 'sz':224, 'bs':160, 'mode':'cosine'},
        {'ep':28, 'sz':288, 'bs':128},
        {'ep':32, 'lr':lr/100, 'mode':'node'},
    ]
    
    lr_scheduler = XLRScheduler(sixteen_machines, datasize=1281167, mode='node', warmup=6, warmup_belr=lr,
                        num_workers=mox.get_hyper_parameter('num_workers', 1), 
                        dis=100//mox.get_hyper_parameter('num_workers', 1)+10)
    return {'learning_rate': 0.1,
            'wd' : 0.0001,
            'lr_scheduler': lr_scheduler,
            'clip_gradient': 5,
            'momentum' : 0.9,
            'multi_precision' : True,
            'rescale_grad': 1.0/8/64/mox.get_hyper_parameter('num_workers',1)
           }, lr_scheduler.bs_c, lr_scheduler.get_every_ep()
    
if __name__ == '__main__':
    print("start load ...")
    args = parse_args()
    cache = '/cache/'
    data_url = mox.get_hyper_parameter('data_url')
    train1 = args.train_file1
    train2 = args.train_file2
    idx1 = args.train_idx1
    idx2 = args.train_idx2
    val = args.val_file
    mox.file.copy(data_url+train1, cache+train1)
    mox.file.copy(data_url+train2, cache+train2)
    mox.file.copy(data_url+idx1, cache+idx1)
    mox.file.copy(data_url+idx2, cache+idx2)
    mox.file.copy(data_url+val, cache+val)
    print("success load")
    
    optimizer_params, dysize, epsize = get_optimizer_params()
    mox.set_hyper_parameter('data_url', cache)
    mox.set_hyper_parameter('val_file', val)
    mox.set_hyper_parameter('kv_store', 'dist_sync_device')
    mox.set_hyper_parameter('train_type', 'progressive_train')
    mox.set_hyper_parameter('save_frequency', args.save_frequency)
    epoch_end_callbacks = mox.save_model(args.ckpt+str(mox.get_hyper_parameter('worker_id', 0)))
    model = get_model(args.network)
    initializer = MSRAPrelu_8k()
    params_tuple = (None,None)
    metrics = [mx.metric.TopKAccuracy(top_k=5)]
    data_name=[train1,train1,train2,train2,train2]
    idx_name=[idx1,idx1,idx2,idx2,idx2]
    data_path = [cache + x for x in data_name]
    idx_path = [cache + x for x in idx_name]
    data_set = ('None_str', 'None_str')
    mox.run(
        data_set, model, params_tuple,
        run_mode=mox.ModeKeys.TRAIN,
        num_epoch=35,
        batch_size=0,
        metrics=metrics,
        optimizer=NAG_8k(**optimizer_params),
        optimizer_params=None,
        initializer=initializer,
        load_epoch=0,
        epoch_end_callbacks=epoch_end_callbacks,
        min_scale_set = [0.08, 0.08, 0.087, 0.087, 0.50],
        batchsize_set = [64, 128, 128, 160, 128],
        gy_size = [[(3, x[0] ,x[0]), 8*x[1]] for x in zip([128, 128, 224, 224, 288], [64, 128, 128, 160, 128])],
        isnewaug=1, is_eval=1, eval_epoch=31, dy_size=dysize[1:], ep_size=epsize,
        data_path=data_path,
        idx_path=idx_path)
    
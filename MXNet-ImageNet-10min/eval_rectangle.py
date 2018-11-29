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

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx
import logging
import argparse
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='train imagenet',
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,
                       help='data path')
parser.add_argument('--checkpoint_url', type=str,
                    help='checkpoint path, path which provide model to read'
                         ' it include model prefix')
parser.add_argument('--load_epoch', type=int, default=35,
                       help='load the model on an epoch using '
                            'the model-load-prefix')
parser.add_argument('--resize_shape', type=int, default=328,
                       help='the resize shape')
parser.add_argument('--image_shape', type=int, default=328,
                       help='the image shape')
parser.add_argument('--disp_batches', type=int, default=1000,
                       help='show progress for every n batches')
parser.add_argument('--num_gpus', type=int, default=4,
                       help='num gpus')
args = parser.parse_args()

data_path = args.data_path
checkpoint_url = args.checkpoint_url
load_epoch = args.load_epoch
resize_shape = args.resize_shape
image_shape = args.image_shape
num_gpus = args.num_gpus
mean = mx.nd.array([[[123.675]], [[116.28]], [[103.53]]])
std = mx.nd.array([[[58.395]], [[57.12]], [[57.375]]])
def eval():
    record = mx.recordio.MXRecordIO(data_path, 'r')
    sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_url, load_epoch)
    devs = [mx.gpu(x) for x in range(num_gpus)]
    models = []
    
    metrics = [mx.metric.create('acc'),
                   mx.metric.create('top_k_accuracy', top_k=5),
                   mx.metric.CrossEntropy()]
    eval_metric = mx.metric.create(metrics)
    batch_end_callback = mx.callback.Speedometer(1, args.disp_batches, auto_reset=False)
    for dev in devs:
        model = mx.mod.Module(context=[dev], symbol=sym)
        model.bind(for_training=False,
                   data_shapes=[('data', (1, 3, image_shape, image_shape))])
        model.set_params(arg_params, aux_params)
        models.append(model)
    nbatch = 0
    ret = 1
    while ret:
        batch_data = []
        labels = []
        for i in range(len(devs)):
            rec_data = record.read()
            if rec_data is None:
                ret = 0
                break
            header, img = mx.recordio.unpack(rec_data)
            img = mx.img.imdecode(img)
            
            label = header.label
            labels.append(label)
            img = mx.img.resize_short(img, resize_shape)
            img, _ = mx.img.center_crop(img, (int(img.shape[1]/1.14), int(img.shape[0]/1.14)))
            img = mx.nd.transpose(img, axes=(2, 0, 1))
            img = img.astype('float32')
            img = mx.img.color_normalize(img, mean, std)
            img = img.expand_dims(axis=0)
            img = img.as_in_context(devs[i])
            batch = mx.io.DataBatch([img], [])
            models[i].forward(batch)
        for i in range(len(labels)):
            eval_metric.update([mx.nd.array([labels[i]])], models[i].get_outputs())
            batch_end_param = mx.model.BatchEndParam(epoch=0, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
            batch_end_callback(batch_end_param)
            nbatch += 1
    logging.info(eval_metric.get_name_value())

if __name__ == '__main__':
    eval()
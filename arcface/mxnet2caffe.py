import sys, argparse
import mxnet as mx
import sys
import os

import caffe

from find import *

import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'
parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='./model_mxnet/model-r34-amf-slim/model')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='./model_caffe/model-r34-amf-slim/model.prototxt')
parser.add_argument('--cf-model',    type=str, default='./model_caffe/model-r34-amf-slim/model.caffemodel')
args = parser.parse_args()

# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
net = caffe.Net(args.cf_prototxt, caffe.TEST)

# Convert
all_keys = list(arg_params.keys()) + list(aux_params.keys())
all_keys.sort()

print('ALL KEYS IN MXNET:')
print('%d KEYS' %len(all_keys))


print('VALID KEYS:')

for i_key,key_i in enumerate(all_keys):

  try:
    print("====% 3d | %s "%(i_key, key_i.ljust(40)))
    
    if 'data' is key_i:
      pass
    
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      print(key_i,key_caffe)
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    elif '_gamma' in key_i and 'relu' in key_i:   # for prelu
      key_caffe = key_i.replace('_gamma','')
      assert (len(net.params[key_caffe]) == 1) #prelu 只有有gamma，没有其他参数，对应caffe 该层也只有一个参数
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      print("key_i",key_i)
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    elif '_gamma' in key_i and 'relu' not in key_i:  #BN gamma
      key_caffe = key_i.replace('_gamma','_scale')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    elif '_beta' in key_i:  #BN beta
      key_caffe = key_i.replace('_beta','_scale')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    #BN包含3个值，mean、var、滑动系数(类似scale)
    elif '_moving_mean' in key_i:  #BN mean
      key_caffe = key_i.replace('_moving_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
      print("{}: {}->{}".format(key_i, aux_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    elif '_moving_var' in key_i:  #BN var
      key_caffe = key_i.replace('_moving_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
      print("{}: {}->{}".format(key_i, aux_params[key_i].shape, net.params[key_caffe][0].data.shape))
    
    #gluon中的名字后缀'_running_mean'、'_running_var'
    
    else:
      # pass
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    print("\nError!  key error mxnet:{}".format(key_i))

net.save(args.cf_model)

print("\n*** PARAMS to CAFFEMODEL Finished. ***\n")




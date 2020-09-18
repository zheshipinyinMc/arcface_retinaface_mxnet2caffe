import caffe
import cv2
import numpy as np
import os
import time

def caffe_demo():
    caffe.set_mode_cpu()
    
    model_def = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.prototxt'
    model_weights = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.caffemodel'
    
    net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          112, 112)  # image size is 227x227
    
    img = cv2.imread("0.jpg")
    
    #img = img[...,::-1] #BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #img=(img-127.5)
    
    #img=img*0.0078125
    
    img = np.transpose( img, (2,0,1) ) #HWC->CHW
    #print(img)
    #print(img.shape)
    
    
    input_blob = np.expand_dims(img, axis=0)
    
    net.blobs['data'].data[...] = input_blob
    
    output = net.forward()
    output_prob = output['fc1'][0]
    
    print(output_prob)
    print(output_prob.shape)



from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb

#=====合并Conv_BN_Scale、FC_BN_Scale，从prototxt中去掉BN、Scale层=====
def write_protxt():
    caffe.set_mode_cpu()
    
    proto_name = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.prototxt'
    model_name = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.caffemodel'
    
    #读取prototxt中各层
    proto = None
    with open(proto_name) as fd:
        proto = caffe_pb2.NetParameter()
        text_format.Merge(fd.read(), proto)
        #print(proto)
    
    layers=proto.layer
    
    #写入新prototxt
    f=open('model_caffe/model_merge_bn.prototxt','w')
    
    if len(proto.name) > 0:
        #print('name:"{}"\n'.format(proto.name))
        f.write('name:"{}"\n'.format(proto.name))
    
    #Convolution--BN--Scale, 记录BN的bottom和scale的top, 合并之后以scale为bottom的层, 将bottom改为BN的bottom
    bnbottom_list=[]
    scaletop_list=[]
    
    bn_scale_index_list=[]
    
    layers_num=len(layers)
    for i in range(layers_num):
        layer_i_type=layers[i].type
        layer_i_name=layers[i].name
        layer_i_bottom=layers[i].bottom
        layer_i_top=layers[i].top
        
        #if layer_i_type=='Convolution':
        if layer_i_type=='Convolution' or layer_i_type=='InnerProduct':
            #print(layer_i_name,len(layer_i_top),layer_i_top[0])
            for j in range(layers_num):
                layer_j_type=layers[j].type
                layer_j_name=layers[j].name
                layer_j_bottom=layers[j].bottom
                layer_j_top=layers[j].top
                
                if layer_j_type=='BatchNorm' and layer_i_top[0] in layer_j_bottom:
                    for k in range(layers_num):
                        layer_k_type=layers[k].type
                        layer_k_name=layers[k].name
                        layer_k_bottom=layers[k].bottom
                        layer_k_top=layers[k].top
                        
                        if layer_k_type=='Scale' and layer_j_top[0] in layer_k_bottom:
                            for l in range(layers_num):
                                layer_l_type=layers[l].type
                                layer_l_name=layers[l].name
                                layer_l_bottom=layers[l].bottom
                                layer_l_top=layers[l].top
                                #print(layer_l_name,layer_l_top[0])
                                #print(k,layers_num)
                                if l!=i and l!=j and l!=k:
                                    layer_l_bottom_num=len(layer_l_bottom)
                                    for b_index in range(layer_l_bottom_num):
                                        if layer_k_top[0]==layer_l_bottom[b_index]:
                                            print(layer_i_name,layer_j_name,layer_k_name,layer_l_name,b_index)
                                            bn_scale_index_list.append(j)
                                            bn_scale_index_list.append(k)
                                            if layer_i_type=='Convolution':
                                                #layers[i].convolution_param.bias_term=True
                                                if layers[i].convolution_param.bias_term == False:
                                                    layers[i].convolution_param.bias_term = True
                                            
                                            layers[l].bottom[b_index]=layers[i].top[0]
                                            print(i,j,k,l)
                            if k==layers_num-1:
                                bn_scale_index_list.append(j)
                                bn_scale_index_list.append(k)
                                
        if i not in bn_scale_index_list:
            f.write('layer{\n'+'{}'.format(pb.text_format.MessageToString(layers[i]))+'}\n')
    f.close()


#=====合并Conv_BN_Scale、FC_BN_Scale，根据修改后prototxt，为对应权重赋值=====
def write_caffemodel():
    caffe.set_mode_cpu()
    
    deploy_name = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.prototxt'
    model_name = 'model_caffe/mobilefacenet-res2-6-10-2-dim512/model.caffemodel'
    
    deploy_new_name='model_caffe/model_merge_bn.prototxt'
    model_new_name = 'model_caffe/model_merge_bn.caffemodel'
    
    net=caffe.Net(deploy_name, model_name, caffe.TEST)
    net_new=caffe.Net(deploy_new_name, caffe.TEST)
    
    proto = None
    with open(deploy_name) as fd:
        proto = caffe_pb2.NetParameter()
        text_format.Merge(fd.read(), proto)
        #print(proto)
    
    layers=proto.layer
    layers_num=len(layers)
    
    bn_scale_index=[]
    for i in range(layers_num):
        layer_i_type=layers[i].type
        layer_i_name=layers[i].name
        layer_i_bottom=layers[i].bottom
        layer_i_top=layers[i].top
        
        if layer_i_name not in net.params.keys():#data ,Eltwise
            continue
        
        bn_index=-1
        scale_index=-1
        merge_flag=False
        
        #if layer_i_type=='Convolution':
        if layer_i_type=='Convolution' or layer_i_type=='InnerProduct':
            #print(layer_i_name,len(layer_i_top),layer_i_top[0])
            for j in range(layers_num):
                layer_j_type=layers[j].type
                layer_j_name=layers[j].name
                layer_j_bottom=layers[j].bottom
                layer_j_top=layers[j].top
                
                if layer_j_type=='BatchNorm' and layer_i_top[0] in layer_j_bottom:
                    for k in range(layers_num):
                        layer_k_type=layers[k].type
                        layer_k_name=layers[k].name
                        layer_k_bottom=layers[k].bottom
                        layer_k_top=layers[k].top
                        
                        if layer_k_type=='Scale' and layer_j_top[0] in layer_k_bottom:
                            bn_index=j
                            scale_index=k
                            merge_flag=True
        
        if merge_flag==True:
            conv_name=layer_i_name
            #print(conv_name)
            
            #conv
            conv=net.params[conv_name]
            weight=conv[0].data
            
            conv_bias_flag=False #默认no bias
            if layer_i_type=='Convolution':
                conv_bias_flag=layers[i].convolution_param.bias_term
                #print(conv_bias_flag)
            
            if conv_bias_flag or layer_i_type=='InnerProduct':
                bias=conv[1].data
            #bias = np.zeros(weight.shape[0]) #no bias
            
            #BN
            bn_name=layers[bn_index].name
            #print(bn_name)
            bn=net.params[bn_name]
            bn_mean=bn[0].data
            bn_var=bn[1].data
            bn_scale=bn[2].data
            bn_eps=layers[bn_index].batch_norm_param.eps
            
            if bn_scale != 0:
                bn_scale = 1. / bn_scale
            
            bn_mean = bn_mean * bn_scale
            bn_var = bn_var * bn_scale
            bn_std = np.sqrt(bn_var + bn_eps)
            
            #Scale
            scale_name=layers[scale_index].name
            #print(scale_name)
            scale=net.params[scale_name]
            scale_gamma=scale[0].data
            scale_beta=scale[1].data
            
            if layer_i_type=='InnerProduct':
                weight_new=weight*np.reshape(scale_gamma/bn_std, (weight.shape[0], 1))
            else:
                weight_new=weight*np.reshape(scale_gamma/bn_std, (weight.shape[0], 1, 1, 1))
            
            if conv_bias_flag:
                bias_new=scale_gamma/bn_std*(bias-bn_mean)+scale_beta
            else:
                bias_new=scale_gamma/bn_std*(-bn_mean)+scale_beta
            
            net_new.params[conv_name][0].data[...]=weight_new
            net_new.params[conv_name][1].data[...]=bias_new
            #print(len(net.params[conv_name]),len(net_new.params[conv_name]))
            
        else:
            #其余层权重也赋值到net_new
            layer_name=layer_i_name
            #print(layer_name)
            if layer_name not in net_new.params.keys():
                print("====>",layer_name)
                continue
            layer_params_num=len(net.params[layer_name])
            for k in range(layer_params_num):
                net_new.params[layer_name][k].data[...]=net.params[layer_name][k].data
        
    net_new.save(model_new_name)

if __name__ == '__main__':
    
    #caffe_demo()
    
    #===合并BN_Scale到Conv或FC，从网络中去掉BN_Scale层===
    #write_protxt()
    #write_caffemodel()
    
    
    





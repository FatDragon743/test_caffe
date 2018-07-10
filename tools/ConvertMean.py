# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
@attention: 计算数据文件的mean，转化为为python可以调用的pyc格式
'''
import sys
import os
import numpy as np
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
#　编写一个函数，将二进制的均值转换为python的均值
def convert_mean_bin(binMean,npyMean):
    '''
    将准备好的mdb文件，也就是lmdb格式的数据文件（label，image）的mean计算出来，我不太清楚计算出来的是RGB还是BGR。。。
    方法是调用compute_image_mean.exe，在caffe生成的文件中的bin目录下，产出是一个二进制文件（*.binaryproto）.
    '''
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
def convert_mean_pyc():
    '''
    转化为为python可以调用的pyc格式
    '''
    base_path = os.path.dirname(os.getcwd())+'\\data\\tmp\\'
    exe_path = caffe_root+"bin/compute_image_mean.exe"
    exe_str =  exe_path+' '+ base_path+"train_lmdb "+base_path+"train_mean.binaryproto"
#     str = caffe_root+"bin\compute_image_mean.exe"
    os.system(exe_str)
# 调用函数转换均值
def test():
    mu = np.load('../data/tmp/train_mean.npy')
    mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
    print ('mean-subtracted values:', zip('BGR', mu))
def get_mean():
    '''
    return [116,117,118]
    '''
    mu = np.load('../data/tmp/train_mean.npy')
    mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
    return mu.tolist()
# binMean='../data/tmp/train_mean.binaryproto'
# npyMean='../data/tmp/train_mean.npy'
# convert_mean(binMean,npyMean)
get_mean()
# test()
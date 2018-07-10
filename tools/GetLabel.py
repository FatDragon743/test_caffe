# -*- coding:utf-8 -*-  
'''
Created on 2018年7月6日

@author: Administrator
@attention: 由文件目录，生成label标签,存为文件
'''
import sys
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')

import os
import glob
import random
import numpy as np
import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb
from tools.ImgProcess import *
# 根据图片和标签转化为对应的lmdb格式
def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMG_HEIGHT,
        height=IMG_WIDTH,
        label=label,
        data=np.rollaxis(img, 2).tostring())


# 创建lmdb的基类
class GenerateLmdb(object):

    def __init__(self, img_path):
        """
        img_path -> multiple calss directory
        like, class_1, class_2, class_3....
        each class has corresponding class image like class_1_1.png
        :param img_path:
        """
        # get all the images in different class directory
        # 获取到多有的图片列表
        self.img_lst = glob.glob(os.path.join(img_path, "*",'*.jpg'))
        print 'input_img list num is %s' % len(self.img_lst)
        print self.img_lst[0:5]
        # shuffle all the images
        # 需要对列表乱序
        random.shuffle(self.img_lst)

    # 根据标签，比例生成训练lmdb以及验证lmdb
    def generate_lmdb(self, label_lst, percentage, train_path, validation_path):
        """
        label_lst like ['class_1', 'class_2', 'class_3', .....]
        percentage like is 5 (4/5) then 80% be train image, (1/5) 20% be validation image
        train_path like that '/data/train/train_lmdb'
        validation_path like '/data/train/validation_lmdb'
        """
        print 'now generate train lmdb'+"*"*30
        self._generate_lmdb(label_lst, percentage, False, train_path)
        print 'now generate validation lmdb'+"*"*30
        self._generate_lmdb(label_lst, percentage, True, validation_path)

        print '\n generate all images'

    def _generate_lmdb(self, label_lst, percentage, b_train, input_path):
        """
        b_train is True means to generate train lmdb, or validation lmdb
        """
        print input_path
        output_db = lmdb.open(input_path, map_size=int(5e8))
        with output_db.begin(write=True) as in_txn:
            for idx, img_path in enumerate(self.img_lst):

                # create train data
                if b_train:
                    # !=0 means validation data then skip loop
                    if idx % percentage != 0:
                        continue
                # create validation data
                else:
                    # ==0 means train data then skip
                    if idx % percentage == 0:
                        continue

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = pre_process_img(img)
                # path like that '../../class_1/0001.png'
                # so img_path.split('/')[-2] -> class_1
                print img_path.split('\\')
                label = label_lst.index(img_path.split('\\')[-2])
                datum = make_datum(img, label)
                in_txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
                print '{:0>5d}'.format(idx) + '->label: ', label, " " + img_path

        output_db.close()


def get_label_lst_by_dir(f_dir):
    """
    f_dir like 'home/user/class', sub dir 'class_1', 'class_2'...'class_n'
    :return: ['class_1', 'class_2'...'class_n']
    """
    return os.listdir(f_dir)
def get_label_lst_to_file(f_dir):
    '''
    生成cata文件，记录了label标签
    '''
    _list = get_label_lst_by_dir(f_dir)
    print _list
    new_list = [(str(idx)+'\t'+i+'\n') for idx,i in enumerate(_list)]
    print new_list
    with open('../data/tmp/cate.txt','w') as f_w:
        for i in new_list:
            f_w.write(i)
    print "over!"
if __name__ == '__main__':
    img_path = '../data/train/'
    cl = GenerateLmdb(img_path)
    train_lmdb = '../data/tmp/train_lmdb'
    validation_lmdb = '../data/tmp/validation_lmdb'
    # 删除原有的lmdb文件
    os.system('del /q ' + train_lmdb.replace("/", "\\"))
    os.system('del /q ' + validation_lmdb.replace("/", "\\"))
    input_path = '../data/train/'
    label_lst = get_label_lst_by_dir(input_path)
    get_label_lst_to_file(input_path)
    print 'label_lst is: %s' % ', '.join(label_lst)
    # (1/10)10% to be validation data, 90% to be train data
    # 1/10的文件为验证lmdb, 9/10为训练lmdb
    percentage = 10#if idx%percentage==0即只有整除的时候进入测试集
    cl.generate_lmdb(label_lst, percentage, train_lmdb, validation_lmdb)
#     print get_label_lst_by_dir('../data/train')
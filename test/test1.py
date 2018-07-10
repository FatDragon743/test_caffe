# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
# list = ["a","b","c"]
# new_list = [(str(idx)+'\t'+i) for idx,i in enumerate(list)]
# print new_list
from tools.test_get_label import get_label_lst_to_file

input_path = '../data/train/'
# label_lst = get_label_lst_by_dir(input_path)/
get_label_lst_to_file(input_path)
# -*- coding:utf-8 -*-  
'''
Created on 2018年7月12日

@author: Administrator
'''
import sys
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2 


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_image_file',
                        help='Output image file')
    parser.add_argument('--rankdir',
                        help=('One of TB (top-bottom, i.e., vertical), '
                              'RL (right-left, i.e., horizontal), or another '
                              'valid dot option; see '
                              'http://www.graphviz.org/doc/info/'
                              'attrs.html#k:rankdir'),
                        default='LR')
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")

    args = parser.parse_args()
    return args


def main():
    args = {
        'input_net_proto_file1':'../data/tmp/train.prototxt',
        'output_image_file1':'../data/tmp/train_net.png',
        'input_net_proto_file2':'../data/tmp/test.prototxt',
        'output_image_file2':'../data/tmp/test_net.png',
        'rankdir':'LR',
        'phase':'ALL'
    }
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args['input_net_proto_file1']).read(), net)
    text_format.Merge(open(args['input_net_proto_file2']).read(), net)
    
    phase=None;
    if args['phase'] == "TRAIN":
        phase = caffe.TRAIN
    elif args['phase'] == "TEST":
        phase = caffe.TEST
    elif args['phase'] != "ALL":
        raise ValueError("Unknown phase: " + args.phase)
    print('Drawing net to %s' % args['output_image_file1'])
    caffe.draw.draw_net_to_file(net, args['output_image_file1'], args['rankdir'],
                                caffe.TRAIN)
    print('Drawing net to %s' % args['output_image_file2'])
    caffe.draw.draw_net_to_file(net, args['output_image_file2'], args['rankdir'],
                                caffe.TEST)
    print "done"


if __name__ == '__main__':
    main()
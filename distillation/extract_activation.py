# --------------------------------------------------------
# Distilling R-FCN Networks
# Author: Joseph K J
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
from symbols import *
from dataset import *
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Extract the activations of a penultimate layer.')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    update_config(args.cfg)
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger
from core.loader import TestLoader
from utils.load_model import load_param

def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, 'activations')

    # load symbol and testing data
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    imdb = eval(config.dataset.dataset)(config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path, result_path=final_output_path)
    roidb = imdb.gt_roidb()

    # get test data iter
    print len(ctx)
    test_data = TestLoader(roidb, config, batch_size=len(ctx), shuffle=True, has_rpn=True)
    print 'Loaded iterators'

    # load model
    prefix = os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix)
    arg_params, aux_params = load_param(prefix, config.TEST.test_epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    print 'done'



if __name__ == '__main__':
    main()

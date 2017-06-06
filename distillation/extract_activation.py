# --------------------------------------------------------
# Distillation of Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Joseph K J
# --------------------------------------------------------

import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
import pprint

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval
from extractor import Extractor, write_activations
from utils.load_model import load_param
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Extract the activations of 4b22 layer and pickle it to output folder')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from utils.create_logger import create_logger


def get_activation(cfg, dataset, image_set, root_path, dataset_path,
                   ctx, prefix, epoch,
                   vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None, output_folder=None):
    if not logger:
        assert False, 'require a logger'

    # pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_of_fourth_layer(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rfcn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create extractor
    extractor = Extractor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    time.sleep(1)
    print 'Extracting the activations and pickling it to ', output_folder
    write_activations(extractor, test_data, output_folder)
    print 'Done.'


def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    output_folder = os.path.join(config.output_path, args.cfg.split('/')[-1].split('.')[0])

    get_activation(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
                   ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
                   args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path, output_folder=output_folder)

if __name__ == '__main__':
    main()

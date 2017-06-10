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
import pprint
import time
import logging
import mxnet as mx
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Student-Teacher RFCN network')
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

from utils.create_logger import create_logger
from core.loader import TestLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param
from symbols import *
from dataset import *
from distillation_symbols import *


def test_network(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None, output_folder=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    # pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.test_symbol + '.' + cfg.test_symbol)()
    student_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_symbol_of_student_teacher_graft(cfg, student_sym_instance, is_train=False)
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    distillation_folder_prefix = os.path.join(output_folder, cfg.distillation_output_folder_name, cfg.TRAIN.model_prefix)
    arg_params, aux_params = load_param(prefix, epoch, process=True)
    arg_params_student, aux_params_student = load_param(distillation_folder_prefix, epoch, process=True)

    arg_params.update(arg_params_student)
    aux_params.update(aux_params_student)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # # decide maximum shape
    # data_names = [k[0] for k in test_data.provide_data_single]
    # label_names = None
    # max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    # if not has_rpn:
    #     max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))
    #
    # # create predictor
    # predictor = Predictor(sym, data_names, label_names,
    #                       context=ctx, max_data_shapes=max_data_shape,
    #                       provide_data=test_data.provide_data, provide_label=test_data.provide_label,
    #                       arg_params=arg_params, aux_params=aux_params)
    #
    # # start detection
    # pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)


def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    # print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.get_activation_for)
    output_folder = os.path.join(config.output_path, args.cfg.split('/')[-1].split('.')[0])

    test_network(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path, output_folder=output_folder)

if __name__ == '__main__':
    main()

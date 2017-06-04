# --------------------------------------------------------
# Distilling R-FCN Networks
# Author: Joseph K J
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
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


def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, 'activations')
    print final_output_path
    print ctx

if __name__ == '__main__':
    main()

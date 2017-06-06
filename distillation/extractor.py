# --------------------------------------------------------
# Distillation of Deformable Convolutional Networks
# Created by Joseph K J
# --------------------------------------------------------
import mxnet as mx
import _init_paths
import cPickle
import os

from core.module import MutableModule
from utils.PrefetchingIter import PrefetchingIter

class Extractor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def extract(self, data_batch):
        self._mod.forward(data_batch)
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]

    def extract_backup(self, data_batch):
        self._mod.forward(data_batch)
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def write_activations(extractor, test_data, output_path):

    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    i = 0
    for im_info, data_batch in test_data:
        scales = [iim_info[0, 2] for iim_info in im_info]
        output_all = extractor.extract(data_batch)
        label = data_batch.label
        data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]

        count = 0
        for output, data_dict, scale in zip(output_all, data_dict_all, scales):
            res4b22_relu_output = output['res4b22_relu_output'].asnumpy()[0]
            pickle_activation(output_path, label[count].split('/')[-1], res4b22_relu_output)
            count += 1

        i += 1
        if i == 1:
            break


def pickle_activation(path, file_name, activation):
    path += '/activations'
    if not os.path.exists(path):
        os.mkdir(path)
    cache_file = os.path.join(path, file_name + '-actv')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(activation, fid, cPickle.HIGHEST_PROTOCOL)

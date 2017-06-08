import _init_paths
import mxnet as mx
import numpy as np
import os
import cPickle

from mxnet.executor_manager import _split_input_slice

from rpn.rpn import get_rpn_batch, assign_anchor


class ActivationLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None, output_folder=None):
        super(ActivationLoader, self).__init__()
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        self.work_load_list = work_load_list
        self.output_folder = output_folder + '/activations'
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        self.data_name = ['data']
        self.label_name = ['label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        return max_data_shape, max_label_shape

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.get_data_and_label_for_each_image(iroidb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def get_data_and_label_for_each_image(self, iroidb):
        # get testing data for multigpu
        data, _ = get_rpn_batch(iroidb, self.cfg)
        label = self.load_activation(iroidb[0]['image'])
        return {'data': data, 'label': label}

    def load_activation(self, file_name):
        acts_file = os.path.join(self.output_folder, file_name.split('/')[-1] + '-actv')
        assert os.path.exists(acts_file), 'The activation file does not exist: {}'.format(acts_file)

        with open(acts_file, 'rb') as fid:
            acts = cPickle.load(fid)

        acts = np.array([acts])
        label = {'label': acts}

        return label

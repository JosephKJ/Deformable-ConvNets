import mxnet as mx
import numpy as np

def get_names():
    pred = ['data']
    label = ['label']
    return pred, label


class DistillationMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(DistillationMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # # pred (b, c, p) or (b, c, h, w)
        # pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        # pred_label = pred_label.reshape((pred_label.shape[0], -1))
        #
        # # label (b, p)
        # label = label.asnumpy().astype('int32')
        #
        # # filter with keep_inds
        # keep_inds = np.where(label != -1)
        # pred_label = pred_label[keep_inds]
        # label = label[keep_inds]
        #
        # self.sum_metric += np.sum(pred_label.flat == label.flat)
        # self.num_inst += len(pred_label.flat)

        print pred
        print label
        self.sum_metric += 0
        self.num_inst += 0

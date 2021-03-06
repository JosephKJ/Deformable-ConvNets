import mxnet as mx
import numpy as np

# Ref: https://github.com/dmlc/mxnet/blob/master/python/mxnet/metric.py#L124

def get_names():
    pred = ['data']
    label = ['label']
    return pred, label


class DistillationMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(DistillationMetric, self).__init__('Student_Loss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):

        pred_label = preds[0].asnumpy()
        label = labels[0].asnumpy()

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

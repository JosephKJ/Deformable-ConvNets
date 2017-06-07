import _init_paths
import mxnet as mx


class ActivationLoader(mx.io.DataIter):

    def __init__(self):
        super(ActivationLoader, self).__init__()

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
        super(ActivationLoader, self).reset()

    def iter_next(self):
        return super(ActivationLoader, self).iter_next()

    def next(self):
        return super(ActivationLoader, self).next()

    def getindex(self):
        return super(ActivationLoader, self).getindex()

    def getpad(self):
        return super(ActivationLoader, self).getpad()

import _init_paths
import mxnet as mx
from utils.symbol import Symbol


class student_symbol(Symbol):

    def __init__(self):
        self.eps = 1e-5

    def create_symbol(self, data):
        """
        Create a symbol that is of the same output dimension as 4b22-layer of ResNet-101
        :param data: Data Symbol
        :return: mx.symbol
        """

        # Conv-BN-Relu 1
        conv1 = mx.symbol.Convolution(name='conv1', data=data, kernel=(7,7), num_filter=64,
                                      pad=(3, 3), stride=(2, 2), no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True,
                                       fix_gamma=False, eps=self.eps)
        relu1 = mx.symbol.Activation(name='conv1_relu', data=bn_conv1, act_type="relu")

        # Pool 1
        pool1 = mx.symbol.Pooling(name='pool1', data=relu1, pooling_convention='full',
                                  pad=(0, 0), kernel=(3, 3), stride=(2, 2), pool_type='max')

        # Conv-BN-Relu 2
        conv2 = mx.symbol.Convolution(name='conv2', data=pool1, kernel=(1,1), num_filter=256,
                                      pad=(0, 0), stride=(1, 1), no_bias=True)
        bn_conv2 = mx.symbol.BatchNorm(name='bn_conv2', data=conv2, use_global_stats=True,
                                       fix_gamma=False, eps=self.eps)
        relu2 = mx.symbol.Activation(name='conv2_relu', data=bn_conv2, act_type="relu")

        # Conv-BN-Relu 3
        conv3 = mx.symbol.Convolution(name='conv3', data=relu2, kernel=(1,1), num_filter=512,
                                      pad=(0, 0), stride=(2, 2), no_bias=True)
        bn_conv3 = mx.symbol.BatchNorm(name='bn_conv3', data=conv3, use_global_stats=True,
                                       fix_gamma=False, eps=self.eps)
        relu3 = mx.symbol.Activation(name='conv3_relu', data=bn_conv3, act_type="relu")

        # Conv-BN-Relu 4
        conv4 = mx.symbol.Convolution(name='conv4', data=relu3, kernel=(1,1), num_filter=1024,
                                      pad=(0, 0), stride=(2, 2), no_bias=True)
        bn_conv4 = mx.symbol.BatchNorm(name='bn_conv4', data=conv4, use_global_stats=True,
                                       fix_gamma=False, eps=self.eps)
        relu4 = mx.symbol.Activation(name='conv4_relu', data=bn_conv4, act_type="relu")

        return relu4

    def get_symbol(self, cfg, is_train=True):
        data = mx.sym.Variable(name='data')
        label = mx.sym.Variable(name='label')

        # Getting all the convolutions
        conv_symbol = self.create_symbol(data)

        # Adding the mean-squared-loss
        mse_symbol = mx.symbol.LinearRegressionOutput(name='mse', data=conv_symbol, label=label)

        self.sym = mse_symbol
        return mse_symbol

    def init_weights(self, cfg, arg_params, aux_params):
        arg_params['conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv1_weight'])
        arg_params['conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv1_bias'])

        arg_params['conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv2_weight'])
        arg_params['conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv2_bias'])

        arg_params['conv3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv3_weight'])
        arg_params['conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv3_bias'])

        arg_params['conv4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv4_weight'])
        arg_params['conv4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv4_bias'])

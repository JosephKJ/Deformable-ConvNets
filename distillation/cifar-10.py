import mxnet as mx
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Character recognition using Lenet.
# Using MNIST dataset.

def GetCifar10():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/cifar/train.rec')) or \
       (not os.path.exists('data/cifar/test.rec')) or \
       (not os.path.exists('data/cifar/train.lst')) or \
       (not os.path.exists('data/cifar/test.lst')):
        os.system("wget -q http://data.mxnet.io/mxnet/data/cifar10.zip -P data/")
        os.chdir("./data")
        os.system("unzip -u cifar10.zip")
        os.chdir("..")

GetCifar10()

batch_size = 128
total_batch = 50000 / 128 + 1

train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)

test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=1)


train_iter = train_dataiter
val_iter = test_dataiter
test_iter = test_dataiter
eps = 1e-5



# # Building the network
# print 'Building Network.'
# data = mx.sym.Variable('data')
# # Conv-BN-Relu 1
# conv1 = mx.sym.Convolution(data=data, kernel=(7,7), num_filter=64, pad=(3, 3), stride=(2, 2))
# bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=eps)
# relu1 = mx.sym.Activation(data=bn_conv1, act_type="relu")
# # Pool 1
# pool1 = mx.sym.Pooling(data=relu1, pool_type="avg", kernel=(3,3), stride=(2,2))
# # Conv-BN-Relu 2
# conv2 = mx.sym.Convolution(data=pool1, kernel=(1,1), num_filter=256, pad=(0, 0), stride=(1, 1))
# bn_conv2 = mx.symbol.BatchNorm(name='bn_conv2', data=conv2, use_global_stats=True, fix_gamma=False, eps=eps)
# relu2 = mx.sym.Activation(data=bn_conv2, act_type="relu")
# # Conv-BN-Relu 3
# conv3 = mx.sym.Convolution(data=relu2, kernel=(1,1), num_filter=512, pad=(0, 0), stride=(2, 2))
# bn_conv3 = mx.symbol.BatchNorm(name='bn_conv3', data=conv3, use_global_stats=True, fix_gamma=False, eps=eps)
# relu3 = mx.sym.Activation(data=bn_conv3, act_type="relu")
# # Conv-BN-Relu 4
# conv4 = mx.sym.Convolution(data=relu3, kernel=(1,1), num_filter=1024, pad=(0, 0), stride=(2, 2))
# bn_conv4 = mx.symbol.BatchNorm(name='bn_conv4', data=conv4, use_global_stats=True, fix_gamma=False, eps=eps)
# relu4 = mx.sym.Activation(data=bn_conv4, act_type="relu")
# # Fully Connected 1
# flatten = mx.sym.Flatten(data=relu4)
# fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=10)
# # Softmax
# net = mx.sym.SoftmaxOutput(data=fc1, name='softmax')

# Building the network
print 'Building Network.'
data = mx.sym.Variable('data')
# Conv-Relu-Pool 1
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))
# Conv-Relu-Pool 2
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))
# Conv-Relu-Pool 3
conv3 = mx.sym.Convolution(data=pool2, kernel=(1,1), num_filter=50)
relu3 = mx.sym.Activation(data=conv3, act_type="relu")
# Fully Connected 1
flatten = mx.sym.Flatten(data=relu3)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=10)
# Softmax
lenet = mx.sym.SoftmaxOutput(data=fc1, name='softmax')


print 'Building Module.'
# Module
net_module = mx.mod.Module(lenet, context=mx.gpu(5))

print 'Training.'
# Train
net_module.fit(train_data=train_iter, eval_data=val_iter, optimizer_params=(('learning_rate', 0.01), ),
                 batch_end_callback=[mx.callback.Speedometer(batch_size, 100)], num_epoch=20)

print 'Testing.'
# Test
acc = mx.metric.Accuracy()
net_module.score(test_iter, acc)

print acc



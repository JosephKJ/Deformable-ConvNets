import mxnet as mx

# Character recognition using Lenet.
# Using MNIST dataset.

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# Building the network

data = mx.sym.var('data')
# Conv-Relu-Pool 1
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))
# Conv-Relu-Pool 2
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))
# Fully Connected 1
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")
# Fully Connected 2
fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=10)
# Softmax
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')


# Module
lenet_module = mx.mod.Module(lenet)

# Train
lenet_module.fit(train_data=train_iter, eval_data=val_iter,
                 batch_end_callback=mx.callback.Speedometer(batch_size, 100), num_epoch=10)

# Test
acc = mx.metric.Accuracy()
lenet_module.score(test_iter, acc)
print acc
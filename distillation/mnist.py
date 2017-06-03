import mxnet as mx
import os

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

# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x32
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
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
# Fully Connected 1
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")
# Fully Connected 2
fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=10)
# Softmax
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')


print 'Building Module.'
# Module
lenet_module = mx.mod.Module(lenet)

print 'Training.'
# Train
lenet_module.fit(train_data=train_iter, eval_data=val_iter,
                 batch_end_callback=mx.callback.Speedometer(batch_size=batch_size), num_epoch=10)

print 'Testing.'
# Test
acc = mx.metric.Accuracy()
lenet_module.score(test_iter, acc)

print acc
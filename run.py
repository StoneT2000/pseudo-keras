import network
import numpy as np
import tensorflow as tf
from activation import Sigmoid
from cost import QuadraticCost, CrossEntropyCost

# initialize network
net = network.Network([784, 32, 10], seed=0, activation=Sigmoid, cost=CrossEntropyCost, log_level=3)

# load data and rescale
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data / 255.0, test_data / 255.0

# flatten the data
train_data = train_data.reshape(len(train_data), 784, 1)
test_data = test_data.reshape(len(test_data), 784, 1)

# one hot encode data
train_labels = tf.keras.utils.to_categorical(train_labels, 10).reshape(len(train_labels), 10, 1)
test_labels = tf.keras.utils.to_categorical(test_labels, 10).reshape(len(test_labels), 10, 1)

# setup a writer to use tensorboard
writer = tf.summary.create_file_writer("./nn-logs")

net.train(train_data=train_data,
          train_labels=train_labels,
          epochs=4,
          batch_size=32,
          validation_split=0.2,
          tf_writer=writer
          )
acc = net.evaluate(data=test_data, labels=test_labels)
print(acc)

writer.close()
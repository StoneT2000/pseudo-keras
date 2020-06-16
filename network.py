import tensorflow as tf
import numpy as np
import time
import sys
import random
import os

from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from cost import Cost, QuadraticCost
from activation import Activation, Sigmoid, Relu
class Network:
    def __init__(self,
                 sizes,
                 cost: Cost = QuadraticCost,
                 activation: Activation = Sigmoid,
                 seed=random.randint(0, 2**32 - 1),
                 log_level=3
                 ):

        # store logger
        self.log = Logger(log_level)

        # set seeds
        np.random.seed(seed)
        tf.random.set_seed(0)
        random.seed(seed)

        # store number of layers, the sizes, and cost and activation
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.activation = activation

        # default initialize weights
        self.default_weight_initializer()

        pass
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            # tx = tf.constant(x, dtype='double')
            # y = tf.convert_to_tensor(y, dtype='int32')
            # x = tf.convert_to_tensor(x, dtype='int32')
            
            # weight = tf.divide(tf.random.normal((y, x)), tf.sqrt(tx))
            # self.weights.append(weight)

    def feed_forward(self, a):
        # a = tf.convert_to_tensor(a, dtype='double')
        """
        feed forward an input `a` and return the results
        """
        for b, w in zip(self.biases, self.weights):
            a = self.activation.of(np.dot(w, a)+b)
        return a
    def train(self,
              train_data=None,
              train_labels=None,
              epochs=None,
              batch_size=None,
              lmbda=0.0,
              rate=0.5,
              validation_split=0.1):
        """
        performs SGD. Expects numpy arrays
        """

        # convert to numpy arrays
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        validation_size = int(validation_split * len(train_data))
        validation_data = train_data[-validation_size:]
        validation_labels = train_labels[-validation_size:]
        train_data = train_data[:-validation_size]
        train_labels = train_labels[:-validation_size]

        val_loss_hist, val_accuracy_hist = [], []
        training_loss_hist, training_accuracy_hist = [], []

        stime = time.time()
        for epoch in range(epochs):
            # shuffle data
            p = np.random.permutation(len(train_data))
            train_data = train_data[p]
            train_labels = train_labels[p]

            # create batches of size batch_size
            batch_datas = [ train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size) ]
            batch_labels = [ train_labels[k:k + batch_size] for k in range(0, len(train_data), batch_size) ]
            stime = time.time()
            for i, (batch_data, batch_label) in enumerate(zip(batch_datas, batch_labels)):
                if i % 10 == 0:
                    self.log.info(
                    "Epoch {0}: {1}/{2}".format(epoch, i, len(batch_datas)))
                    sys.stdout.write("\033[F")
                self.update_batch(batch_data, batch_label, rate)
            sys.stdout.write("\033[K")
            self.log.info("Epoch {0}: {1}/{1}, Time: {2:>10.5}s".format(epoch, len(batch_datas), time.time() - stime))

            # calculate acc and losses
            val_acc = self.evaluate(data=validation_data, labels=validation_labels)
            val_loss = self.total_loss(data=validation_data, labels=validation_labels)
            acc = self.evaluate(data=train_data, labels=train_labels)
            loss = self.total_loss(data=train_data, labels=train_labels)

            # save into history
            val_accuracy_hist.append(val_acc)
            val_loss_hist.append(val_loss)
            training_accuracy_hist.append(acc)
            training_loss_hist.append(loss)
            
            self.log.info("accuracy: {0:.4} - loss: {1:.4} - val_accuracy: {2:.4} - val_loss: {2:.4}".format(acc, loss, val_acc, val_loss))

    def update_batch(self, batch_data, batch_label, rate):

        batch_size = len(batch_data)

        # update biases and weights
        # batch_data is a list of inputs x
        # batch_label is a list of desired outputs y

        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
        # list of nabla_ws for each layer. Each nabla_w is a sum of all the nabla_w across all samples in batch
        sum_nabla_w = [np.zeros(w.shape) for w in self.biases]

        delta_nabla_bs, delta_nabla_ws = self.backprop(batch_data, batch_label)

        for i, (_nb, _nw) in enumerate(zip(delta_nabla_bs, delta_nabla_ws)):
            for nb, nw in zip(_nb, _nw):
                sum_nabla_b[i] = sum_nabla_b[i] + nb
                sum_nabla_w[i] = sum_nabla_w[i] + nw

        # FIXME: Add regularization to weight update
        self.weights = [ w - (rate / batch_size) * nw for w, nw in zip(self.weights, sum_nabla_w) ]
        self.biases = [ b - (rate / batch_size) * nb for b, nb in zip(self.biases, sum_nabla_b) ]

    def backprop(self, batch_data, batch_label):
        nabla_b = [ np.zeros((len(batch_data),) + b.shape) for b in self.biases ]
        nabla_w = [ np.zeros((len(batch_data),) + w.shape) for w in self.weights ]

        activation = batch_data
        activations = [batch_data]
    
        # list of weighted input tensors
        weighted_inputs = []

        # feed forward
        for w, b in zip(self.weights, self.biases):
            # need to compute weighted input z
            z = np.matmul(w, activation) + b
            weighted_inputs.append(z)
            activation = self.activation.of(z)
            activations.append(activation)

        # get output error of final layer and store nablas for the final layer
        delta = self.cost.delta(weighted_inputs[-1], activations[-1], batch_label)
        nabla_b[-1] = delta

        # iterate over each sample's delta and activation at layer L - l + 1
        for i, (d, a) in enumerate(zip(delta, activations[-2])):
            nabla_w[-1][i] = np.dot(d, a.T)

        # backprop error through each layer
        for l in range(2, self.num_layers):
            sp = self.activation.prime(weighted_inputs[-l])
            
            # equal to delta of layer L - l + 1
            delta = np.matmul(self.weights[-l+1].T, delta) * sp 
            nabla_b[-l] = delta

            # iterate over each sample's delta and activation at layer L - l + 1
            for i, (d, a) in enumerate(zip(delta, activations[-l-1])):
                nabla_w[-l][i] = np.dot(d, a.T)
        
        return (nabla_b, nabla_w)

    def evaluate(self, data=None, labels=None):
        res = [ (np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in zip(data, labels) ]
        return sum( int(y1 == y0) for (y1, y0) in res ) / len(data)

    def total_loss(self, data=None, labels=None):
        loss = 0.0
        for x, y in zip(data, labels):
            loss += self.cost.fn(self.feed_forward(x), y)

        return loss / len(data)
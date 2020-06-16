import network
import numpy as np

from activation import Sigmoid
from cost import QuadraticCost, CrossEntropyCost
net = network.Network([3, 100, 5], seed=0, activation=Sigmoid, cost=CrossEntropyCost)
np.random.seed(0)

sample_size = 1000
train = np.random.randn(sample_size,3,1)
labels = np.random.randn(sample_size,5,1)

results = net.feed_forward(train[0])

net.train(train_data=train, train_labels=labels, epochs=10, batch_size=32)
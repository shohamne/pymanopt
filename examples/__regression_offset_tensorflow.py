import tensorflow as tf
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import TrustRegions, SteepestDescent, SGD
from pymanopt.manifolds import Euclidean, Product

class DataSet():
    def __init__(self,x ,y):
        self._num_examples = x.shape[0]
        self._x = x
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x = self._x[perm,:]
            self._y = self._y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._x[start:end,:], self._y[start:end,:]



if __name__ == "__main__":
    # Generate random data
    NSAMPELS=1000
    X = np.random.randn(NSAMPELS,3).astype('float32')
    Y = (X[:, 0:1] - 2*X[:,1:2] + np.random.randn(NSAMPELS,1) + 5).astype(
        'float32')

    dataset = DataSet(X,Y)

    x =  tf.placeholder("float", [None, 3])
    y =  tf.placeholder("float", [None, 1])
    # Cost function is the sqaured test error
    w = tf.Variable(tf.zeros([3, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    cost = tf.reduce_mean(tf.square(y - tf.matmul(x, w) - b))

    # first-order, second-order
    #solver = TrustRegions()
    #solver = SteepestDescent()
    solver = SGD(logverbosity=5,minstepsize=1e-15)

    # R^3 x R^1
    manifold = Product([Euclidean(3, 1), Euclidean(1, 1)])

    # Solve the problem with pymanopt
    #problem = Problem(manifold=manifold, cost=cost, arg=[w, b], verbosity=0)
    problem = Problem(manifold=manifold, cost=cost, arg=[w, b], data=[x,y], verbosity=10)
    wopt, optlog = solver.solve(problem, dataset, batch_size=100)

    print "Optimization log:"
    print optlog

    print('Weights found by pymanopt (top) / '
          'closed form solution (bottom)')

    print(wopt[0].T)
    print(wopt[1])

    print
    X1 = np.concatenate((X, np.ones((NSAMPELS, 1))), axis=1)
    wclosed = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(Y)
    print(wclosed[0:3].T)
    print(wclosed[3])


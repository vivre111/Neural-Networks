# Network.py
# (C) Jeff Orchard, 2019

import numpy as np
from copy import deepcopy
from tqdm import tqdm


#================================================
#================================================
#
# Activation Functions
#
#================================================
#================================================
class Identity(object):
    def __init__(self):
        return

    def __call__(self, x):
        self.n_samples = np.shape(x)[0]
        self.dims = np.shape(x)[-1]
        return x

    def derivative(self):
        return np.ones(shape=(self.n_samples, self.dim))


class Tanh(object):
    def __init__(self):
        self.y = 0.

    def __call__(self, x):
        self.n_samples = np.shape(x)[0]
        self.dims = np.shape(x)[-1]
        self.y = np.tanh(x)
        return self.y

    def derivative(self):
        return 1. - self.y**2


class Logistic(object):
    def __init__(self):
        self.y = 0.

    def __call__(self, x):
        self.n_samples = np.shape(x)[0]
        self.dims = np.shape(x)[-1]
        self.y = 1. / (1. + np.exp(-x))
        return self.y

    def derivative(self):
        return self.y*(1.-self.y)


class LIF(object):
    def __init__(self, tau_ref=0.002, tau_m=0.05):
        self.tau_ref = tau_ref
        self.tau_m = tau_m

    def __call__(self, x):
        self.n_samples = np.shape(x)[0]
        self.dims = np.shape(x)[-1]
        self.A = np.zeros_like(x)
        self.x_factor = np.zeros_like(x)
        # Loop over the P different sets of input currents.
        for p in range(self.n_samples):
            # Compute the theoretical firing rate for each neuron, given
            # its input current.
            for m in range(self.dims):
                if x[p,m]>1:
                    self.A[p,m] = 1 / ( self.tau_ref - self.tau_m*np.log(1-1./x[p,m]) )
                    self.x_factor[p,m] = min(100., 1. / x[p,m] / (x[p,m] - 1.))
        return self.A

    def derivative(self):
        return self.A**2 * self.tau_m * self.x_factor


#================================================
#================================================
#
# Cost Functions
#
#================================================
#================================================
class MSE(object):
    def __init__(self):
        self.dE = []

    def __call__(self, y, t):
        # [1] MSE formula
        self.n_samples = np.shape(t)[0]
        self.dim = np.shape(t)[-1]
        E = np.sum((y-t)**2)/2. / self.n_samples
        self.dE = (y-t) / self.n_samples
        return E

    def derivative(self):
        # [1] Compute the gradient of MSE w.r.t. output
        return self.dE


class MeanCrossEntropy(object):
    def __init__(self):
        self.dE = []

    def __call__(self, y, t):
        # [1] Cross Entropy formula
        self.n_samples, self.dim = np.shape(t)
        E = -np.sum(t*np.log(y) + (1-t)*np.log(1.-y)) / self.n_samples
        self.dE = (y-t) / y / (1.-y) / self.n_samples
        return E

    def derivative(self):
        # [1] Compute the gradient of MSE w.r.t. output
        return self.dE



#==================================================
#==================================================
#
# Layer Class
#
#==================================================
#==================================================
class Layer(object):
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError


class Population(Layer):
    '''
     lyr = Population(nodes, act=Identity())

     Creates a Population layer object.

     Inputs:
       nodes  the number of nodes in the population
       act    activation function (Operation object)

     Usage:
       lyr = Population(3, act=Logistic())
       h = lyr(z)
       print(lyr())   # prints current value of lyr.h
    '''
    def __init__(self, nodes, act=Logistic):
        self.nodes = nodes
        self.z = None
        self.h = None
        self.act = act()
        self.params = []

    def __call__(self, x=None):
        if x is not None:
            self.z = x
            self.h = self.act(x)
        return self.h


class Connection(Layer):

    def __init__(self, from_nodes=1, to_nodes=1, bias='zero'):
        '''
         lyr = Connection(from_nodes=1, to_nodes=1)

         Creates a layer of all-to-all connections.

         Inputs:
          from_nodes  number of nodes in source layer
          to_nodes    number of nodes in receiving layer
          bias        can be 'zero' (default), or 'random'

         Usage:
          lyr = Connection(from_nodes=3, to_nodes=5)
          z = lyr(h)
          lyr.W    # matrix of connection weights
          lyr.b    # vector of biases
        '''
        super().__init__()

        self.W = np.random.randn(from_nodes, to_nodes) / np.sqrt(from_nodes)
        if bias=='zero':
            self.b = np.zeros(to_nodes)
        else:
            self.b = np.random.randn(1, to_nodes)
        self.params = [self.W, self.b]

    def __call__(self, x=None):
        if x is None:
            print('Should not call Connection without arguments.')
            return
        P = len(x)
        if P>1:
            return x@self.W + np.outer(np.ones(P), self.b)
        else:
            return x@self.W + self.b


class DenseLayer(Layer):
    '''
     lyr = DenseLayer(from_nodes=1, to_nodes=1, act=Logistic())

     Creates a DenseLayer object, composed of 2 layer objects:
       L1  a Connection layer of connection weights, and
       L2  a Population layer, consisting of nodes that receives current
           from the Connection layer, and apply the activation function

     Inputs:
       from_nodes  how many nodes are in the layer below
       to_nodes    how many nodes are in the new Population layer
       act         activation function (Operation object)

     Usage:
       lyr = DenseLayer(from_nodes=3, to_nodes=5)
       h2 = lyr(h1)
       lyr.L1.W        # connection weights
       lyr.L2()        # activities of layer
       lyr.L2.act      # activation function of layer
    '''
    def __init__(self, from_nodes=1, to_nodes=1, act=Logistic, bias='zero'):
        '''
         lyr = DenseLayer(from_nodes=1, to_nodes=1, act=logistic)
        '''
        self.L1 = Connection(from_nodes=from_nodes, to_nodes=to_nodes, bias=bias)
        self.L2 = Population(from_nodes, act=act)

    def __call__(self, x=None):
        if x is None:
            return self.L2.h
        else:
            return self.L2(self.L1(x))






#==================================================
#==================================================
#
# Network Class
#
#==================================================
#==================================================
class Network(object):
    '''
     net = Network()

     Creates a Network object.

     Usage:
       net = Network()
       net.add_layer(L)
       ... (add more layers)
       y = net(x)
       net.lyr[1]    # reference to Layer object
    '''
    def __init__(self):
        self.lyr = []
        self.loss = None

    def add_layer(self, L):
        '''
         net.add_layer(L)

         Adds the layer object L to the network.

         Note: It is up to the user to make sure the Layer object
               fits with adjacent layers.
        '''
        self.lyr.append(L)

    def __call__(self, x):
        '''
         y = net(x)

         Feedforward pass of the network.

         Input:
           x  batch of inputs, one input per row

         Output:
           y  corresponding outputs, one per row
        '''
        for l in self.lyr:
            x = l(x)
        return x

    def learn(self, ds, lrate=1., epochs=10):
        '''
         net.Learn(ds, lrate=1., epochs=10)

         Runs backprop on the network, training on the data from
         the Dataset object ds.

         Inputs:
           ds       a Dataset object
           lrate    learning rate
           epochs   number of epochs to run
        '''
        x = ds.inputs()
        t = ds.targets()
        for epoch in range(epochs):
            y = self(x)
            cost = self.backprop(t, lrate=lrate)

            if epoch%50==0:
                # Report progress
                print(f'{epoch}: cost = {cost}')


    def sgd(self, dl, lrate=1., epochs=10):
        '''
         net.Learn(dl)

         Runs SGD on the network, training on the data from
         the dataloader dl.
        '''
        for epoch in tqdm(range(epochs)):
            dl.Reset()
            for b in dl:
                inputs = b[0]
                y = self(x)
                cost = self.backprop(b[1], lrate=lrate)


    def backprop(self, x, t, lrate=1.):
        '''
         loss = net.backprop(x, t, lrate=1.)

         Backpropagates the error gradients and updates the connection
         weights and biases.

         NOTE: This method assumes that the network state has been
               set by the corresponding forward pass.

         Inputs:
           x      batch of inputs, one per row
           t      batch of targets, one per row
           lrate  learning rate

         Output:
           loss   the cost for the batch
        '''
        y = self.lyr[-1]()   # Network output layer

        # Set up for computing the top gradient.
        loss = self.loss(y,t)
        dEdh = self.loss.derivative()

        # Work our way down through the layers
        for i in range(len(self.lyr)-1, 0, -1):

            # References to the layer below, and layer above
            pre = self.lyr[i-1]   # layer below, (i-1)
            post = self.lyr[i]    # layer above, (i)
            # Note that:
            #   post.L1.W contains the connection weights
            #   post.L1.b contains the biases
            #   post.L2.z contains the input currents
            #   post.L2.h contains the upper layer's activities

            # Compute dEdz from dEdh
            #dEdz = dEdh
            dEdz = post.L2.act.derivative() * dEdh

            # Parameter gradients
            #dEdW = np.zeros_like(post.L1.W)
            #dEdb = np.zeros_like(post.L1.b)
            dEdW = pre().T @ dEdz
            dEdb = np.sum(dEdz, axis=0)

            # Project gradient through connection, to layer below
            #dEdh = np.zeros_like(pre.h)
            dEdh = dEdz @ post.L1.W.T

            # Update weight parameters
            post.L1.W = post.L1.W - lrate*dEdW
            post.L1.b = post.L1.b - lrate*dEdb

        return loss


# end

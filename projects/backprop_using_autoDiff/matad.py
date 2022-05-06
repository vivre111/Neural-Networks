# matad.py
# (C) Jeff Orchard, University of Waterloo, 2021

import numpy as np
import copy



'''
=========================================

 mat

=========================================
'''
class Mat(object):
    def __init__(self, val):
        '''
         v = Mat(val)

         Creates a Mat object for the 2D numpy array val.

         Inputs:
           val   a 2D numpy array, dimensions DxN

         Output:
           v     object of Mat class

         Then, we can get its value using any of:
           v.val
           v()
         both of which return a numpy array.

         The member v.creator is either None, or is a reference
         to the MatOperation object that created x.
         If v was creates by a MatOperation, then v.evaluate()
         will re-evaluate that MatOperation (and the subgraph below it).

         You can set the value using one of:
           v.val = np.array([[1,2],[3,4.]])
           v.set(np.array([[1,2],[3,4.]]))

         The object also stores a derivative in v.grad. It is the
         derivative of the expression with respect to v.

         Usage:
           v = Mat(np.array([[1,2],[3,4.]]))
           len(v)  # returns number of rows
           v()     # returns v.val (a numpy array)
           v.evaluate() # re-evaluates the creator, if there is one
        '''
        self.val = np.array(copy.deepcopy(val), ndmin=2)
        self.rows, self.cols = np.shape(self.val)
        self.grad = np.zeros_like(self.val)
        self.creator = None

    def set(self, val):
        self.val = np.array(val, dtype=float, ndmin=2)
        #self.val[:,:] = val[:,:]

    def evaluate(self):
        if self.creator!=None:
            self.val = self.creator.evaluate()
        return self.val

    def set_creator(self, op):
        self.creator = op

    def zero_grad(self):
        self.grad = np.zeros_like(self.val)
        if self.creator!=None:
            self.creator.zero_grad()

    def backward(self, s=None):
        if s is None:
            s = np.ones_like(self.val)
        self.grad = self.grad + s
        if self.creator!=None:
            self.creator.backward(s)

    def __len__(self):
        return self.rows

    def __call__(self):
        return self.val

    def __add__(self, b):
        '''
         Wrapper so that 'v + b' is the same as 'v.plus(b)'
        '''
        return plus(self, b)

    def __sub__(self, b):
        '''
         Wrapper so that 'v - b' is the same as 'v.minus(b)'
        '''
        return minus(self, b)

    def __mul__(self, b):
        '''
         Wrapper so that 'v * b' is the Hadamard product.
        '''
        return Hadamard([self,b])()
        #return hadamard(self, b)

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return f'Mat({self.val})'

'''
=========================================

 Wrapper Functions

=========================================
'''
# Binary functions ====================
def plus(a, b):
    '''
     v = plus(a, b)

     Adds two Mat objects, element-by-element.
    '''
    assert a.rows==b.rows, 'plus: dimension mismatch'
    assert a.cols==b.cols, 'plus: dimension mismatch'
    return Plus([a,b])()

def minus(a, b):
    '''
     v = minus(a, b)

     Subtracts two Mat objects, element-by-element.
    '''
    return Minus([a,b])()

def hadamard(a, b):
    return Hadamard([a,b])()

# Activation functions =====================
# Unary functions
def tanh(a):
    '''
     v = tanh(a)

     Applies the hyperbolic tangent function to each element.
    '''
    return Tanh([a])()

def identity(a):
    '''
     v = identity(a)

     Applies the identity function to each element.
    '''
    return Identity([a])()

def relu(a):
    '''
     v = relu(a)

     Applies the ReLU function to each element.
    '''
    return ReLU([a])()

# Vector activation functions
def softmax(a):
    '''
     v = softmax(a)

     Applies the softmax function to each row of a.
    '''
    return Softmax([a])()

# Reducing functions
def sum(a, axis=None):
    '''
     v = sum(a, axis=None)

     Adds up the elements.

     Inputs:
       a     Mat object, dimensions PxN (ie. P samples)
       axis  which axis to sum along

     Output:
       v     Mat object

     If axis=1, it adds within samples (output is Px1)
     If axis=0, it adds across samples (output is 1xN)
     If axis=None, it adds everything (output is 1x1)
    '''
    return Sum([a], axis=axis)()

def batch_mean(a):
    '''
     v = batch_mean(a)

     Computes the mean over batches. If the input is PxN (ie. P samples)
     then the output is 1xN.
    '''
    return BatchMean([a])()


# Loss functions ========
def categorical_ce(a, target):
    '''
     E = categorical_ce(a, target)

     Computes the mean categorical cross-entropy.
     The categorical CE is applied to each sample (along each row),
     and the mean is taken across samples (down the resulting column).
     The targets are assumed to be one-hot vectors in each row.
     The output, E, is a scalar.
    '''
    return CategoricalCE([a], target)()

# Loss functions ========
def mean_ce(a, target):
    '''
     E = mean_ce(a, target)

     Computes the mean cross-entropy.
     The cross-entropy is applied to each sample (along each row),
     and the mean is taken across samples (down the resulting column).
     The output, E, is a 1x1 Mat scalar.
    '''
    c1 = CrossEntropy([a], target)()
    return BatchMean([c1])()

def mse(a, target):
    '''
     v = mean_ce(a, target)

     Computes the mean squared error.
     The squared error is applied to each sample (along each row),
     and the mean is taken across samples (down the resulting column).

     Inputs:
      a       Mat object of dim (D,K)
      target  a (D,K) NumPy array of target values

     Output:
      v       a (1x1) Mat object
    '''
    #return batch_mean( squared_error(a, target) )
    return MSE([a], target)()


'''
=========================================

 MatOperation

=========================================
'''
class MatOperation():
    '''
     op = MatOperation(args)

     MatOperation is an abstract base class for mathematical operations
     on matrices.

     Inputs:
       args  list of Mat objects

     Output:
       op    a MatOperation object

     The MatOperation object op stores its arguments in the list op.args,
     and has the functions:
       op.__call__()
       op.evaluate()
       op.zero_grad()
       op.backward()

     Usage:
       op()  # evaluates the operation without re-evaluating the args
       op.evaluate()  # re-evaluates the op after re-evaluating the args
       op.zero_grad() # resets grad to zero for all the args
       op.backward()  # propagates the derivative to the Vars below
    '''
    def __init__(self, args):
        self.args = args

    def __call__(self):
        raise NotImplementedError

    def evaluate(self):
        for a in self.args:
            a.evaluate()
        val = self()
        return val()

    def zero_grad(self):
        for a in self.args:
            a.zero_grad()

    def backward(self, s=1.):
        raise NotImplementedError


'''
=========================================

 Operation Implementations

=========================================
'''

class Plus(MatOperation):
    def __call__(self):
        v = Mat(self.args[0].val + self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(s*np.ones_like(self.args[1].val))

class Minus(MatOperation):
    def __call__(self):
        v = Mat(self.args[0].val - self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(-s*np.ones_like(self.args[1].val))

class Hadamard(MatOperation):
    def __call__(self):
        v = Mat(self.args[0].val*self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*self.args[1].val)
        self.args[1].backward(s*self.args[0].val)

# Activation Functions

class Tanh(MatOperation):
    def __call__(self):
        v = Mat(np.tanh(self.args[0].val))
        self.deriv = 1. - v.val**2
        v.creator = self
        return v

    def backward(self, s=1.):
        deriv = s * self.deriv
        self.args[0].backward(deriv)

class Identity(MatOperation):
    def __call__(self):
        v = Mat(self.args[0].val)
        v.creator = self
        return v

    def backward(self, s=1.):
        self.args[0].backward(s)

class ReLU(MatOperation):
    def __call__(self):
        v = Mat(np.clip(self.args[0].val, 0, None))
        v.creator = self
        return v

    def backward(self, s=1.):
        val = np.ceil( np.clip(self.args[0].val, 0, 1) )
        self.args[0].backward(s*val)

class Softmax(MatOperation):
    '''
     act = Softmax()

     Creates a MatOperation object that represents the softmax
     function. The softmax is applied to the rows of the input.

     Usage:
      act = Softmax()
      act([[0., 0.5]])
    '''
    def __call__(self):
        num = np.exp(self.args[0].val)
        denom = np.sum(num, axis=1)
        self.y = num / np.tile(denom[:,np.newaxis], [1,np.shape(num)[1]])
        v = Mat(self.y)
        v.creator = self
        return v

    def backward(self, s):
        '''
         act.derivative(s)

         Computes and the derivative of the softmax function.
         Note that the __call__ function must be called before this
         function can be called.

         Input:
           s       NumPy array the same size as z, which multiplies the
                   derivative
                   NOTE: s is a mandatory argument (not optional)
                   NOTE: s should have only a single non-zero element
        '''
        #print(f'Softmax.backward: {self.args[0].val.shape}')
        #print(s)
        idx = np.nonzero(s)[1]
        y = self.y
        s_gamma = np.zeros_like(s)
        y_gamma = np.zeros_like(y)
        kronecker = np.zeros_like(s)
        for k,gamma in enumerate(idx):
            s_gamma[k,:] = s[k,gamma]
            y_gamma[k,:] = y[k,gamma]
            kronecker[k,gamma] = 1.
        dydz = s_gamma*y_gamma*(kronecker-y)
        self.args[0].backward(dydz)



# Reducing functions

class Sum(MatOperation):
    def __init__(self, args, axis=None):
        super().__init__(args)
        self.axis = axis

    def __call__(self):
        v = Mat(np.sum(self.args[0].val, axis=self.axis, keepdims=True))
        v.creator = self
        return v

    def backward(self, s=1.):
        self.args[0].backward(s*np.ones_like(self.args[0].val))


class BatchMean(MatOperation):
    '''
     Computes the mean over batches. If the input is PxN (ie. P samples)
     then the output is 1xN.
    '''
    def __call__(self):
        P = self.args[0].rows
        v = Mat(np.sum(self.args[0].val, axis=0, keepdims=True)/P)
        v.creator = self
        return v

    def backward(self, s=1.):
        self.args[0].backward(s*np.ones_like(self.args[0].val)/self.args[0].rows)



# Functions with additional parameters ========

class MSE(MatOperation):
    def __init__(self, arg, target):
        super().__init__(arg)
        self.target = target

    def __call__(self):
        self.diff = self.args[0].val - self.target
        v = Mat( 0.5 * np.sum(self.diff**2, keepdims=True) / self.args[0].rows )
        v.creator = self
        return v

    def backward(self, s=1.):
        self.args[0].backward(s*self.diff/self.args[0].rows)


class CategoricalCE(MatOperation):
    def __init__(self, arg: Mat, target):
        super().__init__(arg)
        self.target = target  # numpy array

    def set_target(self, target):
        self.target = target

    def __call__(self):
        v = Mat(-np.sum(self.target * np.log(self.args[0].val)) / len(self.target))
        v.creator = self
        return v

    def backward(self, s=1.):
        #print(f'CategoricalCE.backward: {self.args[0].val.shape}')
        self.args[0].backward( s * -self.target/self.args[0].val / len(self.target) )

__author__ = 'joerg'





######                           Imports
########################################
import theano
import theano.tensor as T
import numpy as np

from recnet.error_function import CTC

error = CTC()

network_output = T.tensor3(dtype=theano.config.floatX)
true_output = T.imatrix()


cost = error.output_error(network_output, true_output)


f = theano.function(
        inputs = [network_output,true_output],
        outputs = cost,
    allow_input_downcast=True
    )

Y = T.ivector('Y')
predict = T.matrix(dtype=theano.config.floatX)
cost = error.ctc_cost(predict, Y)

n = theano.function(
        inputs = [predict,Y],
        outputs = cost,
    allow_input_downcast=True
    )

a = np.eye(11)[:,:10]
a.shape

b = np.arange(10,dtype=np.int32)
b.shape
n(a,b)


### Test with noise input -> high cost

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div



net_out = np.random.uniform(0,1,[32,10])
net_out = softmax(net_out)
net_out = np.reshape(net_out,[32,1,10]).astype(theano.config.floatX)

true_out = np.random.randint(0,6,7)
true_out = np.append(true_out,true_out.__len__())
true_out = np.reshape(true_out,[1,8]) #.astype(theano.config.floatX)
f(net_out, true_out)


### Test with constructed input -> high cost


net_out = np.array([ [0, 0, 0, 0, 0, 4, 0, 0.1, 0, 0.1],
                     [0, 4, 0, 0, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 4, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 1, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 4, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 0, 4, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0.1],
                     [0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0.1],
                     [0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 4, 0, 0, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0.1],
                     [0, 6, 0, 0, 0, 0, 0, 0.1, 0, 0.1],])


net_out = softmax(net_out)
net_out[1,:]
net_out = np.reshape(net_out,[net_out.shape[0],1,net_out.shape[1]]).astype(theano.config.floatX)
net_out[1,0,:]

true_out = np.array([5,1,5,3,5,6,5,1,8])
true_out = np.random.randint(0,10,9)
true_out = np.append(true_out,true_out.__len__())
true_out = np.reshape(true_out,[1,true_out.shape[0]]) #.astype(theano.config.floatX)
f(net_out, true_out)

n(net_out, true_out)

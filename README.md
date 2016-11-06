
# RecNet - Recurrent Neural Network Framework

[![Build Status](https://travis-ci.org/joergfranke/recnet.svg?branch=master)](https://travis-ci.org/joergfranke/recnet)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/recnet/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)
[![Theano](https://img.shields.io/badge/theano-0.8.2-yellow.svg)](http://deeplearning.net/software/theano/)

## About
*RecNet* is a framework for recurrent neural networks. It implements a deep uni/bidirectional Conventional/LSTM/GRU architecture in Python with use of the
[Theano](http://deeplearning.net/software/theano/) library. The intension is a easy to use, light weight and flat implementation with
the opportunity to check out new ideas and to implement the latest research.

__Current implemented features:__

- Conventional Recurrent Layers (tanh/relu activation)
- LSTM (with and without peepholes) and GRU
- uni/bidirectional Training
- Layer Normalization
- Softmax Output
- SGD, Nesterov momentum, RMSprop and AdaDelta optimization
- Dropout Training
- Cross-Entropy Loss and Weighted Cross-Entropy Loss
- normal and log Connectionist Temporal Classification
- Regularization (L1/L2)
- Noisy Inputs
- Mini Batch Training



__Example of use:__

- [Little timer task](https://github.com/joergfranke/recnet/tree/master/examples/little_timer_task)
- [Phoneme Recognition](https://github.com/joergfranke/phoneme_recognition)


## How to install it

```bash
git clone https://github.com/joergfranke/recnet.git
cd recnet
python setup.py install
```

*In case of error try to update pip/setuptools.*

## How to use it

__1.__
Please provide our data in form of two lists and storage it in a klepto file. One list contains sequences of features
and another the corresponding targets. Each element of the list should be a matrix with shape `sequence length | feature/target size` .

```bash
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = input_features #example shape [ [123,26] , [254,26] , [180,26] , [340,26] , ... ]
    d['y'] = output_targets #example shape [ [123,61] , [254,61] , [180,61] , [340,61] , ... ]
    d.dump()
    d.clear()
```

__2.__
Define the parameters for the recurrent neural network

| Parameter           | Description                                        | Value          |
| ------------------- | ---------------------------------------------------| ---------------- |
| train_data_name     | Name of the training data set | String |
| valid_data_name     | Name of the validation data set | String |
| data_location     | Path/dictionary to the data set in kelpto files | Path |
| batch_size     | Size of the mini batches | Integer >=1 |
| net_size            | input size, size of each hidden layer, output size | List of integer |
| net_unit_type       | unit type of each layer (input, GRU, LSTM, conv, GRU_ln ...) | List of unit types |
| net_act_type        | activation function of each layer (tanh, relu, softplus) | List of activation functions |
| net_arch            | architecture of each layer (unidirectional, bidirectional, feed forward)  | List of architectures |
| epochs             | Number of epochs to train                          | Integer >=1          |
| learn_rate         | Lerning rate for optimization algorithm            | Float [0.0001...0.5] |
| optimization       | Optimization algorithm                             | "sgd" / "rmsprop" / "nesterov_momentum" / "adadelta" |
| use_dropout        | Use of dropout between layers vertical             | False/True       |
| regularization     | Use of regularization (L1/L2)                      | False/"L1"/"L2"  |
| noisy_input        | Add noise to the input                             | True/False          |
| loss_function      | Loss/Error function (weighted or normal ce)        | "w2_cross_entropy" / "cross_entropy"          |

*View documentation for full list of RecNet parameters*

__3.__
- The framework contains the `rnnModel` which provides functions to train, validate and test the model.
The `rnnModel` takes the parameters.
```bash
from recnet.build_model import rnnModel
model = rnnModel(parameter)
train_fn    = model.get_training_function()
valid_fn    = model.get_validation_function()
forward_fn  = model.get_forward_function()
```

| Function | Arguments | Return   |
|----------|-----------|----------|
| `train_fn` | features, targets, mini  batch mask | training error, network output |
| `valid_fn` | features, targets, mini batch mask | validation error, network output |
| `forward_fn` | features, mini batch mask | network output |

Difference between train_fn and valid_fn is no use of dropout, noise or weight update.


- Please feel free to orient oneself on the [example](https://github.com/joergfranke/recnet/tree/master/examples/little_timer_task) provided in this repository.


## Credits
* Theano implementation of CTC by [Shawn Tan](https://github.com/shawntan/theano-ctc/), [Rakesh Var](https://github.com/rakeshvar/rnn_ctc) and [Mohammad Pezeshki](https://github.com/mohammadpz)


## Further work

- Extend documentation
- Add tests
- Implementations:
    - CTC
    - Mix of SGD and others like AdaDelta

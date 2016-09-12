
# RecNet - Recurrent Neural Network Framework

[![Build Status](https://travis-ci.org/joergfranke/recnet.svg?branch=master)](https://travis-ci.org/joergfranke/recnet)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/recnet/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)
[![Theano](https://img.shields.io/badge/theano-0.8.2-yellow.svg)](http://deeplearning.net/software/theano/)

## About
*RecNet* is a framework for recurrent neural networks. It implements a deep bidirectional LSTM neural network in Python with use of the
[Theano](http://deeplearning.net/software/theano/) library. The intension is a light weight flat implementation with
the opportunity to check out easily new ideas and to implement the latest research.

__Current implemented features:__
- bidirectional LSTM Layers
- Softmax
- RMSprop
- AdaDelta
- Dropout
- Identity function layer-wise
- Weighted cross-entropy loss
- Noisy inputs



__Example of use:__

<table>
  <tr>
    <td><a href="https://github.com/joergfranke/recnet/tree/master/examples/little_timer_task">Little timer task</a></td>
    <td><img src="examples/little_timer_task/sample.png"  width="250"></td>
  </tr>
</table>


## How to install it

```bash
git clone https://github.com/joergfranke/recnet.git
cd recnet
python setup.py install
```

## How to use it

- Please provide our data in form of two lists and save it in a klepto file. One list contains sequences of features
and another the corresponding targets.

```bash
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = input_features #example shape [ [123,26] , [254,26] , [180,26] , [340,26] , ... ]
    d['y'] = output_targets #example shape [ [123,61] , [254,61] , [180,61] , [340,61] , ... ]
    d.dump()
    d.clear()
```


- Use the data handler from the framework to create mini batches. It deals with sequences with different length.
Therefore it shuffles the sequences and sorts them in respect to there length. Now mini batches gets build and shorter
sequences will be pad with zeros. Additionally the data handler creates masks for the padding zeros which are used in
training. The data handler provides the function `load_minibatches` which takes the data set location, data set name and
mini batch size.
```bash
from recnet.data_handler import load_minibatches
data_set_x,data_set_y,data_set_mask = load_minibatches(data_location, data_name, mini_batch_size)
```


- Define the parameters for the recurrent neural network

| Strucutre Parameter | Describtion                                        | Value          | 
| ------------------- | ---------------------------------------------------| ---------------- | 
| net_size            | input size, size of each hidden layer, output size | List of integer | 
| hidden_layer        | number of hidden layers                            | Integer          | 
| bi_directional      | Use bidirectional architecture, layer size gets split in forward and backward layer  | True/False |
| identity_func       | Identity function parallel to hidden layer | float [0...1] | 
| train_set_len       | Length of training set (number of batches) | Integer | 
| valid_set_len       | Length of validation set (number of batches) | Integer | 
| output_location     | Path/dictionary for saving the log/prm files | Path | 
| output_type         | Log during training in console, log-file or both | "console"/"file"/"both" | 


| Optimization Parameter | Describtion                                        | Value          | 
| ------------------- | ---------------------------------------------------| ---------------- | 
| epochs             | Number of epochs to train                          | Integer          | 
| learn_rate         | Lerning rate for optimization algorithm            | Float [0.0001...0.5] | 
| momentum           | Momentum for some optimization algorithms          | Float [0...1]    | 
| decay_rate         | Decay rate for some optimization algorithms        | Float [0...1]    | 
| use_dropout        | Use of dropout between layers vertical             | True/False       | 
| dropout_level      | Probability of dropout                             | Float [0...1]    | 
| regularization     | Use of regularization (L1/L2)                      | False/"L1"/"L2"  | 
| reg_factor         | Influence of regularization                        | Float [0...1]    | 
| optimization       | Optimization algorithm                             | "sgd" / "rmsprop" / "nesterov_momentum" / "adadelta" |
| noisy_input        | Add noise to the input                             | True/False          | 
| noise_level        | Factor for noise level                             | Float [0...1]    | 
| loss_function      | Loss/Error function (weighted or normal ce)        | "w2_cross_entropy" / "cross_entropy"          |
| bound_weight       | Weight for weighted cross entropy                  | Integer          | 



- The framework contains the `rnnModel` which provides functions to train, validate and test the model.
The `rnnModel` takes the structure and optimization parameters as a dictionary and a random number stream for numpy `rng` and theano `trng`.
```bash
from recnet.build_model import rnnModel
rnn = rnnModel(prm_structure, prm_optimization, rng, trng)
train_fn    = rnn.get_training_function()
valid_fn    = rnn.get_validation_function()
forward_fn  = rnn.get_forward_function()
```

| Function | Arguments | Return   |
|----------|-----------|----------|
| `train_fn` | features, targets, minibatch mask | training error, network output |  
| `valid_fn` | features, targets, minibatch mask | validation error, network output | 
| `forward_fn` | features, minibatch mask | network output | 

Difference between train_fn and valid_fn is no use of dropout or noise.



- Feel free to orient oneself on the [example](https://github.com/joergfranke/recnet/tree/master/examples/little_timer_task) provided in this reposetory.

## Further work

- Extend documentation
- Add tests
- Implementations:
    - Layer-Normalization
    - GPU Layers
    - Mix of SGD and others like AdaDelta

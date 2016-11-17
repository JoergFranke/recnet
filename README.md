
# RecNet - Recurrent Neural Network Framework

[![Build Status](https://travis-ci.org/joergfranke/recnet.svg?branch=master)](https://travis-ci.org/joergfranke/recnet)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/recnet/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)
[![Theano](https://img.shields.io/badge/theano-0.8.2-yellow.svg)](http://deeplearning.net/software/theano/)

## About
`RecNet` is a easy to use framework for recurrent neural networks. It implements a deep uni/bidirectional
Conventional/LSTM/GRU architecture in Python with use of the [Theano](http://deeplearning.net/software/theano/)
library. The intension is a easy handling, light weight implementation with the opportunity to check out new
ideas and to implement the current research.

__Current implemented features:__

- Conventional Recurrent Layers (tanh/relu activation)
- LSTM (with and without peepholes) and GRU [1,2]
- uni/bidirectional Training
- Layer Normalization [3]
- Softmax Output
- SGD, Nesterov momentum, RMSprop and AdaDelta optimization [4, 5]
- Dropout Training [6]
- Cross-Entropy Loss and Weighted Cross-Entropy Loss
- normal and log Connectionist Temporal Classification [7]
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
    d = klepto.archives.file_archive("train_data_set.klepto")
    d['x'] = input_features #example shape [ [123,26] , [254,26] , [180,26] , [340,26] , ... ]
    d['y'] = output_targets #example shape [ [123,61] , [254,61] , [180,61] , [340,61] , ... ]
    d.dump()
    d.clear()
```

__2.__
Instantiate RecNet, define parameters and create model.

```bash
rn = rnnModel()
rn.parameter["train_data_name"] = "train_data_set.klepto"
rn.parameter["net_size"      ] = [      2,     10,         2]
rn.parameter["net_unit_type" ] = ['input',  'GRU', 'softmax']
rn.parameter["net_arch"      ] = [    '-',    'bi',     'ff']
rn.parameter["optimization"  ] = "adadelta"
rn.parameter["loss_function" ] = "cross_entropy"
rn.create()
```
*Please find a full list of possible parameters below.*

__3.__
Use the provided function for generating mini batches, training, validation or usage.
```bash
mb_train_x, mb_train_y, mb_mask = rn.get_mini_batches("train")
for j in range(train_batch_quantity):
    net_out, train_error = rn.train_fn( mb_train_x[j], mb_train_y[j], mb_mask[j] )
```
*Please find complete training and usage scripts in the provided examples.*

## Documentation

#### Parameters

| Parameter           | Description                                        | Value          |
| ------------------- | ---------------------------------------------------| ---------------- |
| train_data_name     | Name of the training data set | String |
| valid_data_name     | Name of the validation data set | String |
| data_location     | Path/dictionary to the data set in kelpto files | Path |
| batch_size     | Size of the mini batches | Integer >=1 |
| output_location     | Path/dictionary for saving the log/prm files | Path |
| output_type         | Log during training in console, log-file or both | "console"/"file"/"both" |
| net_size            | input size, size of each hidden layer, output size | List of integer |
| net_unit_type       | unit type of each layer (input, GRU, LSTM, conv, GRU_ln ...) | List of unit types |
| net_act_type        | activation function of each layer (tanh, relu, softplus) | List of activation functions |
| net_arch            | architecture of each layer (unidirectional, bidirectional, feed forward)  | List of architectures |
| epochs             | Number of epochs to train                          | Integer >=1          |
| learn_rate         | Lerning rate for optimization algorithm            | Float [0.0001...0.5] |
| optimization       | Optimization algorithm                             | "sgd" / "rmsprop" / "nesterov_momentum" / "adadelta" |
| momentum           | Momentum for some optimization algorithms          | Float [0...1]    |
| decay_rate         | Decay rate for some optimization algorithms        | Float [0...1]    |
| use_dropout        | Use of dropout between layers vertical             | False/True       |
| dropout_level      | Probability of dropout                             | Float [0...1]    |
| regularization     | Use of regularization (L1/L2)                      | False/"L1"/"L2"  |
| reg_factor         | Influence of regularization                        | Float [0...1]    |
| noisy_input        | Add noise to the input                             | True/False          |
| noise_level        | Factor for noise level                             | Float [0...1]    |
| loss_function      | Loss/Error function (weighted or normal ce)        | w2_cross_entropy/cross_entropy/CTC/CTClog|
| bound_weight       | Weight for weighted cross entropy                  | Integer          |


#### Functionality

| Function | Describtion | Arguments | Return   |
|----------|-------------|-----------|----------|
| `create` | Create model and compile functions | List of function to compile ['train','valid','forward']| - |
| `pub` | Publish in console or log-file | String of text  | - |
| `get_mini_batches` | Create model and compile functions | Name of data set 'train'/'valid'/'test'| - |
| `dump` | Make a dump of current model | -  | - |
| `train_fn` | Train model with mini batch | features, targets, mask | training error, network output |
| `valid_fn` | Determine validation error without update | features, targets, mask | validation error, network output |
| `forward_fn` | Determin output based on mini batch | features, mask | network output |


## Credits
* Theano implementation of CTC by [Shawn Tan](https://github.com/shawntan/theano-ctc/), [Rakesh Var](https://github.com/rakeshvar/rnn_ctc) and [Mohammad Pezeshki](https://github.com/mohammadpz)

## References

1. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
2. Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).
3. Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
4. Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
5. Hinton, Geoffrey, N. Srivastava, and Kevin Swersky. "Lecture 6a Overview of mini-‐batch gradient descent." Coursera Lecture slides https://class. coursera. org/neuralnets-2012-001/lecture,[Online.
6. Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural network regularization." arXiv preprint arXiv:1409.2329 (2014).
7. Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." Proceedings of the 23rd international conference on Machine learning. ACM, 2006.


## Further work

- Extend documentation
- Add tests
- Implementations:
    - CTC decoder
    - Lern initialization
    - Annealed Gradient Descent
    - Mix of SGD and others like AdaDelta

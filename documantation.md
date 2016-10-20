




| Parameter           | Description                                        | Value          |
| ------------------- | ---------------------------------------------------| ---------------- |
| net_size            | input size, size of each hidden layer, output size | List of integer |
| net_unit_type       | unit type of each layer (input, GRU, LSTM, conv, GRU_ln ...) | List of unit types |
| net_act_type        | activation function of each layer (tanh, relu, softplus) | List of activation functions |
| net_arch            | architecture of each layer (unidirectional, bidirectional, feed forward)  | List of architectures |



| output_location     | Path/dictionary for saving the log/prm files | Path |
| output_type         | Log during training in console, log-file or both | "console"/"file"/"both" |
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
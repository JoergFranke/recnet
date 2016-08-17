
## Little timer task

Based on example designed by [Herbert Jaeger](http://www.pdx.edu/sites/www.pdx.edu.sysc/files/Jaeger_TrainingRNNsTutorial.2005.pdf).
This task has two imput signal, a start signal and a duration signal. The target output is a signal which starts at each start signal peek and builds a rectangular signal with length of duration signal value at starting peek.

<table>
  <tr>
    <td><img src="sample.png" ></td>
  </tr>
</table>


### How to use it

1. Generate data set
    - Run `make_data_set.py`
2. Train model
    - Run `train_model.py`
3. Use model
    - Add name of the network parameter file from outcome to `use_model.py`
    - Run `use_model.py`


### Hints

For training its usefull to have minimum two output signal. So in this example we use the target output and the inverse output. Otherwise cross entropy does't work.


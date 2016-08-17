#!/usr/bin/env python
__author__ = 'Joerg Franke'


import klepto
import numpy as np
import matplotlib.pyplot as plt





def make_start_signal(rng, length):
    start_time = rng.randint(0,10,1)

    start_signal = np.zeros(length)
    i = start_time
    while(i<length):
        start_signal[i] = 1
        i += rng.randint(10,15,1)

    return start_signal


def make_duration_signal(rng, length):
    duration_signal = np.zeros(length)
    for i in xrange(0,length, 1):
        duration_signal[i] = rng.randint(1,9,1)
    return duration_signal


def make_target_signal(start_signal, duration_signal):
    target_signal = np.zeros([start_signal.shape[0], 2])
    counter = 0
    for i in xrange(target_signal.shape[0]):
        if start_signal[i] == 1:
            counter = duration_signal[i]
        if counter > 0:
            target_signal[i, 0] = 1
            counter -= 1
    target_signal[:,1] = 1 - target_signal[:,0]
    return target_signal


def make_data_set(rng, samples):
    input_data = []
    output_data = []
    for i in xrange(samples):
        length = rng.randint(100,200,1)
        start_signal = make_start_signal(rng, length)
        duration_signal = make_duration_signal(rng, length)
        target_signal = make_target_signal(start_signal, duration_signal)
        input_data.append(np.concatenate([start_signal.reshape([length,1]),duration_signal.reshape([length,1])],axis=1))
        output_data.append(target_signal)
    return input_data, output_data


def make_klepto_file(set_name, input_data, output_data):
    file_name = "data_set/little-timer_" + set_name + ".klepto"
    print("data set name: " + file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = input_data
    d['y'] = output_data
    d.dump()
    d.clear()




def print_sample(sample_no, input_data, output_data):
    fig = plt.figure()
    fig.suptitle('Little timer task - Sample')
    plt.subplot(2,2,1)
    plt.xlabel('start signal')
    plt.plot(input_data[sample_no][:,0])
    plt.ylim([0,1.1])
    plt.xlim([0,80])
    plt.subplot(2,2,3)
    plt.xlabel('duration signal')
    plt.plot(input_data[sample_no][:,1])
    plt.ylim([0,9.1])
    plt.xlim([0,80])
    plt.subplot(1,2,2)
    plt.xlabel('target signal')
    plt.plot(output_data[sample_no][:,0])
    plt.ylim([0,1.1])
    plt.xlim([0,80])


if __name__ == "__main__":

    train_samples = 10000
    valid_samples = 500
    rng = np.random.RandomState(100)
    input_data, output_data = make_data_set(rng, train_samples)
    make_klepto_file('train', input_data, output_data)
    input_data, output_data = make_data_set(rng, valid_samples)
    make_klepto_file('valid', input_data, output_data)
    input_data, output_data = make_data_set(rng, valid_samples)
    make_klepto_file('test', input_data, output_data)

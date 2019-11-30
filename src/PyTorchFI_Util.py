"""
Copyright (c) 2019 University of Illinois
All rights reserved.

Developed by:
                          RSIM Research Group
                          University of Illinois at Urbana-Champaign
                        http://rsim.cs.illinois.edu/
"""

import os.path
import time
import torch
import torch.nn as nn
import random

from pytorchfi import PyTorchFI_Core as core

DEBUG = False

# replace specified layer of weights with zeroes
def zero_layer_weights(zero_layer=0):
    return core.declare_weight_fi(layer=zero_layer, zero=True)

# generates a random neuron injection (default value range [-1, 1]) in every layer of each batch element
def random_inj_per_layer(min_val=-1, max_val=1):
    conv_num = []
    batch = []
    c_rand = []
    w_rand = []
    h_rand = []
    value = []
    for i in range(core.get_total_batches()):
        for j in range(core.get_total_conv()):
            conv_num.append(j)
            batch.append(i)
            c_rand.append(random.randint(0, core.get_fmaps_num(j) - 1))
            h_rand.append(random.randint(0, core.get_fmaps_H(j) - 1))
            w_rand.append(random.randint(0, core.get_fmaps_W(j) - 1))
            value.append(random.randint(min_val, max_val))
    return core.declare_neuron_fi(conv_num=conv_num, batch=batch, c=c_rand, h=h_rand, w=w_rand, value=value)


# generates a single neuron random injection (default value range [-1, 1]) in each batch element
def random_inj(min_val=-1, max_val=1):
    conv_num = []
    batch = []
    c_rand = []
    h_rand = []
    w_rand = []
    value = []
    for i in range(core.get_total_batches()):
        conv_num.append(random.randint(0, core.get_total_conv() - 1))
        batch.append(i)
        c_rand.append(random.randint(0, core.get_fmaps_num(conv_num[i]) - 1))
        h_rand.append(random.randint(0, core.get_fmaps_H(conv_num[i]) - 1))
        w_rand.append(random.randint(0, core.get_fmaps_W(conv_num[i]) - 1))
        value.append(random.randint(min_val, max_val))
    return core.declare_neuron_fi(conv_num=conv_num, batch=batch, c=c_rand, h=h_rand, w=w_rand, value=value)


def compare_golden(input_data):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utilcomparegolden/
    """
    softmax = nn.Softmax(dim=1)

    model = core.get_original_model()
    golden_output = model(input_data)
    golden_output_softmax = softmax(golden_output)
    golden = list(torch.argmax(golden_output_softmax, dim=1))

    corrupted_model = core.get_corrupted_model()
    corrupted_output = corrupted_model(input_data)
    corrupted_output_softmax = softmax(corrupted_output)
    corrupted = list(torch.argmax(corrupted_output_softmax, dim=1))

    return [golden, corrupted]


def time_model(model, input_data, iterations=100):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utiltimemodel/
    """
    start_time = time.time()
    for i in range(iterations):
        model(input_data)
    end_time = time.time()
    return (end_time - start_time) / iterations


def random_batch_fi_gen(conv_number, fmap_number, H_size, W_size, min_value, max_value):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utilrandfigen/
    """
    conv_fi = [conv_number] * core.get_total_batches()
    batch_fi = list(range(core.get_total_batches()))
    c_fi = [fmap_number] * core.get_total_batches()
    h_fi = []
    w_fi = []
    val_fi = []

    for i in range(core.get_total_batches()):
        h_fi.append(random.randint(0, H_size - 1))
        w_fi.append(random.randint(0, W_size - 1))
        val_fi.append(random.uniform(min_value, max_value))

    return [conv_fi, batch_fi, c_fi, h_fi, w_fi, val_fi]


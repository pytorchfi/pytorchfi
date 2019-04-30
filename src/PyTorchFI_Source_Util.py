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

from src import PyTorchFI_Source_Core as pytorchfi_core

INPUT_DATA = None
BATCH_SIZE = 0

USE_CUDA = False
DEBUG = False

DATA_SIZE = []


def init(model, data, **kwargs):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utilinit/
    """
    global MODEL, INPUT_DATA, DATA_SIZE, BATCH_SIZE
    MODEL, INPUT_DATA, DATA_SIZE, BATCH_SIZE = model, data[0], data[0].size(), data[1]
    pytorchfi_core.init(model, DATA_SIZE[2], DATA_SIZE[3], BATCH_SIZE, **kwargs)


def declare_weight_fi(index, min_value, max_value):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utildeclareweightfi/
    """
    global MODEL
    MODEL = pytorchfi_core.declare_weight_fi(index, min_value, max_value)


def declare_neuron_fi(**kwargs):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utildeclareneuronfi/
    """
    global MODEL
    MODEL = pytorchfi_core.declare_neuron_fi(**kwargs)


def compare_golden():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utilcomparegolden/
    """
    softmax = nn.Softmax(dim=1)

    model = pytorchfi_core.get_original_model()
    golden_output = model(INPUT_DATA)
    golden_output_softmax = softmax(golden_output)
    golden = list(torch.argmax(golden_output_softmax, dim=1))

    corrupted_output = MODEL(INPUT_DATA)
    corrupted_output_softmax = softmax(corrupted_output)
    corrupted = list(torch.argmax(corrupted_output_softmax, dim=1))

    return [golden, corrupted]


def time_model():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utiltimemodel/
    """
    start_time = time.time()
    MODEL(INPUT_DATA)
    end_time = time.time()
    return end_time - start_time


def random_batch_fi_gen(conv_number, fmap_number, H_size, W_size, min_value, max_value):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/util/utilrandfigen/
    """
    conv_fi = [conv_number] * BATCH_SIZE
    batch_fi = list(range(BATCH_SIZE))
    c_fi = [fmap_number] * BATCH_SIZE
    h_fi = []
    w_fi = []
    val_fi = []

    for i in range(BATCH_SIZE):
        h_fi.append(random.randint(0, H_size - 1))
        w_fi.append(random.randint(0, W_size - 1))
        val_fi.append(random.uniform(min_value, max_value))

    return [conv_fi, batch_fi, c_fi, h_fi, w_fi, val_fi]

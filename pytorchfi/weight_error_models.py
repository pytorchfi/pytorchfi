"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import logging
import torch
from pytorchfi import core
from pytorchfi.util import *

# Helper functions
def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


def random_weight_location(pfi, layer=-1):
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_weights_dim(layer)
    shape = pfi.get_weights_size(layer)

    dim0_shape = shape[0]
    k = random.randint(0, dim0_shape - 1)
    if dim > 1:
        dim1_shape = shape[1]
        dim1_rand = random.randint(0, dim1_shape - 1)
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)
    else:
        dim2_rand = None
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)
    else:
        dim3_rand = None

    return ([layer], [k], [dim1_rand], [dim2_rand], [dim3_rand])


# Weight Perturbation Models
def random_weight_inj(pfi, corrupt_layer=-1, min_val=-1, max_val=1):
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_layer)
    faulty_val = [random_value(min_val=min_val, max_val=max_val)]

    return pfi.declare_weight_fi(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zero_func_rand_weight(pfi):
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fi(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    new_data = data[location] * 0
    return new_data

def FFRA(pfi, sdc_p,function):
    error_index = [[], [], [], [],[]]
    for layer_idx in range(pfi.get_total_layers()):
        shape = pfi.get_weights_size(layer_idx)
        if len(shape) == 2:
            idx = [[x, y, None, None]
                   for x in range(shape[0]) for y in range(1)]
        elif len(shape) == 4:
            idx = [[k, dim1, dim2, dim3] for k in range(
                shape[0]) for dim1 in range(shape[1]) for dim2 in range(shape[2]) for dim3 in range(shape[3])]
        elif len(shape) == 1:
            idx = [[k, None, None, None] for k in range(shape[0])]
        elif len(shape) == 3:
            idx = [[k, dim1, dim2, None] for k in range(shape[0]) for dim1 in range(shape[1]) for dim2 in range[2]]
        for loc in idx:
            p = random.random()
            if p < sdc_p:
                error_index[0].append(layer_idx)
                for sub, v in enumerate(loc):
                    error_index[sub+1].append(v)
    return pfi.declare_weight_fi(
        layer_num=error_index[0], k=error_index[1], dim1=error_index[2], dim2=error_index[3], dim3 = error_index[4],function=function)

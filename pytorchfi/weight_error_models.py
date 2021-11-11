"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import random
import logging
import torch
from pytorchfi import core

# Helper functions


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


def random_weight_location(pfi, layer=-1):
    loc = []
    total_layers = pfi.get_total_layers()

    corrupt_layer = random.randint(0, total_layers - 1) if layer == -1 else layer
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    if curr_layer != total_layers or len(loc) != 5:
        raise AssertionError

    return loc


# Weight Perturbation Models
def random_weight_inj(pfi, corrupt_conv=-1, min_val=-1, max_val=1):
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi.declare_weight_fi(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zero_func_rand_weight(pfi):
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fi(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData

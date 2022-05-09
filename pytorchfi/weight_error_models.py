"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import random

import pytorchfi.core as core
from pytorchfi.util import random_value


def random_weight_location(pfi, layer: int = -1):
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
def random_weight_inj(
    pfi, corrupt_layer: int = -1, min_val: int = -1, max_val: int = 1
):
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_layer)
    faulty_val = [random_value(min_val=min_val, max_val=max_val)]

    return pfi.declare_weight_fault_injection(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zero_func_rand_weight(pfi: core.FaultInjection):
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fault_injection(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    new_data = data[location] * 0
    return new_data


def multi_weight_inj(pfi, sdc_p=1e-5, function=_zero_rand_weight):
    corrupt_idx = [[], [], [], [], []]
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        dim_len = len(shape)
        shape.extend([1 for i in range(4 - len(shape))])
        for k in range(shape[0]):
            for dim1 in range(shape[1]):
                for dim2 in range(shape[2]):
                    for dim3 in range(shape[3]):
                        if random.random() < sdc_p:
                            idx = [layer_idx, k, dim1, dim2, dim3]
                            for i in range(dim_len + 1):
                                corrupt_idx[i].append(idx[i])
                            for i in range(dim_len + 1, 5):
                                corrupt_idx[i].append(None)
    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=function,
    )

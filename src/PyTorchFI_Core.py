"""
Copyright (c) 2019 University of Illinois
All rights reserved.

Developed by:         
                          RSIM Research Group
                          University of Illinois at Urbana-Champaign
                        http://rsim.cs.illinois.edu/
"""

import copy
import random
import torch
import torch.nn as nn

MODEL = None
USE_CUDA = False
DEBUG = False
BATCH_SIZE = 0

RANDOM_INJECTION = False
CUSTOM_INJECTION = False
INJECTION_FUNCTION = None

CORRUPT_BATCH = -1
CORRUPT_CONV = -1
CORRUPT_C = -1
CORRUPT_H = -1
CORRUPT_W = -1
CORRUPT_VALUE = None
MIN_CORRUPT_VALUE = -500
MAX_CORRUPT_VALUE = 500

BCHW = []
OUTPUT_SIZE = []
CURRENT_CONV = 0


def fi_reset():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/corefireset/
    """
    global CURRENT_CONV, RANDOM_INJECTION, CORRUPT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE, MIN_CORRUPT_VALUE, MAX_CORRUPT_VALUE
    CURRENT_CONV, RANDOM_INJECTION, CORRUPT_BATCH, CORRUPT_CONV, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE, MIN_CORRUPT_VALUE, MAX_CORRUPT_VALUE = (
        0,
        False,
        -1,
        -1,
        -1,
        -1,
        -1,
        None,
        -500,
        500,
    )

    if DEBUG:
        print("Fault injector reset")


def init(model, h, w, batch_size, **kwargs):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coreinit/
    """
    b = kwargs.get("b", 1)
    use_cuda = kwargs.get("use_cuda", False)
    c = kwargs.get("c", 3)

    global MODEL
    if use_cuda:
        MODEL = nn.DataParallel(model)
    else:
        MODEL = model

    global BATCH_SIZE
    BATCH_SIZE = batch_size

    for param in MODEL.modules():
        if isinstance(param, nn.Conv2d):
            param.register_forward_hook(save_output_size)

    global BCHW
    BCHW = [b, c, h, w]

    MODEL(torch.randn(b, c, h, w))

    if DEBUG:
        print("Model output size:")
        print(
            "\n".join(
                ["".join(["{:4}".format(item) for item in row]) for row in OUTPUT_SIZE]
            )
        )


def declare_weight_fi(index, min_value, max_value):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coredeclareweightfi/
    """
    fi_reset()
    model = copy.deepcopy(MODEL)
    model.features[index].weight.data.clamp_(min=min_value, max=max_value)
    return model


def declare_neuron_fi(**kwargs):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coredeclareneuronfi/
    """
    fi_reset()

    if kwargs:
        if "function" in kwargs:
            global CUSTOM_INJECTION, INJECTION_FUNCTION
            CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
        else:
            global CORRUPT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE, MIN_CORRUPT_VALUE, MAX_CORRUPT_VALUE
            CORRUPT_CONV = kwargs.get("conv_num", -1)
            CORRUPT_BATCH = kwargs.get("batch", -1)
            CORRUPT_C = kwargs.get("c", -1)
            CORRUPT_H = kwargs.get("h", -1)
            CORRUPT_W = kwargs.get("w", -1)
            CORRUPT_VALUE = kwargs.get("value", None)
            MIN_CORRUPT_VALUE = kwargs.get("min_value", -500)
            MAX_CORRUPT_VALUE = kwargs.get("max_value", 500)

            if DEBUG:
                print("Declaring Specified Fault Injector")
                print("Convolution: %s" % CORRUPT_CONV)
                print("Batch, x, y, z:")
                print(
                    "%s, %s, %s, %s" % (CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W)
                )

    else:
        global RANDOM_INJECTION
        RANDOM_INJECTION = True

        if DEBUG:
            print("Declaring Randomized Fault Injector")

    model = copy.deepcopy(MODEL)

    for param in model.modules():
        if isinstance(param, nn.Conv2d):
            if RANDOM_INJECTION:
                param.register_forward_hook(random_value)
            elif CUSTOM_INJECTION:
                param.register_forward_hook(INJECTION_FUNCTION)
            else:
                param.register_forward_hook(set_value)

    return model


def validate_fi(**kwargs):
    if type(CORRUPT_CONV) == list:
        index = kwargs.get("index", -1)
        return not (
            CORRUPT_CONV[index] < 0
            or CORRUPT_CONV[index] >= len(OUTPUT_SIZE)
            or CORRUPT_BATCH[index] >= BATCH_SIZE
            or CORRUPT_C[index] > OUTPUT_SIZE[CORRUPT_CONV[index]][1]
            or CORRUPT_H[index] > OUTPUT_SIZE[CORRUPT_CONV[index]][2]
            or CORRUPT_W[index] > OUTPUT_SIZE[CORRUPT_CONV[index]][3]
            or CORRUPT_CONV == -1
            or CORRUPT_BATCH == -1
            or CORRUPT_C == -1
            or CORRUPT_H == -1
            or CORRUPT_W == -1
        )
    else:
        return not (
            CORRUPT_CONV < 0
            or CORRUPT_CONV >= len(OUTPUT_SIZE)
            or CORRUPT_BATCH >= BATCH_SIZE
            or CORRUPT_C > OUTPUT_SIZE[CORRUPT_CONV][1]
            or CORRUPT_H > OUTPUT_SIZE[CORRUPT_CONV][2]
            or CORRUPT_W > OUTPUT_SIZE[CORRUPT_CONV][3]
            or CORRUPT_CONV == -1
            or CORRUPT_BATCH == -1
            or CORRUPT_C == -1
            or CORRUPT_H == -1
            or CORRUPT_W == -1
        )


def random_value(self, input, output):
    global CURRENT_CONV
    random_batch = random.randint(0, OUTPUT_SIZE[CURRENT_CONV][0] - 1)
    random_x = random.randint(0, OUTPUT_SIZE[CURRENT_CONV][1] - 1)
    random_y = random.randint(0, OUTPUT_SIZE[CURRENT_CONV][2] - 1)
    random_z = random.randint(0, OUTPUT_SIZE[CURRENT_CONV][3] - 1)
    if CORRUPT_VALUE is not None:
        value = CORRUPT_VALUE
    else:
        value = random.uniform(MIN_CORRUPT_VALUE, MAX_CORRUPT_VALUE)

    if DEBUG:
        print(
            "Original value at [%d][%d][%d][%d]: %d"
            % (
                random_batch,
                random_x,
                random_y,
                random_z,
                output[random_batch][random_x][random_y][random_z],
            )
        )
        print("Changing value to %d" % value)

    output[random_batch][random_x][random_y][random_z] = value
    CURRENT_CONV = CURRENT_CONV + 1


def set_value(self, input, output):
    global CURRENT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE, CORRUPT_CONV

    if type(CORRUPT_CONV) == list:
        try:
            i = CORRUPT_CONV.index(CURRENT_CONV)
            if validate_fi(index=i):
                if DEBUG:
                    print(
                        "Original value at [%d][%d][%d][%d]: %d"
                        % (
                            CORRUPT_BATCH[i],
                            CORRUPT_C[i],
                            CORRUPT_H[i],
                            CORRUPT_W[i],
                            output[CORRUPT_BATCH[i]][CORRUPT_C[i]][CORRUPT_H[i]][
                                CORRUPT_W[i]
                            ],
                        )
                    )
                    print("Changing value to %d" % CORRUPT_VALUE[i])
                output[CORRUPT_BATCH[i]][CORRUPT_C[i]][CORRUPT_H[i]][
                    CORRUPT_W[i]
                ] = CORRUPT_VALUE[i]
                del (
                    CORRUPT_BATCH[i],
                    CORRUPT_C[i],
                    CORRUPT_CONV[i],
                    CORRUPT_H[i],
                    CORRUPT_W[i],
                    CORRUPT_VALUE[i],
                )
            else:
                print("Fault injection not valid!")
        except ValueError:
            pass

        CURRENT_CONV = CURRENT_CONV + 1
    else:
        if validate_fi():
            if CURRENT_CONV == CORRUPT_CONV:
                if DEBUG:
                    print(
                        "Original value at [%d][%d][%d][%d]: %d"
                        % (
                            CORRUPT_BATCH,
                            CORRUPT_C,
                            CORRUPT_H,
                            CORRUPT_W,
                            output[CORRUPT_BATCH][CORRUPT_C][CORRUPT_H][CORRUPT_W],
                        )
                    )
                    print("Changing value to %d" % CORRUPT_VALUE)
                output[CORRUPT_BATCH][CORRUPT_C][CORRUPT_H][CORRUPT_W] = CORRUPT_VALUE
            CURRENT_CONV = CURRENT_CONV + 1
        else:
            print("Fault injection not valid!")


def save_output_size(self, input, output):
    global OUTPUT_SIZE
    OUTPUT_SIZE.append(list(output.size()))


def get_original_model():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coregetoriginalmodel/
    """
    return MODEL


def get_output_size():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coregetoutputsize/
    """
    return OUTPUT_SIZE


def get_bchw():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coregetbchw/
    """
    return BCHW


def get_b():
    return BCHW[0]


def get_c():
    return BCHW[1]


def get_h():
    return BCHW[2]


def get_w():
    return BCHW[3]

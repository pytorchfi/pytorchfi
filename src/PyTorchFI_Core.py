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

ORIG_MODEL = None
DEBUG = False
BATCH_SIZE = -1

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
CURRENT_CONV = -1
HANDLES = []


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


    global HANDLES

    for i in range(len(HANDLES)):
        HANDLES[i].remove()

    if DEBUG:
        print("Fault injector reset")


def init(model, h, w, batch_size, **kwargs):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coreinit/
    """

    fi_reset()
    global OUTPUT_SIZE
    OUTPUT_SIZE = []

    b = kwargs.get("b", 1)
    c = kwargs.get("c", 3)
    use_cuda = kwargs.get("use_cuda", False)

    global ORIG_MODEL
    if use_cuda:
        ORIG_MODEL = nn.DataParallel(model)
    else:
        ORIG_MODEL = model

    global BATCH_SIZE
    BATCH_SIZE = batch_size

    handles = []
    for param in ORIG_MODEL.modules():
        if isinstance(param, nn.Conv2d):
            handles.append(param.register_forward_hook(save_output_size))

    global BCHW
    BCHW = [b, c, h, w]

    ORIG_MODEL(torch.randn(b, c, h, w))

    for i in range(len(handles)):
        handles[i].remove()

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
    pfi_model = copy.deepcopy(ORIG_MODEL)
    pfi_model.features[index].weight.data.clamp_(min=min_value, max=max_value)
    return pfi_model


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

    pfi_model = copy.deepcopy(ORIG_MODEL)

    global HANDLES
    for param in pfi_model.modules():
        if isinstance(param, nn.Conv2d):
            if CUSTOM_INJECTION:
                HANDLES.append(param.register_forward_hook(INJECTION_FUNCTION))
            else:
                HANDLES.append(param.register_forward_hook(set_value))

    return pfi_model


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

# generates a random injection (default value range [-1, 1]) in every layer of each batch element
def random_inj_per_layer(min_val=-1, max_val=1):
    conv_num = []
    batch = []
    c_rand = []
    w_rand = []
    h_rand = []
    value = []
    for i in range(get_total_batches()):
        for j in range(get_total_conv()):
            conv_num.append(j)
            batch.append(i)
            c_rand.append(random.randint(0, get_fmaps_num(j) - 1))
            h_rand.append(random.randint(0, get_fmaps_H(j) - 1))
            w_rand.append(random.randint(0, get_fmaps_W(j) - 1))
            value.append(random.randint(min_val, max_val))
    return declare_neuron_fi(conv_num=conv_num, batch=batch, c=c_rand, h=h_rand, w=w_rand, value=value)

# generates a single random injection (default value range [-1, 1]) in each batch element
def random_inj(min_val=-1, max_val=1):
    conv_num = []
    batch = []
    c_rand = []
    h_rand = []
    w_rand = []
    value = []
    for i in range(get_total_batches()):
        conv_num.append(random.randint(0, get_total_conv() - 1))
        batch.append(i)
        c_rand.append(random.randint(0, get_fmaps_num(conv_num[i]) - 1))
        h_rand.append(random.randint(0, get_fmaps_H(conv_num[i]) - 1))
        w_rand.append(random.randint(0, get_fmaps_W(conv_num[i]) - 1))
        value.append(random.randint(min_val, max_val))
    return declare_neuron_fi(conv_num=conv_num, batch=batch, c=c_rand, h=h_rand, w=w_rand, value=value)


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
                            output[CORRUPT_BATCH[i]][CORRUPT_C[i]][CORRUPT_H[i]][CORRUPT_W[i]],
                        )
                    )
                    print("Changing value to %d" % CORRUPT_VALUE[i])
                output[CORRUPT_BATCH[i]][CORRUPT_C[i]][CORRUPT_H[i]][CORRUPT_W[i]] = CORRUPT_VALUE[i]
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
    return ORIG_MODEL


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

# returns total batches
def get_total_batches():
    return BATCH_SIZE

# returns total number of convs
def get_total_conv():
    return len(OUTPUT_SIZE)

# returns total number of fmaps in a layer
def get_fmaps_num(layer):
    return OUTPUT_SIZE[layer][1]

# returns fmap H size
def get_fmaps_H(layer):
    return OUTPUT_SIZE[layer][2]

# returns fmap W size
def get_fmaps_W(layer):
    return OUTPUT_SIZE[layer][3]

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
CORRUPTED_MODEL = None
DEBUG = False
_BATCH_SIZE = -1

CUSTOM_INJECTION = False
INJECTION_FUNCTION = None

CORRUPT_BATCH = -1
CORRUPT_CONV = -1
CORRUPT_C = -1
CORRUPT_H = -1
CORRUPT_W = -1
CORRUPT_VALUE = None

OUTPUT_SIZE = []
CURRENT_CONV = 0
HANDLES = []


def fi_reset():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/corefireset/
    """
    global CURRENT_CONV, CORRUPT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE
    CURRENT_CONV, CORRUPT_BATCH, CORRUPT_CONV, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE = (
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
        None,
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

    global _BATCH_SIZE
    _BATCH_SIZE = batch_size

    handles = []
    for param in ORIG_MODEL.modules():
        if isinstance(param, nn.Conv2d):
            handles.append(param.register_forward_hook(_save_output_size))

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


def declare_weight_fi(index, min_value=-1, max_value=1):
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coredeclareweightfi/
    """
    #TODO check injection bounds
    fi_reset()
    global CORRUPTED_MODEL
    CORRUPTED_MODEL = copy.deepcopy(ORIG_MODEL)
    CORRUPTED_MODEL.features[index].weight.data.clamp_(min=min_value, max=max_value)
    return CORRUPTED_MODEL


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
            global CORRUPT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE
            CORRUPT_CONV = kwargs.get("conv_num", -1)
            CORRUPT_BATCH = kwargs.get("batch", -1)
            CORRUPT_C = kwargs.get("c", -1)
            CORRUPT_H = kwargs.get("h", -1)
            CORRUPT_W = kwargs.get("w", -1)
            CORRUPT_VALUE = kwargs.get("value", None)

            if DEBUG:
                print("Declaring Specified Fault Injector")
                print("Convolution: %s" % CORRUPT_CONV)
                print("Batch, x, y, z:")
                print(
                    "%s, %s, %s, %s" % (CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W)
                )
    else:
        raise ValueError("Please specify an injection or injection function")

    # make a deep copy of the input model to corrupt
    global CORRUPTED_MODEL
    CORRUPTED_MODEL = copy.deepcopy(ORIG_MODEL)

    # attach hook with injection functions to each conv module
    global HANDLES
    for param in CORRUPTED_MODEL.modules():
        if isinstance(param, nn.Conv2d):
            if CUSTOM_INJECTION:
                HANDLES.append(param.register_forward_hook(INJECTION_FUNCTION))
            else:
                HANDLES.append(param.register_forward_hook(_set_value))

    return CORRUPTED_MODEL


def assert_inj_bounds(**kwargs):
    # checks for specfic injection out of a list
    if type(CORRUPT_CONV) == list:
        index = kwargs.get("index", -1)
        assert CORRUPT_CONV[index] >= 0 and CORRUPT_CONV[index] < get_total_conv(), "invalid conv"
        assert CORRUPT_BATCH[index] >= 0 and CORRUPT_BATCH[index] < _BATCH_SIZE, "invalid batch"
        assert CORRUPT_C[index] >= 0 and CORRUPT_C[index] < \
            OUTPUT_SIZE[CORRUPT_CONV[index]][1], "invalid c"
        assert CORRUPT_H[index] >= 0 and CORRUPT_H[index] < \
            OUTPUT_SIZE[CORRUPT_CONV[index]][2], "invalid h"
        assert CORRUPT_W[index] >= 0 and CORRUPT_W[index] < \
            OUTPUT_SIZE[CORRUPT_CONV[index]][3], "invalid w"
    # checks for single injection
    else:
        assert CORRUPT_CONV >= 0 and CORRUPT_CONV < get_total_conv(), "invalid conv"
        assert CORRUPT_BATCH >= 0 and CORRUPT_BATCH < _BATCH_SIZE, "invalid batch"
        assert CORRUPT_C >= 0 and CORRUPT_C < OUTPUT_SIZE[CORRUPT_CONV][1], "invalid c"
        assert CORRUPT_H >= 0 and CORRUPT_H < OUTPUT_SIZE[CORRUPT_CONV][2], "invalid h"
        assert CORRUPT_W >= 0 and CORRUPT_W < OUTPUT_SIZE[CORRUPT_CONV][3], "invalid w"


def _set_value(self, input, output):
    global CURRENT_CONV, CORRUPT_BATCH, CORRUPT_C, CORRUPT_H, CORRUPT_W, CORRUPT_VALUE, CORRUPT_CONV

    if type(CORRUPT_CONV) == list:
        # extract injections in this layer
        inj_list = list(filter(lambda x: CORRUPT_CONV[x] == CURRENT_CONV, range(len(CORRUPT_CONV))))
        # perform each injection for this layer
        for i in inj_list:
            # check that the injection indices are valid
            assert_inj_bounds(index=i)
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
            # inject value
            output[CORRUPT_BATCH[i]][CORRUPT_C[i]][CORRUPT_H[i]][CORRUPT_W[i]] = CORRUPT_VALUE[i]
        # useful for injection hooks
        CURRENT_CONV += 1

    else: # single injection (not a list of injections)
        # check that the injection indices are valid
        assert_inj_bounds()
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
            # inject value
            output[CORRUPT_BATCH][CORRUPT_C][CORRUPT_H][CORRUPT_W] = CORRUPT_VALUE
        CURRENT_CONV += 1


def _save_output_size(self, input, output):
    global OUTPUT_SIZE
    OUTPUT_SIZE.append(list(output.size()))

def get_original_model():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coregetoriginalmodel/
    """
    return ORIG_MODEL

def get_corrupted_model():
    return CORRUPTED_MODEL

def get_output_size():
    """
    https://n3a9.github.io/pytorchfi-docs-beta/docs/functionlist/core/coregetoutputsize/
    """
    return OUTPUT_SIZE

# returns total batches
def get_total_batches():
    return _BATCH_SIZE

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

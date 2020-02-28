"""
pytorchfi.core contains the core functionality for fault injections.
"""

import copy
import random
import torch
import torch.nn as nn


class core:
    def __init__(self, model, h, w, batch_size, **kwargs):
        self.ORIG_MODEL = None
        self.CORRUPTED_MODEL = None
        self.DEBUG = False
        self._BATCH_SIZE = -1

        self.CUSTOM_INJECTION = False
        self.INJECTION_FUNCTION = None

        self.CORRUPT_BATCH = -1
        self.CORRUPT_CONV = -1
        self.CORRUPT_C = -1
        self.CORRUPT_H = -1
        self.CORRUPT_W = -1
        self.CORRUPT_VALUE = None

        self.OUTPUT_SIZE = []
        self.CURRENT_CONV = 0
        self.HANDLES = []

        b = kwargs.get("b", 1)
        c = kwargs.get("c", 3)
        use_cuda = kwargs.get("use_cuda", False)

        self.ORIG_MODEL = nn.DataParallel(model) if use_cuda else model

        self._BATCH_SIZE = batch_size

        handles = []
        for param in self.ORIG_MODEL.modules():
            if isinstance(param, nn.Conv2d):
                handles.append(param.register_forward_hook(self._save_output_size))

        self.ORIG_MODEL(torch.randn(b, c, h, w))

        for i in range(len(handles)):
            handles[i].remove()

        if self.DEBUG:
            print("Model output size:")
            print(
                "\n".join(
                    [
                        "".join(["{:4}".format(item) for item in row])
                        for row in self.OUTPUT_SIZE
                    ]
                )
            )

    def fi_reset(self):
        (
            self.CURRENT_CONV,
            self.CORRUPT_BATCH,
            self.CORRUPT_CONV,
            self.CORRUPT_C,
            self.CORRUPT_H,
            self.CORRUPT_W,
            self.CORRUPT_VALUE,
        ) = (
            0,
            -1,
            -1,
            -1,
            -1,
            -1,
            None,
        )

        for i in range(len(self.HANDLES)):
            self.HANDLES[i].remove()

        if self.DEBUG:
            print("Fault injector reset")

    def declare_weight_fi(self, **kwargs):
        self.fi_reset()
        custom_function = False
        zero_layer = False

        if kwargs:
            if "function" in kwargs:
                # custom function specified injection
                custom_function, function = True, kwargs.get("function")
                corrupt_idx = kwargs.get("index", 0)
            elif kwargs.get("zero", False):
                # zero layer
                zero_layer = True
            elif kwargs.get("rand", False):
                rand_inj = True
                min_val = kwargs.get("min_rand_val")
                max_val = kwargs.get("max_rand_val")
            else:
                # specified injection
                corrupt_value = kwargs.get("value", -1)
                corrupt_idx = kwargs.get("index", -1)
            corrupt_layer = kwargs.get("layer", -1)
        else:
            raise ValueError("Please specify an injection or injection function")

        # make deep copy of input model to corrupt
        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)

        # get number of weight files
        num_layers = 0
        for name, param in self.CORRUPTED_MODEL.named_parameters():
            if name.split(".")[-1] == "weight":
                num_layers += 1
        curr_layer = 0
        orig_value = 0

        if rand_inj:
            corrupt_layer = random.randint(0, num_layers - 1)
        for name, param in self.CORRUPTED_MODEL.named_parameters():
            if name.split(".")[-1] == "weight":
                if curr_layer == corrupt_layer:
                    if zero_layer:
                        param.data[:] = 0
                        if self.DEBUG:
                            print("Zero weight layer")
                            print("Layer index: %s" % corrupt_layer)
                    else:
                        if rand_inj:
                            corrupt_value = random.uniform(min_val, max_val)
                            corrupt_idx = list()
                            for dim in param.size():
                                corrupt_idx.append(random.randint(0, dim - 1))
                        orig_value = param.data[tuple(corrupt_idx)].item()
                        # Use function if specified
                        if custom_function:
                            corrupt_value = function(param.data[tuple(corrupt_idx)])
                        # Inject corrupt value
                        param.data[tuple(corrupt_idx)] = corrupt_value
                        if self.DEBUG:
                            print("Weight Injection")
                            print("Layer index: %s" % corrupt_layer)
                            print("Module: %s" % name)
                            print("Original value: %s" % orig_value)
                            print("Injected value: %s" % corrupt_value)
                curr_layer += 1
        return self.CORRUPTED_MODEL

    def declare_neuron_fi(self, **kwargs):
        self.fi_reset()

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
            else:
                self.CORRUPT_CONV = kwargs.get("conv_num", -1)
                self.CORRUPT_BATCH = kwargs.get("batch", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
                self.CORRUPT_VALUE = kwargs.get("value", None)

                if self.DEBUG:
                    print("Declaring Specified Fault Injector")
                    print("Convolution: %s" % self.CORRUPT_CONV)
                    print("Batch, x, y, z:")
                    print(
                        "%s, %s, %s, %s"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                        )
                    )
        else:
            raise ValueError("Please specify an injection or injection function")

        # make a deep copy of the input model to corrupt
        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)

        # attach hook with injection functions to each conv module
        for param in self.CORRUPTED_MODEL.modules():
            if isinstance(param, nn.Conv2d):
                if CUSTOM_INJECTION:
                    self.HANDLES.append(param.register_forward_hook(INJECTION_FUNCTION))
                else:
                    self.HANDLES.append(param.register_forward_hook(self._set_value))

        return self.CORRUPTED_MODEL

    def assert_inj_bounds(self, **kwargs):
        # checks for specfic injection out of a list
        if type(self.CORRUPT_CONV) == list:
            index = kwargs.get("index", -1)
            assert (
                self.CORRUPT_CONV[index] >= 0
                and self.CORRUPT_CONV[index] < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH[index] >= 0
                and self.CORRUPT_BATCH[index] < self._BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C[index] >= 0
                and self.CORRUPT_C[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][1]
            ), "Invalid C!"
            assert (
                self.CORRUPT_H[index] >= 0
                and self.CORRUPT_H[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W[index] >= 0
                and self.CORRUPT_W[index]
                < self.OUTPUT_SIZE[self.CORRUPT_CONV[index]][3]
            ), "Invalid W!"
        # checks for single injection
        else:
            assert (
                self.CORRUPT_CONV >= 0 and self.CORRUPT_CONV < self.get_total_conv()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH >= 0 and self.CORRUPT_BATCH < self._BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C >= 0
                and self.CORRUPT_C < self.OUTPUT_SIZE[self.CORRUPT_CONV][1]
            ), "Invalid C!"
            assert (
                self.CORRUPT_H >= 0
                and self.CORRUPT_H < self.OUTPUT_SIZE[self.CORRUPT_CONV][2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W >= 0
                and self.CORRUPT_W < self.OUTPUT_SIZE[self.CORRUPT_CONV][3]
            ), "Invalid W!"

    def _set_value(self, input, output):
        if type(self.CORRUPT_CONV) == list:
            # extract injections in this layer
            inj_list = list(
                filter(
                    lambda x: self.CORRUPT_CONV[x] == CURRENT_CONV,
                    range(len(self.CORRUPT_CONV)),
                )
            )
            # perform each injection for this layer
            for i in inj_list:
                # check that the injection indices are valid
                self.assert_inj_bounds(index=i)
                if self.DEBUG:
                    print(
                        "Original value at [%d][%d][%d][%d]: %d"
                        % (
                            self.CORRUPT_BATCH[i],
                            self.CORRUPT_C[i],
                            self.CORRUPT_H[i],
                            self.CORRUPT_W[i],
                            output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                                self.CORRUPT_H[i]
                            ][self.CORRUPT_W[i]],
                        )
                    )
                    print("Changing value to %d" % self.CORRUPT_VALUE[i])
                # inject value
                output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                    self.CORRUPT_W[i]
                ] = self.CORRUPT_VALUE[i]
            # useful for injection hooks
            CURRENT_CONV += 1

        else:  # single injection (not a list of injections)
            # check that the injection indices are valid
            self.assert_inj_bounds()
            if CURRENT_CONV == self.CORRUPT_CONV:
                if self.DEBUG:
                    print(
                        "Original value at [%d][%d][%d][%d]: %d"
                        % (
                            self.CORRUPT_BATCH,
                            self.CORRUPT_C,
                            self.CORRUPT_H,
                            self.CORRUPT_W,
                            output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                                self.CORRUPT_W
                            ],
                        )
                    )
                    print("Changing value to %d" % self.CORRUPT_VALUE)
                # inject value
                output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ] = self.CORRUPT_VALUE
            CURRENT_CONV += 1

    def _save_output_size(self, input, output):
        self.OUTPUT_SIZE.append(list(output.size()))

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL

    def get_output_size(self):
        return self.OUTPUT_SIZE

    # returns total batches
    def get_total_batches(self):
        return self._BATCH_SIZE

    # returns total number of convs
    def get_total_conv(self):
        return len(self.OUTPUT_SIZE)

    # returns total number of fmaps in a layer
    def get_fmaps_num(self, layer):
        return self.OUTPUT_SIZE[layer][1]

    # returns fmap H size
    def get_fmaps_H(self, layer):
        return self.OUTPUT_SIZE[layer][2]

    # returns fmap W size
    def get_fmaps_W(self, layer):
        return self.OUTPUT_SIZE[layer][3]

    def get_fmap_HW(self, layer):
        return tuple(self.OUTPUT_SIZE[layer[2:4]])

    def set_debug(self, debug):
        self.DEBUG = debug

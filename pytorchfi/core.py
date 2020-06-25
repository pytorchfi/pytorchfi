"""
pytorchfi.core contains the core functionality for fault injections.
"""

import copy
import random

import logger
import torch
import torch.nn as nn


class fault_injection:
    def __init__(self, model, h, w, batch_size, **kwargs):
        logger.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        self.ORIG_MODEL = None
        self.CORRUPTED_MODEL = None
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
        use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)
        model_dtype = next(model.parameters()).dtype

        self.ORIG_MODEL = model
        self._BATCH_SIZE = batch_size

        handles = []
        for param in self.ORIG_MODEL.modules():
            if isinstance(param, nn.Conv2d):
                handles.append(param.register_forward_hook(self._save_output_size))

        device = "cuda" if use_cuda else None
        _dummyTensor = torch.randn(b, c, h, w, dtype=model_dtype, device=device)

        self.ORIG_MODEL(_dummyTensor)

        for i in range(len(handles)):
            handles[i].remove()

        logger.info("Model output size")
        logger.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.OUTPUT_SIZE
                ]
            )
        )

    def fi_reset(self):
        self._fi_state_reset()
        self.CORRUPTED_MODEL = None
        logger.info("Reset fault injector")

    def _fi_state_reset(self):
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

    def declare_weight_fi(self, **kwargs):
        self._fi_state_reset()
        custom_function = False
        zero_layer = False
        rand_inj = False

        if kwargs:
            if "function" in kwargs:
                custom_function, function = True, kwargs.get("function")
                corrupt_idx = kwargs.get("index", 0)
            elif kwargs.get("zero", False):
                zero_layer = True
            elif kwargs.get("rand", False):
                rand_inj = True
                min_val = kwargs.get("min_rand_val")
                max_val = kwargs.get("max_rand_val")
            else:
                corrupt_value = kwargs.get("value", -1)
                corrupt_idx = kwargs.get("index", -1)
            corrupt_layer = kwargs.get("layer", -1)
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)

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
                        logger.info("Zero weight layer")
                        logger.info("Layer index: %s" % corrupt_layer)
                    else:
                        if rand_inj:
                            corrupt_value = random.uniform(min_val, max_val)
                            corrupt_idx = list()
                            for dim in param.size():
                                corrupt_idx.append(random.randint(0, dim - 1))
                        corrupt_idx = (
                            tuple(corrupt_idx)
                            if isinstance(corrupt_idx, list)
                            else corrupt_idx
                        )
                        orig_value = param.data[corrupt_idx].item()
                        if custom_function:
                            corrupt_value = function(param.data[tuple(corrupt_idx)])

                        param.data[corrupt_idx] = corrupt_value
                        logger.info("Weight Injection")
                        logger.info("Layer index: %s" % corrupt_layer)
                        logger.info("Module: %s" % name)
                        logger.info("Original value: %s" % orig_value)
                        logger.info("Injected value: %s" % corrupt_value)

                curr_layer += 1
        return self.CORRUPTED_MODEL

    def declare_neuron_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        INJECTION_FUNCTION = False

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

                logger.info("Declaring Specified Fault Injector")
                logger.info("Convolution: %s" % self.CORRUPT_CONV)
                logger.info("Batch, x, y, z:")
                logger.info(
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

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)

        for param in self.CORRUPTED_MODEL.modules():
            if isinstance(param, nn.Conv2d):
                hook = INJECTION_FUNCTION if CUSTOM_INJECTION else self._set_value
                self.HANDLES.append(param.register_forward_hook(hook))

        return self.CORRUPTED_MODEL

    def assert_inj_bounds(self, **kwargs):
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

    def _set_value(self, model, input, output):
        if type(self.CORRUPT_CONV) == list:
            inj_list = list(
                filter(
                    lambda x: self.CORRUPT_CONV[x] == self.CURRENT_CONV,
                    range(len(self.CORRUPT_CONV)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logger.info(
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
                logger.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                    self.CORRUPT_W[i]
                ] = self.CORRUPT_VALUE[i]
            self.CURRENT_CONV += 1

        else:
            self.assert_inj_bounds()
            if self.CURRENT_CONV == self.CORRUPT_CONV:
                logger.info(
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
                logger.info("Changing value to %d" % self.CORRUPT_VALUE)
                output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ] = self.CORRUPT_VALUE
            self.CURRENT_CONV += 1

    def _save_output_size(self, module, input, output):
        self.OUTPUT_SIZE.append(list(output.size()))

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def get_total_batches(self):
        return self._BATCH_SIZE

    def get_total_conv(self):
        return len(self.OUTPUT_SIZE)

    def get_fmaps_num(self, layer):
        return self.OUTPUT_SIZE[layer][1]

    def get_fmaps_H(self, layer):
        return self.OUTPUT_SIZE[layer][2]

    def get_fmaps_W(self, layer):
        return self.OUTPUT_SIZE[layer][3]

    def get_fmap_HW(self, layer):
        return (self.get_fmaps_H(layer), self.get_fmaps_W(layer))

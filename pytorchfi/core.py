"""
pytorchfi.core contains the core functionality for fault injections.
"""

import copy
import logging

import torch
import torch.nn as nn


class fault_injection:
    def __init__(self, model, h, w, batch_size, layer_types=[nn.Conv2d], **kwargs):
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        self.ORIG_MODEL = None
        self.CORRUPTED_MODEL = None
        self._BATCH_SIZE = -1

        self.CUSTOM_INJECTION = False
        self.INJECTION_FUNCTION = None

        self.CORRUPT_BATCH = -1
        self.CORRUPT_LAYER = -1
        self.CORRUPT_C = -1
        self.CORRUPT_H = -1
        self.CORRUPT_W = -1
        self.CORRUPT_VALUE = None

        self.CURR_LAYER = 0
        self.OUTPUT_SIZE = []
        self.LAYER_TYPE = []
        self.LAYER_DIM = []
        self.HANDLES = []

        self.imageC = kwargs.get("c", 3)
        self.imageH = h
        self.imageW = w

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)
        model_dtype = next(model.parameters()).dtype

        self.ORIG_MODEL = model
        self._BATCH_SIZE = batch_size
        self._INJ_LAYER_TYPES = layer_types

        handles, shapes = self._traverseModelAndSetHooks(
            self.ORIG_MODEL, self._INJ_LAYER_TYPES
        )

        b = 1  # profiling only needs one batch element

        device = "cuda" if self.use_cuda else None
        _dummyTensor = torch.randn(
            b, self.imageC, self.imageH, self.imageW, dtype=model_dtype, device=device
        )

        self.ORIG_MODEL(_dummyTensor)

        for i in range(len(handles)):
            handles[i].remove()

        logging.info("Model output size")
        logging.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.OUTPUT_SIZE
                ]
            )
        )

    def _traverseModelAndSetHooks(self, model, layer_types):
        handles = []
        shape = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                for i in layer_types:
                    if isinstance(layer, i):
                        handles.append(
                            layer.register_forward_hook(self._save_output_size)
                        )
                        shape.append(layer)
            # unpack node
            else:
                subHandles, subBase = self._traverseModelAndSetHooks(layer, layer_types)
                for i in subHandles:
                    handles.append(i)
                for i in subBase:
                    shape.append(i)

        return (handles, shape)

    def fi_reset(self):
        self._fi_state_reset()
        self.CORRUPTED_MODEL = None
        logging.info("Reset fault injector")

    def _fi_state_reset(self):
        (
            self.CURR_LAYER,
            self.CORRUPT_BATCH,
            self.CORRUPT_LAYER,
            self.CORRUPT_C,
            self.CORRUPT_H,
            self.CORRUPT_W,
            self.CORRUPT_VALUE,
            self._INJ_LAYER_TYPES,
        ) = (0, -1, -1, -1, -1, -1, None, [nn.Conv2d])

        for i in range(len(self.HANDLES)):
            self.HANDLES[i].remove()

    def declare_weight_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        CUSTOM_FUNCTION = False

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, CUSTOM_FUNCTION = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)
            else:
                corrupt_layer = kwargs.get("layer_num", -1)
                corrupt_k = kwargs.get("k", -1)
                corrupt_c = kwargs.get("c", -1)
                corrupt_kH = kwargs.get("h", -1)
                corrupt_kW = kwargs.get("w", -1)
                corrupt_value = kwargs.get("value", -1)
        else:
            raise ValueError("Please specify an injection or injection function")

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        corrupt_idx = [corrupt_k, corrupt_c, corrupt_kH, corrupt_kW]

        curr_layer = 0
        for name, param in self.CORRUPTED_MODEL.named_parameters():
            if "weight" in name and "features" in name:
                if curr_layer == corrupt_layer:
                    corrupt_idx = (
                        tuple(corrupt_idx)
                        if isinstance(corrupt_idx, list)
                        else corrupt_idx
                    )
                    orig_value = param.data[corrupt_idx].item()
                    if CUSTOM_INJECTION:
                        corrupt_value = CUSTOM_FUNCTION(param.data, corrupt_idx)
                    param.data[corrupt_idx] = corrupt_value

                    logging.info("Weight Injection")
                    logging.info("Layer index: %s" % corrupt_layer)
                    logging.info("Module: %s" % name)
                    logging.info("Original value: %s" % orig_value)
                    logging.info("Injected value: %s" % corrupt_value)

                curr_layer += 1
        return self.CORRUPTED_MODEL

    def declare_neuron_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        INJECTION_FUNCTION = False

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, INJECTION_FUNCTION = True, kwargs.get("function")
                self.CORRUPT_LAYER = kwargs.get("layer_num", -1)
                self.CORRUPT_BATCH = kwargs.get("batch", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
            else:
                self.CORRUPT_LAYER = kwargs.get("layer_num", -1)
                self.CORRUPT_BATCH = kwargs.get("batch", -1)
                self.CORRUPT_C = kwargs.get("c", -1)
                self.CORRUPT_H = kwargs.get("h", -1)
                self.CORRUPT_W = kwargs.get("w", -1)
                self.CORRUPT_VALUE = kwargs.get("value", None)

                logging.info("Declaring Specified Fault Injector")
                logging.info("Convolution: %s" % self.CORRUPT_LAYER)
                logging.info("Batch, x, y, z:")
                logging.info(
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
        if type(self.CORRUPT_LAYER) == list:
            index = kwargs.get("index", -1)
            assert (
                self.CORRUPT_LAYER[index] >= 0
                and self.CORRUPT_LAYER[index] < self.get_total_layers()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH[index] >= 0
                and self.CORRUPT_BATCH[index] < self._BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C[index] >= 0
                and self.CORRUPT_C[index]
                < self.OUTPUT_SIZE[self.CORRUPT_LAYER[index]][1]
            ), "Invalid C!"
            assert (
                self.CORRUPT_H[index] >= 0
                and self.CORRUPT_H[index]
                < self.OUTPUT_SIZE[self.CORRUPT_LAYER[index]][2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W[index] >= 0
                and self.CORRUPT_W[index]
                < self.OUTPUT_SIZE[self.CORRUPT_LAYER[index]][3]
            ), "Invalid W!"
        else:
            assert (
                self.CORRUPT_LAYER >= 0 and self.CORRUPT_LAYER < self.get_total_layers()
            ), "Invalid convolution!"
            assert (
                self.CORRUPT_BATCH >= 0 and self.CORRUPT_BATCH < self._BATCH_SIZE
            ), "Invalid batch!"
            assert (
                self.CORRUPT_C >= 0
                and self.CORRUPT_C < self.OUTPUT_SIZE[self.CORRUPT_LAYER][1]
            ), "Invalid C!"
            assert (
                self.CORRUPT_H >= 0
                and self.CORRUPT_H < self.OUTPUT_SIZE[self.CORRUPT_LAYER][2]
            ), "Invalid H!"
            assert (
                self.CORRUPT_W >= 0
                and self.CORRUPT_W < self.OUTPUT_SIZE[self.CORRUPT_LAYER][3]
            ), "Invalid W!"

    def _set_value(self, module, input, output):
        if type(self.CORRUPT_LAYER) == list:
            inj_list = list(
                filter(
                    lambda x: self.CORRUPT_LAYER[x] == self.get_curr_layer(),
                    range(len(self.CORRUPT_LAYER)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
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
                logging.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                    self.CORRUPT_W[i]
                ] = self.CORRUPT_VALUE[i]

        else:
            self.assert_inj_bounds()
            if self.get_curr_layer() == self.CORRUPT_LAYER:
                logging.info(
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
                logging.info("Changing value to %d" % self.CORRUPT_VALUE)
                output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ] = self.CORRUPT_VALUE

        self.updateConv()

    def _save_output_size(self, module, input, output):
        shape = list(output.size())
        dim = len(shape)

        self.LAYER_TYPE.append(module)
        self.LAYER_DIM.append(dim)
        self.OUTPUT_SIZE.append(shape)

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def get_layer_types(self):
        return self._INJ_LAYER_TYPES

    def updateConv(self, value=1):
        self.CURR_LAYER += value

    def reset_curr_layer(self):
        self.CURR_LAYER = 0

    def set_corrupt_layer(self, value):
        self.CORRUPT_LAYER = value

    def get_curr_layer(self):
        return self.CURR_LAYER

    def get_corrupt_layer(self):
        return self.CORRUPT_LAYER

    def get_total_batches(self):
        return self._BATCH_SIZE

    def get_total_layers(self):
        return len(self.OUTPUT_SIZE)

    def get_fmaps_num(self, layer):
        return self.OUTPUT_SIZE[layer][1]

    def get_fmaps_H(self, layer):
        return self.OUTPUT_SIZE[layer][2]

    def get_fmaps_W(self, layer):
        return self.OUTPUT_SIZE[layer][3]

    def get_fmap_HW(self, layer):
        return (self.get_fmaps_H(layer), self.get_fmaps_W(layer))

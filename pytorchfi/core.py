"""
pytorchfi.core contains the core functionality for fault injections.
"""

import copy
import logging
import warnings

import torch
import torch.nn as nn


class fault_injection:
    def __init__(self, model, batch_size, input_shape=[3, 224, 224], layer_types=[nn.Conv2d], **kwargs):
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.ORIG_MODEL = model
        self.OUTPUT_SIZE = list()
        self.LAYERS_TYPE = list()
        self.LAYERS_DIM = list()
        self._INPUT_SHAPE = input_shape
        self._BATCH_SIZE = batch_size
        self._INJ_LAYER_TYPES = layer_types

        self.CORRUPTED_MODEL = None
        self.CURR_LAYER = 0
        self.HANDLES = list()
        self.CORRUPT_BATCH = list()
        self.CORRUPT_LAYER = list()
        self.CORRUPT_DIM1 = list()  # C
        self.CORRUPT_DIM2 = list()  # H
        self.CORRUPT_DIM3 = list()  # W
        self.CORRUPT_VALUE = list()

        self.CUSTOM_INJECTION = False
        self.INJECTION_FUNCTION = None

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)

        assert isinstance(input_shape, list), "Error: Input shape must be provided as a list."
        assert isinstance(batch_size, int) and batch_size >= 1, "Error: Batch size must be an integer greater than 1."
        assert len(layer_types) >= 0 , "Error: At least one layer type must be selected."

        handles, shapes = self._traverseModelAndSetHooks(
            self.ORIG_MODEL, self._INJ_LAYER_TYPES
        )

        dummy_shape = (1, *self._INPUT_SHAPE)  # profiling only needs one batch element
        model_dtype = next(model.parameters()).dtype
        device = "cuda" if self.use_cuda else None
        _dummyTensor = torch.randn(
            dummy_shape, dtype=model_dtype, device=device
        )

        self.ORIG_MODEL(_dummyTensor)

        for i in range(len(handles)):
            handles[i].remove()

        logging.info("Input shape:")
        logging.info(dummy_shape[1:])

        logging.info("Model layer sizes:")
        logging.info(
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
        logging.info("Reset fault injector")

    def _fi_state_reset(self):
        (
            self.CURR_LAYER,
            self.CORRUPT_BATCH,
            self.CORRUPT_LAYER,
            self.CORRUPT_DIM1,
            self.CORRUPT_DIM2,
            self.CORRUPT_DIM3,
            self.CORRUPT_VALUE,
        ) = (0, list(), list(), list(), list(), list(), list())

        for i in range(len(self.HANDLES)):
            self.HANDLES[i].remove()

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

    def _traverseModelAndSetHooksNeurons(self, model, layer_types, customInj, injFunc):
        handles = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                for i in layer_types:
                    if isinstance(layer, i):
                        hook = injFunc if customInj else self._set_value
                        handles.append(layer.register_forward_hook(hook))
            # unpack node
            else:
                subHandles = self._traverseModelAndSetHooksNeurons(layer, layer_types, customInj, injFunc)
                for i in subHandles:
                    handles.append(i)

        return handles

    def declare_weight_fi(self, **kwargs):
        self._fi_state_reset()
        CUSTOM_INJECTION = False
        CUSTOM_FUNCTION = False

        if kwargs:
            if "function" in kwargs:
                CUSTOM_INJECTION, CUSTOM_FUNCTION = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", list())
                corrupt_k = kwargs.get("k", list())
                corrupt_c = kwargs.get("c", list())
                corrupt_kH = kwargs.get("h", list())
                corrupt_kW = kwargs.get("w", list())
            else:
                corrupt_layer = kwargs.get("layer_num", )
                corrupt_k = kwargs.get("k", list())
                corrupt_c = kwargs.get("c", list())
                corrupt_kH = kwargs.get("h", list())
                corrupt_kW = kwargs.get("w", list())
                corrupt_value = kwargs.get("value", list())
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
                self.CORRUPT_LAYER = kwargs.get("layer_num", list())
                self.CORRUPT_BATCH = kwargs.get("batch", list())
                self.CORRUPT_DIM1 = kwargs.get("c", list())
                self.CORRUPT_DIM2 = kwargs.get("h", list())
                self.CORRUPT_DIM3 = kwargs.get("w", list())
            else:
                self.CORRUPT_LAYER = kwargs.get("layer_num", list())
                self.CORRUPT_BATCH = kwargs.get("batch", list())
                self.CORRUPT_DIM1 = kwargs.get("c", list())
                self.CORRUPT_DIM2 = kwargs.get("h", list())
                self.CORRUPT_DIM3 = kwargs.get("w", list())
                self.CORRUPT_VALUE = kwargs.get("value", list())

                logging.info("Declaring Specified Fault Injector")
                logging.info("Convolution: %s" % self.CORRUPT_LAYER)
                logging.info("Batch, x, y, z:")
                logging.info(
                    "%s, %s, %s, %s"
                    % (
                        self.CORRUPT_BATCH,
                        self.CORRUPT_DIM1,
                        self.CORRUPT_DIM2,
                        self.CORRUPT_DIM3,
                    )
                )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.checkBounds(self.CORRUPT_BATCH, self.CORRUPT_LAYER, self.CORRUPT_DIM1, self.CORRUPT_DIM2, self.CORRUPT_DIM3)

        self.CORRUPTED_MODEL = copy.deepcopy(self.ORIG_MODEL)
        handles_neurons = self._traverseModelAndSetHooksNeurons(
            self.CORRUPTED_MODEL,
            self._INJ_LAYER_TYPES,
            CUSTOM_INJECTION,
            INJECTION_FUNCTION)

        for i in handles_neurons:
            self.HANDLES.append(i)

        return self.CORRUPTED_MODEL

    def checkBounds(self, b, l, dim1, dim2, dim3):
        assert len(b) == len(l), "Injection location missing values."
        assert len(b) == len(dim1), "Injection location missing values."
        assert len(b) == len(dim2), "Injection location missing values."
        assert len(b) == len(dim3), "Injection location missing values."

        logging.info("Checking bounds before runtime")
        for i in range(len(b)):
            self.assert_inj_bounds(i)



    def assert_inj_bounds(self, index, **kwargs):
        assert (index >= 0
        ), "Invalid injection index: %d" %(index)
        assert (
                self.CORRUPT_BATCH[index] < self.get_total_batches()
        ), "%d < %d: Invalid batch element!" %(self.CORRUPT_BATCH[index], self.get_total_batches())
        assert (
            self.CORRUPT_LAYER[index] < self.get_total_layers()
        ), "%d < %d: Invalid layer!" %(self.CORRUPT_LAYER[index], self.get_total_layers())

        corruptLayerNum = self.CORRUPT_LAYER[index]
        layerType = self.LAYERS_TYPE[corruptLayerNum]
        layerDim = self.LAYERS_DIM[corruptLayerNum]
        layerShape = self.OUTPUT_SIZE[corruptLayerNum]

        assert (
            self.CORRUPT_DIM1[index]
            < layerShape[1]
        ), "%d < %d: Out of bounds error in Dimension 1!" %(self.CORRUPT_DIM1[index], layerShape[1])
        if layerDim > 2:
            assert (
                self.CORRUPT_DIM2[index]
                < layerShape[2]
            ), "%d < %d: Out of bounds error in Dimension 2!" %(self.CORRUPT_DIM2[index], layerShape[2])
            assert (
                self.CORRUPT_DIM3[index]
                < layerShape[3]
            ), "%d < %d: Out of bounds error in Dimension 3!" %(self.CORRUPT_DIM3[index], layerShape[3])

        if layerDim <= 2:
            if self.CORRUPT_DIM2[index] != None or self.CORRUPT_DIM3[index] != None:
                warnings.warn("Values in Dim2 and Dim3 ignored, since layer is %s" %(layerType))

        logging.info("Finished checking bounds on inj '%d'"%(index))

    def _set_value(self, module, input, output):
        logging.info("Processing hook of Layer %d: %s" %(self.get_curr_layer(), self.get_layer_type(self.get_curr_layer())))
        inj_list = list(
            filter(
                lambda x: self.CORRUPT_LAYER[x] == self.get_curr_layer(),
                range(len(self.CORRUPT_LAYER)),
            )
        )

        layerDim = self.LAYERS_DIM[self.get_curr_layer()]

        logging.info("Layer %d injection list size: %d" %(self.get_curr_layer(), len(inj_list)))
        if layerDim == 2:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    "Original value at [%d][%d]: %d"
                    % (
                        self.CORRUPT_BATCH[i],
                        self.CORRUPT_DIM1[i],
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]]
                    )
                )
                logging.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]
                ] = self.CORRUPT_VALUE[i]

        elif layerDim == 4:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    "Original value at [%d][%d][%d][%d]: %d"
                    % (
                        self.CORRUPT_BATCH[i],
                        self.CORRUPT_DIM1[i],
                        self.CORRUPT_DIM2[i],
                        self.CORRUPT_DIM3[i],
                        output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][
                            self.CORRUPT_DIM2[i]
                        ][self.CORRUPT_DIM3[i]],
                    )
                )
                logging.info("Changing value to %d" % self.CORRUPT_VALUE[i])
                output[self.CORRUPT_BATCH[i]][self.CORRUPT_DIM1[i]][self.CORRUPT_DIM2[i]][
                    self.CORRUPT_DIM3[i]
                ] = self.CORRUPT_VALUE[i]

        self.updateLayer()

    def _save_output_size(self, module, input, output):
        shape = list(output.size())
        dim = len(shape)

        self.LAYERS_TYPE.append(type(module))
        self.LAYERS_DIM.append(dim)
        self.OUTPUT_SIZE.append(shape)

    def get_original_model(self):
        return self.ORIG_MODEL

    def get_corrupted_model(self):
        return self.CORRUPTED_MODEL

    def get_output_size(self):
        return self.OUTPUT_SIZE

    def get_layer_type(self, layer_num):
        return self.LAYERS_TYPE[layer_num]

    def get_layer_dim(self, layer_num):
        return self.LAYERS_DIM[layer_num]

    def get_inj_layer_types(self):
        return self._INJ_LAYER_TYPES

    def updateLayer(self, value=1):
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

    def print_pytorchfi_layer_summary(self):
        summary_str = "==================== PYTORCHFI INIT SUMMARY ====================="+ "\n\n"

        summary_str += "Layer types allowing injections:\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        for l_type in self._INJ_LAYER_TYPES:
            summary_str += "{:>5}".format("- ")
            substring = str(l_type).split(".")[-1].split("'")[0]
            summary_str += substring
            # summary_str += "{:>15}".format(
            #     str(substring),
            # )
            summary_str += "\n"
        summary_str += "\n"

        summary_str += "Model Info:\n"
        summary_str += "----------------------------------------------------------------" + "\n"

        summary_str += "   - Shape of input into the model: ("
        for dim in self._INPUT_SHAPE:
            summary_str += str(dim) + " "
        summary_str += ")\n"

        summary_str += "   - Batch Size: " + str(self._BATCH_SIZE) + "\n"
        summary_str += "   - CUDA Enabled: " + str(self.use_cuda) + "\n\n"

        summary_str += "Layer Info:\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        line_new = "{:>5}  {:>15}  {:>15} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Output Shape")
        summary_str += line_new + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        for layer in range(len(self.OUTPUT_SIZE)):
            line_new = "{:>5}  {:>15}  {:>15} {:>20}".format(
                layer,
                str(self.LAYERS_TYPE[layer]).split(".")[-1].split("'")[0],
                str(self.LAYERS_DIM[layer]),
                str(self.OUTPUT_SIZE[layer]),
            )
            summary_str += line_new + "\n"

        summary_str += "================================================================" + "\n"

        return summary_str


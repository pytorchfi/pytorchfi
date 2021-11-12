"""pytorchfi.core contains the core functionality for fault injections"""

import copy
import logging
import warnings

import torch
import torch.nn as nn


class fault_injection:
    def __init__(self, model, batch_size, input_shape=None, layer_types=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        if layer_types is None:
            layer_types = [nn.Conv2d]
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.original_model = model
        self.output_size = []
        self.layers_type = []
        self.layers_dim = []
        self.weights_size = []

        self._input_shape = input_shape
        self._batch_size = batch_size
        self._inj_layer_types = layer_types

        self.corrupted_model = None
        self.current_layer = 0
        self.handles = []
        self.corrupt_batch = []
        self.corrupt_layer = []
        self.corrupt_dim1 = []  # C
        self.corrupt_dim2 = []  # H
        self.corrupt_dim3 = []  # W
        self.corrupt_value = []

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)

        if not isinstance(input_shape, list):
            raise AssertionError("Error: Input shape must be provided as a list.")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise AssertionError("Error: Batch size must be an integer greater than 1.")
        if len(layer_types) < 0:
            raise AssertionError("Error: At least one layer type must be selected.")

        handles, _shapes, self.weights_size = self._traverse_model_set_hooks(
            self.original_model, self._inj_layer_types
        )

        dummy_shape = (1, *self._input_shape)  # profiling only needs one batch element
        model_dtype = next(model.parameters()).dtype
        device = "cuda" if self.use_cuda else None
        _dummyTensor = torch.randn(dummy_shape, dtype=model_dtype, device=device)

        self.original_model(_dummyTensor)

        for index, _handle in enumerate(handles):
            handles[index].remove()

        logging.info("Input shape:")
        logging.info(dummy_shape[1:])

        logging.info("Model layer sizes:")
        logging.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.output_size
                ]
            )
        )

    def fi_reset(self):
        self._fi_state_reset()
        self.corrupted_model = None
        logging.info("Reset fault injector")

    def _fi_state_reset(self):
        (
            self.current_layer,
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim1,
            self.corrupt_dim2,
            self.corrupt_dim3,
            self.corrupt_value,
        ) = (0, [], [], [], [], [], [])

        for index, _handle in enumerate(self.handles):
            self.handles[index].remove()

    def _traverse_model_set_hooks(self, model, layer_types):
        handles = []
        output_shape = []
        weights_shape = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in layer_types:
                    handles.append(layer.register_forward_hook(self._save_output_size))
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            # neurons
                            handles.append(
                                layer.register_forward_hook(self._save_output_size)
                            )
                            output_shape.append(layer)

                            # weights
                            weights_shape.append(layer.weight.shape)
            # unpack node
            else:
                subHandles, subBase, subWeight = self._traverse_model_set_hooks(
                    layer, layer_types
                )
                for i in subHandles:
                    handles.append(i)
                for i in subBase:
                    output_shape.append(i)
                for i in subWeight:
                    weights_shape.append(i)

        return (handles, output_shape, weights_shape)

    def _traverse_model_set_hooks_neurons(self, model, layer_types, customInj, injFunc):
        handles = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in layer_types:
                    hook = injFunc if customInj else self._set_value
                    handles.append(layer.register_forward_hook(hook))
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            hook = injFunc if customInj else self._set_value
                            handles.append(layer.register_forward_hook(hook))
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(
                    layer, layer_types, customInj, injFunc
                )
                for i in subHandles:
                    handles.append(i)

        return handles

    def declare_weight_fi(self, **kwargs):
        self._fi_state_reset()
        custom_injection = False
        CUSTOM_FUNCTION = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, CUSTOM_FUNCTION = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", [])
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
            else:
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_value = kwargs.get("value", [])
        else:
            raise ValueError("Please specify an injection or injection function")

        self.corrupted_model = copy.deepcopy(self.original_model)
        corrupt_idx = [corrupt_k, corrupt_c, corrupt_kH, corrupt_kW]

        current_layer = 0
        for layer in self.corrupted_model.children():
            for i in self.get_inj_layer_types():
                if isinstance(layer, i):
                    if current_layer == corrupt_layer:
                        corrupt_idx = (
                            tuple(corrupt_idx)
                            if isinstance(corrupt_idx, list)
                            else corrupt_idx
                        )
                        orig_value = layer.weight[corrupt_idx].item()
                        if custom_injection:
                            corrupt_value = CUSTOM_FUNCTION(layer.weight, corrupt_idx)
                        layer.weight[corrupt_idx] = corrupt_value

                        logging.info("Weight Injection")
                        logging.info("Layer index: %s", corrupt_layer)
                        logging.info("Module: %s", layer)
                        logging.info("Original value: %s", orig_value)
                        logging.info("Injected value: %s", corrupt_value)
                    current_layer += 1
        return self.corrupted_model

    def declare_neuron_fi(self, **kwargs):
        self._fi_state_reset()
        custom_injection = False
        injection_function = False

        if kwargs:
            if "function" in kwargs:
                logging.info("Declaring Custom Function")
                custom_injection, injection_function = True, kwargs.get("function")
            else:
                logging.info("Declaring Specified Fault Injector")
                self.corrupt_value = kwargs.get("value", [])

            self.corrupt_layer = kwargs.get("layer_num", [])
            self.corrupt_batch = kwargs.get("batch", [])
            self.corrupt_dim1 = kwargs.get("dim1", [])
            self.corrupt_dim2 = kwargs.get("dim2", [])
            self.corrupt_dim3 = kwargs.get("dim3", [])

            logging.info("Convolution: %s", self.corrupt_layer)
            logging.info("Batch, x, y, z:")
            logging.info(
                "%s, %s, %s, %s",
                self.corrupt_batch,
                self.corrupt_dim1,
                self.corrupt_dim2,
                self.corrupt_dim3,
            )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.check_bounds(
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim1,
            self.corrupt_dim2,
            self.corrupt_dim3,
        )

        self.corrupted_model = copy.deepcopy(self.original_model)
        handles_neurons = self._traverse_model_set_hooks_neurons(
            self.corrupted_model,
            self._inj_layer_types,
            custom_injection,
            injection_function,
        )

        for i in handles_neurons:
            self.handles.append(i)

        return self.corrupted_model

    def check_bounds(self, b, l, dim1, dim2, dim3):
        if len(b) != len(l):
            raise AssertionError("Injection location missing values.")
        if len(b) != len(dim1):
            raise AssertionError("Injection location missing values.")
        if len(b) != len(dim2):
            raise AssertionError("Injection location missing values.")
        if len(b) != len(dim3):
            raise AssertionError("Injection location missing values.")

        logging.info("Checking bounds before runtime")
        for i in range(len(b)):
            self.assert_inj_bounds(i)

    def assert_inj_bounds(self, index):
        if index < 0:
            raise AssertionError("Invalid injection index: %d" % (index))
        if self.corrupt_batch[index] >= self.get_total_batches():
            raise AssertionError(
                "%d < %d: Invalid batch element!"
                % (
                    self.corrupt_batch[index],
                    self.get_total_batches(),
                )
            )
        if self.corrupt_layer[index] >= self.get_total_layers():
            raise AssertionError(
                "%d < %d: Invalid layer!"
                % (
                    self.corrupt_layer[index],
                    self.get_total_layers(),
                )
            )

        corrupt_layer_num = self.corrupt_layer[index]
        layer_type = self.layers_type[corrupt_layer_num]
        layer_dim = self.layers_dim[corrupt_layer_num]
        layer_shape = self.output_size[corrupt_layer_num]

        if self.corrupt_dim1[index] >= layer_shape[1]:
            raise AssertionError(
                "%d < %d: Out of bounds error in Dimension 1!"
                % (
                    self.corrupt_dim1[index],
                    layer_shape[1],
                )
            )

        if layer_dim > 2 and self.corrupt_dim2[index] >= layer_shape[2]:
            raise AssertionError(
                "%d < %d: Out of bounds error in Dimension 2!"
                % (
                    self.corrupt_dim2[index],
                    layer_shape[2],
                )
            )

        if layer_dim > 3 and self.corrupt_dim3[index] >= layer_shape[3]:
            raise AssertionError(
                "%d < %d: Out of bounds error in Dimension 3!"
                % (
                    self.corrupt_dim3[index],
                    layer_shape[3],
                )
            )

        if layer_dim <= 2 and (
            self.corrupt_dim2[index] is not None or self.corrupt_dim3[index] is not None
        ):
            warnings.warn(
                "Values in Dim2 and Dim3 ignored, since layer is %s" % (layer_type)
            )

        if layer_dim <= 3 and self.corrupt_dim3[index] is not None:
            warnings.warn("Values Dim3 ignored, since layer is %s" % (layer_type))

        logging.info("Finished checking bounds on inj '%d'", (index))

    def _set_value(self, module, input_val, output):
        logging.info(
            "Processing hook of Layer %d: %s",
            self.get_current_layer(),
            self.get_layer_type(self.get_current_layer()),
        )
        inj_list = list(
            filter(
                lambda x: self.corrupt_layer[x] == self.get_current_layer(),
                range(len(self.corrupt_layer)),
            )
        )

        layer_dim = self.layers_dim[self.get_current_layer()]

        logging.info(
            "Layer %d injection list size: %d", self.get_current_layer(), len(inj_list)
        )
        if layer_dim == 2:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    "Original value at [%d][%d]: %f",
                    self.corrupt_batch[i],
                    self.corrupt_dim1[i],
                    output[self.corrupt_batch[i]][self.corrupt_dim1[i]],
                )
                logging.info("Changing value to %f", self.corrupt_value[i])
                output[self.corrupt_batch[i]][
                    self.corrupt_dim1[i]
                ] = self.corrupt_value[i]
        elif layer_dim == 3:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    "Original value at [%d][%d][%d]: %f",
                    self.corrupt_batch[i],
                    self.corrupt_dim1[i],
                    self.corrupt_dim2[i],
                    output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                        self.corrupt_dim2[i]
                    ],
                )
                logging.info("Changing value to %f", self.corrupt_value[i])
                output[self.corrupt_batch[i]][
                    self.corrupt_dim1[i], self.corrupt_dim2[i]
                ] = self.corrupt_value[i]
        elif layer_dim == 4:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    "Original value at [%d][%d][%d][%d]: %f",
                    self.corrupt_batch[i],
                    self.corrupt_dim1[i],
                    self.corrupt_dim2[i],
                    self.corrupt_dim3[i],
                    output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                        self.corrupt_dim2[i]
                    ][self.corrupt_dim3[i]],
                )
                logging.info("Changing value to %f", self.corrupt_value[i])
                output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                    self.corrupt_dim2[i]
                ][self.corrupt_dim3[i]] = self.corrupt_value[i]

        self.updateLayer()

    def _save_output_size(self, module, input_val, output):
        shape = list(output.size())
        dim = len(shape)

        self.layers_type.append(type(module))
        self.layers_dim.append(dim)
        self.output_size.append(shape)

    def get_original_model(self):
        return self.original_model

    def get_corrupted_model(self):
        return self.corrupted_model

    def get_output_size(self):
        return self.output_size

    def get_weights_size(self, layer_num):
        return self.weights_size[layer_num]

    def get_weights_dim(self, layer_num):
        return len(self.weights_size[layer_num])

    def get_layer_type(self, layer_num):
        return self.layers_type[layer_num]

    def get_layer_dim(self, layer_num):
        return self.layers_dim[layer_num]

    def get_layer_shape(self, layer_num):
        return self.output_size[layer_num]

    def get_inj_layer_types(self):
        return self._inj_layer_types

    def updateLayer(self, value=1):
        self.current_layer += value

    def reset_current_layer(self):
        self.current_layer = 0

    def set_corrupt_layer(self, value):
        self.corrupt_layer = value

    def get_current_layer(self):
        return self.current_layer

    def get_corrupt_layer(self):
        return self.corrupt_layer

    def get_total_batches(self):
        return self._batch_size

    def get_total_layers(self):
        return len(self.output_size)

    def get_fmaps_num(self, layer):
        return self.output_size[layer][1]

    def get_fmaps_H(self, layer):
        return self.output_size[layer][2]

    def get_fmaps_W(self, layer):
        return self.output_size[layer][3]

    def get_fmap_HW(self, layer):
        return (self.get_fmaps_H(layer), self.get_fmaps_W(layer))

    def print_pytorchfi_layer_summary(self):
        summary_str = (
            "============================ PYTORCHFI INIT SUMMARY =============================="
            + "\n\n"
        )

        summary_str += "Layer types allowing injections:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for l_type in self._inj_layer_types:
            summary_str += "{:>5}".format("- ")
            substring = str(l_type).split(".")[-1].split("'")[0]
            summary_str += substring + "\n"
        summary_str += "\n"

        summary_str += "Model Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )

        summary_str += "   - Shape of input into the model: ("
        for dim in self._input_shape:
            summary_str += str(dim) + " "
        summary_str += ")\n"

        summary_str += "   - Batch Size: " + str(self._batch_size) + "\n"
        summary_str += "   - CUDA Enabled: " + str(self.use_cuda) + "\n\n"

        summary_str += "Layer Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Weight Shape", "Output Shape"
        )
        summary_str += line_new + "\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for layer, _dim in enumerate(self.output_size):
            line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
                layer,
                str(self.layers_type[layer]).split(".")[-1].split("'")[0],
                str(self.layers_dim[layer]),
                str(list(self.weights_size[layer])),
                str(self.output_size[layer]),
            )
            summary_str += line_new + "\n"

        summary_str += (
            "=================================================================================="
            + "\n"
        )

        print(summary_str)
        return summary_str

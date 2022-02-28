"""pytorchfi.core contains the core functionality for fault injections"""

import copy
import logging
import warnings

import torch
import torch.nn as nn


class FaultInjection:
    def __init__(self, model, batch_size, input_shape=None, layer_types=None, **kwargs):
        if not input_shape:
            input_shape = [3, 224, 224]
        if not layer_types:
            layer_types = [nn.Conv2d]
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.original_model = model
        self.output_size = []
        self.layers_type = []
        self.layers_dim = []
        self.weights_size = []
        self.batch_size = batch_size

        self._input_shape = input_shape
        self._inj_layer_types = layer_types

        self.corrupted_model = None
        self.current_layer = 0
        self.handles = []
        self.corrupt_batch = []
        self.corrupt_layer = []
        self.corrupt_dim = [[], [], []]  # C, H, W
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
        _dummy_tensor = torch.randn(dummy_shape, dtype=model_dtype, device=device)

        self.original_model(_dummy_tensor)

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
        logging.info("Fault injector reset.")

    def _fi_state_reset(self):
        (
            self.current_layer,
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
            self.corrupt_value,
        ) = (0, [], [], [[], [], []], [])

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
                subhandles, subbase, subweight = self._traverse_model_set_hooks(
                    layer, layer_types
                )
                for i in subhandles:
                    handles.append(i)
                for i in subbase:
                    output_shape.append(i)
                for i in subweight:
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
        custom_function = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, custom_function = True, kwargs.get("function")
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

        # TODO: bound check here

        self.corrupted_model = copy.deepcopy(self.original_model)

        current_weight_layer = 0
        for layer in self.corrupted_model.modules():
            if isinstance(layer, tuple(self._inj_layer_types)):
                inj_list = list(
                    filter(
                        lambda x: corrupt_layer[x] == current_weight_layer,
                        range(len(corrupt_layer)),
                    )
                )

                for inj in inj_list:
                    corrupt_idx = tuple(
                        [
                            corrupt_k[inj],
                            corrupt_c[inj],
                            corrupt_kH[inj],
                            corrupt_kW[inj],
                        ]
                    )
                    orig_value = layer.weight[corrupt_idx].item()

                    with torch.no_grad():
                        if custom_injection:
                            corrupt_value = custom_function(layer.weight, corrupt_idx)
                            layer.weight[corrupt_idx] = corrupt_value
                        else:
                            layer.weight[corrupt_idx] = corrupt_value[inj]

                    logging.info("Weight Injection")
                    logging.info(f"Layer index: {corrupt_layer}")
                    logging.info(f"Module: {layer}")
                    logging.info(f"Original value: {orig_value}")
                    logging.info(f"Injected value: {layer.weight[corrupt_idx]}")
                current_weight_layer += 1
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
            self.corrupt_dim[0] = kwargs.get("dim1", [])
            self.corrupt_dim[1] = kwargs.get("dim2", [])
            self.corrupt_dim[2] = kwargs.get("dim3", [])

            logging.info(f"Convolution: {self.corrupt_layer}")
            logging.info("Batch, x, y, z:")
            logging.info(
                f"{self.corrupt_batch}, {self.corrupt_dim[0]}, {self.corrupt_dim[1]}, {self.corrupt_dim[2]}"
            )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.check_bounds(
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
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

    def check_bounds(self, batch, layer, dim):
        if (
            len(batch) != len(layer)
            or len(batch) != len(dim[0])
            or len(batch) != len(dim[1])
            or len(batch) != len(dim[2])
        ):
            raise AssertionError("Injection location missing values.")

        logging.info("Checking bounds before runtime")
        for i in range(len(batch)):
            self.assert_inj_bounds(i)

    def assert_inj_bounds(self, index):
        if index < 0:
            raise AssertionError(f"Invalid injection index: {index}")
        if self.corrupt_batch[index] >= self.batch_size:
            raise AssertionError(
                f"{self.corrupt_batch[index]} < {self.batch_size()}: Invalid batch element!"
            )
        if self.corrupt_layer[index] >= len(self.output_size):
            raise AssertionError(
                f"{self.corrupt_layer[index]} < {len(self.output_size)}: Invalid layer!"
            )

        corrupt_layer_num = self.corrupt_layer[index]
        layer_type = self.layers_type[corrupt_layer_num]
        layer_dim = self.layers_dim[corrupt_layer_num]
        layer_shape = self.output_size[corrupt_layer_num]

        for d in range(1, 4):
            if layer_dim > d and self.corrupt_dim[d - 1][index] >= layer_shape[d]:
                raise AssertionError(
                    f"{self.corrupt_dim[d - 1][index]} < {layer_shape[d]}: Out of bounds error in Dimension {d}!"
                )

        if layer_dim <= 2 and (
            self.corrupt_dim[1][index] is not None
            or self.corrupt_dim[2][index] is not None
        ):
            warnings.warn(
                f"Values in Dim2 and Dim3 ignored, since layer is {layer_type}"
            )

        if layer_dim <= 3 and self.corrupt_dim[2][index] is not None:
            warnings.warn(f"Values Dim3 ignored, since layer is {layer_type}")

        logging.info(f"Finished checking bounds on inj '{index}'")

    def _set_value(self, module, input_val, output):
        logging.info(
            f"Processing hook of Layer {self.current_layer}: {self.layers_type[self.current_layer]}"
        )
        inj_list = list(
            filter(
                lambda x: self.corrupt_layer[x] == self.current_layer,
                range(len(self.corrupt_layer)),
            )
        )

        layer_dim = self.layers_dim[self.current_layer]

        logging.info(f"Layer {self.current_layer} injection list size: {len(inj_list)}")
        if layer_dim == 2:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]]}"
                )
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 3:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]]}"
                )
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i], self.corrupt_dim[1][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 4:
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                logging.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}"
                )
                logging.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = self.corrupt_value[i]

        self.current_layer += 1

    def _save_output_size(self, module, input_val, output):
        shape = list(output.size())
        dim = len(shape)

        self.layers_type.append(type(module))
        self.layers_dim.append(dim)
        self.output_size.append(shape)

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

    def get_total_layers(self):
        return len(self.output_size)

    def get_fmaps_C(self, layer):
        return self.output_size[layer][1]

    def get_fmaps_H(self, layer):
        return self.output_size[layer][2]

    def get_fmaps_W(self, layer):
        return self.output_size[layer][3]

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

        summary_str += "   - Batch Size: " + str(self.batch_size) + "\n"
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

        logging.info(summary_str)
        return summary_str

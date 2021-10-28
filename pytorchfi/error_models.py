"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import random
import logging
import torch
from pytorchfi import core


# Helper Functions


def random_batch_element(pfi):
    return random.randint(0, pfi.get_total_batches() - 1)


def random_neuron_location(pfi, layer=-1):
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_layer_dim(layer)
    shape = pfi.get_layer_shape(layer)

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

    return (layer, dim1_rand, dim2_rand, dim3_rand)


def random_weight_location(pfi, layer=-1):
    loc = []
    total_layers = pfi.get_total_layers()

    corrupt_layer = random.randint(0, total_layers - 1) if layer == -1 else layer
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    if curr_layer != total_layers or len(loc) != 5:
        raise AssertionError

    return loc


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


# Neuron Perturbation Models


# single random neuron error in single batch element
def random_neuron_inj(pfi, min_val=-1, max_val=1):
    b = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi.declare_neuron_fi(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(pfi, min_val=-1, max_val=1, rand_loc=True, rand_val=True):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    if not rand_loc:
        (layer, C, H, W) = random_neuron_location(pfi)
    if not rand_val:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi.get_total_batches()):
        if rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi)
        if rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi, min_val=-1, max_val=1):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi)
    for i in range(pfi.get_total_layers()):
        (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        batch.append(b)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi, min_val=-1, max_val=1, rand_loc=True, rand_val=True
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi.get_total_layers()):
        if not rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        if not rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi.get_total_batches()):
            if rand_loc:
                (layer, C, H, W) = random_neuron_location(pfi, layer=i)
            if rand_val:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, batch_size, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    @staticmethod
    def _twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("Original Value: %d", orig_value)

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("Quantum: %d", quantum)
        logging.info("Twos Couple: %d", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("Bits: %s", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info("sign extend bits %s", bits)

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("New bits: %s", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("Out: %s", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("New Value: %d", new_value)

        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input_val, output):
        corrupt_conv_set = self.get_corrupt_layer()
        range_max = self.get_conv_max(self.get_current_layer())
        logging.info("Current layer: %s", self.get_current_layer())
        logging.info("Range_max: %s", range_max)

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_current_layer(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                    self.corrupt_dim2[i]
                ][self.corrupt_dim3[i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("Random Bit: %d", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.corrupt_batch[i]][self.corrupt_dim1[i]][
                    self.corrupt_dim2[i]
                ][self.corrupt_dim3[i]] = new_value

        else:
            if self.get_current_layer() == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim1][
                    self.corrupt_dim2
                ][self.corrupt_dim3]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("Random Bit: %d", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim1][self.corrupt_dim2][
                    self.corrupt_dim3
                ] = new_value

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()


def random_neuron_single_bit_inj_batched(pfi, layer_ranges, rand_loc=True):
    pfi.set_conv_max(layer_ranges)
    batch, layer_num, c_rand, h_rand, w_rand = ([] for i in range(5))

    if not rand_loc:
        (layer, C, H, W) = random_neuron_location(pfi)

    for i in range(pfi.get_total_batches()):
        if rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)

    return pfi.declare_neuron_fi(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        function=pfi.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi, layer_ranges):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)

    return pfi.declare_neuron_fi(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi.single_bit_flip_signed_across_batch,
    )


# Weight Perturbation Models


def random_weight_inj(pfi, corrupt_conv=-1, min_val=-1, max_val=1):
    layer, k, c_in, kH, kW = random_weight_location(pfi, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi.declare_weight_fi(
        layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, value=faulty_val
    )


def zero_func_rand_weight(pfi):
    layer, k, c_in, kH, kW = random_weight_location(pfi)
    return pfi.declare_weight_fi(
        function=_zero_rand_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData

"""
pytorchfi.errormodels provides different error models out-of-the-box for use.
"""

import random
import logging
import torch
from pytorchfi import core


"""
helper functions
"""


def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, conv=-1):
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)

    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (conv, c, h, w)


def random_weight_location(pfi_model, conv=-1):
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1

    assert curr_layer == pfi_model.get_total_conv()
    assert len(loc) == 5

    return tuple(loc)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


"""
Neuron Perturbation Models
"""


# single random neuron error in single batch element
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (conv, C, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=b, conv_num=conv, c=C, h=H, w=W, value=err_val
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    if not randLoc:
        (conv, C, H, W) = random_neuron_location(pfi_model)
    if not randVal:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model)
        if randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi_model, min_val=-1, max_val=1):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi_model)
    for i in range(pfi_model.get_total_conv()):
        (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
        batch.append(b)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi_model.get_total_conv()):
        if not randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
        if not randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi_model.get_total_batches()):
            if randLoc:
                (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
            if randVal:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            conv_num.append(conv)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, h, w, batch_size, **kwargs):
        super().__init__(model, h, w, batch_size, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    def _twos_comp_shifted(self, val, nbits):
        if val < 0:
            val = (1 << nbits) + val
        else:
            val = self._twos_comp(val, nbits)
        return val

    def _twos_comp(self, val, bits):
        # compute the 2's complement of int value val
        if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val  # return positive value as is

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("orig value:", orig_value)

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        assert len(bits) == total_bits
        logging.info("sign extend bits", bits)

        # flip a bit
        # use MSB -> LSB indexing
        assert bit_pos < total_bits

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        assert bits_str_new.isdigit()
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input, output):
        corrupt_conv_set = self.get_corrupt_conv()
        range_max = self.get_conv_max(self.get_curr_conv())
        logging.info("curr_conv", self.get_curr_conv())
        logging.info("range_max", range_max)

        if type(corrupt_conv_set) == list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_curr_conv(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                    self.CORRUPT_H[i]
                ][self.CORRUPT_W[i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                    self.CORRUPT_W[i]
                ] = new_value

        else:
            self.assert_inj_bounds()
            if self.get_curr_conv() == corrupt_conv_set:
                prev_value = output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ] = new_value

        self.updateConv()
        if self.get_curr_conv() >= self.get_total_conv():
            self.reset_curr_conv()


def random_neuron_single_bit_inj_batched(pfi_model, layer_ranges, randLoc=True):
    pfi_model.set_conv_max(layer_ranges)
    batch, conv_num, c_rand, h_rand, w_rand = ([] for i in range(5))

    if not randLoc:
        (conv, C, H, W) = random_neuron_location(pfi_model)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv_num,
        c=c_rand,
        h=h_rand,
        w=w_rand,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi_model, layer_ranges):
    pfi_model.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi_model)
    (conv, C, H, W) = random_neuron_location(pfi_model)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv,
        c=C,
        h=H,
        w=W,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


"""
Weight Perturbation Models
"""


def random_weight_inj(pfi_model, corrupt_conv=-1, min_val=-1, max_val=1):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_weight_fi(
        conv_num=conv, k=k, c=c_in, h=kH, w=kW, value=faulty_val
    )


def zeroFunc_rand_weight(pfi_model):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model)
    return pfi_model.declare_weight_fi(
        function=_zero_rand_weight, conv_num=conv, k=k, c=c_in, h=kH, w=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData

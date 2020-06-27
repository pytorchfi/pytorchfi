"""
pytorchfi.errormodels provides different error models out-of-the-box for use.
"""

import random

from pytorchfi import core


# ###################
#  helper functions #
# ###################
def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model):
    conv = random.randint(0, pfi_model.get_total_conv() - 1)
    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (conv, c, h, w)


def random_neuron_location_conv(pfi_model, conv):
    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (c, h, w)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


# ###################
#    Error Models   #
# ###################
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

    if randLoc is False:
        (conv, C, H, W) = random_neuron_location(pfi_model)
    if randVal is False:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc is True:
            (conv, C, H, W) = random_neuron_location(pfi_model)
        if randVal is True:
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
        (C, H, W) = random_neuron_location_conv(pfi_model, i)
        batch.append(b)
        conv_num.append(i)
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
        if randLoc is False:
            (C, H, W) = random_neuron_location_conv(pfi_model, i)
        if randVal is False:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi_model.get_total_batches()):
            if randLoc is True:
                (C, H, W) = random_neuron_location_conv(pfi_model, i)
            if randVal is True:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            conv_num.append(i)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


def zero_layer_weights(pfi_model, zero_layer=0):
    return pfi_model.declare_weight_fi(layer=zero_layer, zero=True)


def random_weight_inj(pfi_model, min_val=-1, max_val=1):
    return pfi_model.declare_weight_fi(
        rand=True, min_rand_val=min_val, max_rand_val=max_val
    )

"""
pytorchfi.errormodels provides different error models out-of-the-box for use.
"""

import random


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


def random_weight_location(pfi_model):
    loc = list()

    corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
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


# #################################
#    Neuron Perturbation Models   #
# #################################
# single random neuron error in single batch element
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (conv, C, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=b, conv_num=conv, c=C, h=H, w=W, value=err_val
    )


def random_neuron_single_bit_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    conv = random.randint(0, pfi_model.get_total_conv() - 1)
    pfi_model.setCorruptConv(conv)

    return pfi_model.declare_neuron_fi(function=_single_bit_flip_signed_across_batch)


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

def _single_bit_flip_signed_across_batch(self, input, output):
    curr_layer = pfi_model.getCurrConv()
    print(curr_layer)
    #if self.getCorruptConv == curr_layer:
    #    print("GOT HERE", curr_layer)


# #################################
#    Weight Perturbation Models   #
# #################################
def random_weight_inj(pfi_model, min_val=-1, max_val=1):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model)
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

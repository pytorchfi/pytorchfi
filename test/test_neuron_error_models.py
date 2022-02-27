import torch
import random
import pytest

from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.neuron_error_models import (
    single_bit_flip_func,
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
)
from .util_test import CIFAR10_set_up_custom


class TestNeuronErrorModels:
    """Testing neuron perturbation error models."""

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = False

        model, dataset = CIFAR10_set_up_custom(batch_size, workers)
        dataiter = iter(dataset)

        self.images, self.labels = dataiter.next()

        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=use_gpu,
        )

    def test_random_neuron_inj(self):
        # TODO make sure only one batch element is different
        corrupt_model = random_neuron_inj(self.p, min_val=10000, max_val=20000)

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

    @pytest.mark.parametrize(
        "loc, val",
        [(True, True), (False, True), (True, False), (False, False)],
    )
    def test_random_neuron_inj_batched(self, loc, val):
        # TODO make better test
        corrupt_model = random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, rand_loc=loc, rand_val=val
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

    def test_random_inj_per_layer(self):
        # TODO make better test
        corrupt_model = random_inj_per_layer(self.p, min_val=10000, max_val=20000)

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

    @pytest.mark.parametrize(
        "loc, val",
        [(True, True), (False, True), (True, False), (False, False)],
    )
    def test_random_inj_per_layer_batched(self, loc, val):
        # TODO make better test
        corrupt_model = random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, rand_loc=loc, rand_val=val
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError


class TestNeuronErrorModelsFunc:
    """Testing neuron perturbation error models."""

    def setup_class(self):
        torch.manual_seed(1)

        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = False

        model, dataset = CIFAR10_set_up_custom(batch_size, workers)
        dataiter = iter(dataset)

        self.images, self.labels = dataiter.next()

        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

        self.p = single_bit_flip_func(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=use_gpu,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)

        corrupt_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output[0].eq(self.golden_output[0])):
            raise AssertionError
        if torch.all(corrupt_output[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output[2].eq(self.golden_output[2])):
            raise AssertionError
        if torch.all(corrupt_output[3].eq(self.golden_output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)

        corrupt_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, rand_loc=False
        )
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output[0].eq(self.golden_output[0])):
            raise AssertionError
        if torch.all(corrupt_output[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output[2].eq(self.golden_output[2])):
            raise AssertionError
        if torch.all(corrupt_output[3].eq(self.golden_output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)

        corrupt_model = random_neuron_single_bit_inj(self.p, self.ranges)
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if not torch.all(corrupt_output[0].eq(self.golden_output[0])):
            raise AssertionError
        if not torch.all(corrupt_output[1].eq(self.golden_output[1])):
            raise AssertionError
        if not torch.all(corrupt_output[2].eq(self.golden_output[2])):
            raise AssertionError
        if torch.all(corrupt_output[3].eq(self.golden_output[3])):
            raise AssertionError

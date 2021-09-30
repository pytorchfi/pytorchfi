import torch
import random
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.error_models import (
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
        # random.seed(0)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = CIFAR10_set_up_custom(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
        )

    def test_random_neuron_inj(self):
        # TODO make sure only one batch element is different
        self.inj_model = random_neuron_inj(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_neuron_inj_batched_locTrue_valTrue(self):
        # TODO make sure only all batch elements are different
        self.inj_model = random_neuron_inj_batched(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_neuron_inj_batched_locFalse_valTrue(self):
        # TODO make better test
        self.inj_model = random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_neuron_inj_batched_locTrue_valFalse(self):
        # TODO make better test
        self.inj_model = random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_neuron_inj_batched_locFalse_valFalse(self):
        # TODO make better test
        self.inj_model = random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_inj_per_layer(self):
        # TODO make better test
        self.inj_model = random_inj_per_layer(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_inj_per_layer_batched_locTrue_valTrue(self):
        # TODO make better test
        self.inj_model = random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=True, randVal=True
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_inj_per_layer_batched_locFalse_valTrue(self):
        # TODO make better test
        self.inj_model = random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=True
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_inj_per_layer_batched_locTrue_valFalse(self):
        # TODO make better test
        self.inj_model = random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=True, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

    def test_random_inj_per_layer_batched_locFalse_valFalse(self):
        # TODO make better test
        self.inj_model = random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError


class TestNeuronErrorModelsFunc:
    """Testing neuron perturbation error models."""

    def setup_class(self):
        torch.manual_seed(1)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = CIFAR10_set_up_custom(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = single_bit_flip_func(
            self.model,
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
            bits=8,
        )
        self.ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453]

    def test_random_neuron_single_bit_inj_rand(self):
        random.seed(3)
        self.inj_model = random_neuron_single_bit_inj_batched(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_sameLoc(self):
        random.seed(2)
        self.inj_model = random_neuron_single_bit_inj_batched(
            self.p, self.ranges, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_random_neuron_single_bit_inj_single(self):
        random.seed(0)
        self.inj_model = random_neuron_single_bit_inj(self.p, self.ranges)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if not torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

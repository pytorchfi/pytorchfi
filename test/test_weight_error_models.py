import random

import torch

from pytorchfi.core import FaultInjection as pfi_core
from pytorchfi.weight_error_models import (
    multi_weight_inj,
    random_weight_inj,
    random_weight_location,
    zero_func_rand_weight,
)

from .util_test import CIFAR10_set_up_custom


class TestWeightErrorModels:
    """
    Testing weight perturbation error models.
    TODO: Update for Weights
    """

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = False

        model, self.dataset = CIFAR10_set_up_custom(batch_size, workers)
        dataiter = iter(self.dataset)
        self.images, self.labels = dataiter.next()

        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            layer_types=[torch.nn.Conv2d, torch.nn.Linear],
            use_cuda=use_gpu,
        )

    def test_random_weight_loc(self):
        random.seed(3)

        (a1, b1, c1, d1, e1) = random_weight_location(self.p)
        assert (a1, b1, c1, d1, e1) == ([1], [151], [16], [2], [4])

        (a2, b2, c2, d2, e2) = random_weight_location(self.p, layer=3)
        assert (a2, b2, c2, d2, e2) == ([3], [242], [320], [2], [0])

    def test_random_weight_inj(self):
        random.seed(2)

        corrupt_model = random_weight_inj(self.p, min_val=10000, max_val=20000)
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output.eq(self.golden_output))

    def test_random_weight_inj_conv(self):
        random.seed(1)

        corrupt_model = random_weight_inj(
            self.p, corrupt_layer=3, min_val=10000, max_val=20000
        )
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output.eq(self.golden_output))

    def test_random_weight_zero_inj(self):
        random.seed(2)

        corrupt_model = zero_func_rand_weight(self.p)
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output.eq(self.golden_output))

    def test_multi_weight_inj(self):
        random.seed(1)
        corrupt_model = multi_weight_inj(self.p)
        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)
        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

import torch
import random
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.error_models import (
    random_weight_inj,
    zero_func_rand_weight,
)

from .util_test import CIFAR10_set_up_custom


class TestWeightErrorModels:
    """Testing weight perturbation error models."""

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
            use_cuda=use_gpu,
        )

    def test_random_weight_inj(self):
        # TODO Update for Weights
        random.seed(2)
        self.inj_model = random_weight_inj(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.golden_output)):
            raise AssertionError

    def test_random_weight_inj_conv(self):
        # TODO Update for Weights
        random.seed(1)
        corrupt_model = random_weight_inj(
            self.p, corrupt_conv=3, min_val=10000, max_val=20000
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

    def test_random_weight_zero_inj(self):
        # TODO Update for Weights
        random.seed(2)
        corrupt_model = zero_func_rand_weight(self.p)

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        if torch.all(corrupt_output.eq(self.golden_output)):
            raise AssertionError

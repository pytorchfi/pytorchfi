import torch
import random
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.errormodels import (
    random_weight_inj,
    zeroFunc_rand_weight,
)

from .util_test import helper_setUp_CIFAR10_same


class TestWeightErrorModels:
    """
    Testing weight perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(0)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_random_weight_inj(self):
        # TODO Update for Weights
        random.seed(2)
        self.inj_model = random_weight_inj(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_weight_inj_conv(self):
        # TODO Update for Weights
        random.seed(1)
        self.inj_model = random_weight_inj(
            self.p, corrupt_conv=3, min_val=10000, max_val=20000
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_weight_zero_inj(self):
        # TODO Update for Weights
        random.seed(2)
        self.inj_model = zeroFunc_rand_weight(self.p)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

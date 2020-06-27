import pytest
import torch
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.errormodels import error_models as em

from .util_test import helper_setUp_CIFAR10_same


class TestWeightErrorModels:
    """
    Testing weight perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(0)
        #random.seed(0)


        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(self.BATCH_SIZE, self.WORKERS)
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


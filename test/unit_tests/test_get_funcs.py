import torch

from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10


class TestCoreGetFuncs:
    """
    Testing focuses on neuron perturbations on the CPU with a single batch element.
    """

    def setup_class(self):
        self.BATCH_SIZE = 1
        self.WORKERS = 1
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)

        self.dataiter = iter(self.dataset)
        self.model.eval()

        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_get_output_size(self):
        shape = [
            [1, 64, 8, 8],
            [1, 192, 4, 4],
            [1, 384, 2, 2],
            [1, 256, 2, 2],
            [1, 256, 2, 2],
        ]
        assert self.p.get_output_size() == shape

    def test_get_total_batches(self):
        assert self.p.get_total_batches() == self.BATCH_SIZE

    def test_get_total_conv(self):
        assert self.p.get_total_conv() == 5

    def test_get_fmap_num(self):
        assert self.p.get_fmaps_num(0) == 64
        assert self.p.get_fmaps_num(1) == 192
        assert self.p.get_fmaps_num(2) == 384
        assert self.p.get_fmaps_num(3) == 256
        assert self.p.get_fmaps_num(4) == 256

    def test_get_fmaps_H(self):
        assert self.p.get_fmaps_H(0) == 8
        assert self.p.get_fmaps_H(1) == 4
        assert self.p.get_fmaps_H(2) == 2
        assert self.p.get_fmaps_H(3) == 2
        assert self.p.get_fmaps_H(4) == 2

    def test_get_fmaps_W(self):
        assert self.p.get_fmaps_W(0) == 8
        assert self.p.get_fmaps_W(1) == 4
        assert self.p.get_fmaps_W(2) == 2
        assert self.p.get_fmaps_W(3) == 2
        assert self.p.get_fmaps_W(4) == 2

    def test_get_fmap_HW(self):
        assert self.p.get_fmap_HW(0) == (8, 8)
        assert self.p.get_fmap_HW(1) == (4, 4)
        assert self.p.get_fmap_HW(2) == (2, 2)
        assert self.p.get_fmap_HW(3) == (2, 2)
        assert self.p.get_fmap_HW(4) == (2, 2)

import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10_same


class TestCoreGetFuncs:
    """
    Testing focuses on neuron perturbations on the CPU with a single batch element.
    """

    def setup_class(self):
        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.LAYER_TYPES = [torch.nn.Conv2d]
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )

        self.dataiter = iter(self.dataset)
        self.model.eval()

        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            layer_types=self.LAYER_TYPES,
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
        assert self.p.get_total_batches() == 4

    def test_get_inj_layer_types(self):
        assert self.p.get_inj_layer_types() == [torch.nn.Conv2d]

    def test_get_layer_type(self):
        assert self.p.get_layer_type(3) == torch.nn.Conv2d

    def test_get_layer_dim(self):
        assert self.p.get_layer_dim(3) == 4

    def test_get_total_layers(self):
        assert self.p.get_total_layers() == 5

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

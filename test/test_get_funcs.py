import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import CIFAR10_set_up_custom


class TestCoreGetFuncs:
    """
    Testing focuses on neuron perturbations on
    the CPU with a single batch element.
    """

    def setup_class(self):
        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        self.use_gpu = False

        model, dataset = CIFAR10_set_up_custom(batch_size, workers)

        dataiter = iter(dataset)
        model.eval()

        torch.no_grad()
        self.images, self.labels = dataiter.next()

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            layer_types=layer_types,
            use_cuda=self.use_gpu,
        )

    def test_get_output_size(self):
        shape = [
            [1, 64, 8, 8],
            [1, 192, 4, 4],
            [1, 384, 2, 2],
            [1, 256, 2, 2],
            [1, 256, 2, 2],
            [1, 10],
        ]
        assert self.p.get_output_size() == shape

    def test_get_weights_size(self):
        shape = [
            [64, 3, 11, 11],
            [192, 64, 5, 5],
            [384, 192, 3, 3],
            [256, 384, 3, 3],
            [256, 256, 3, 3],
            [10, 256],
        ]

        for i in range(6):
            assert list(self.p.get_weights_size(i)) == shape[i]

    def test_get_total_batches(self):
        assert self.p.get_total_batches() == 4

    def test_get_inj_layer_types(self):
        assert self.p.get_inj_layer_types() == [torch.nn.Conv2d, torch.nn.Linear]

    def test_get_layer_type(self):
        assert self.p.get_layer_type(3) == torch.nn.Conv2d
        assert self.p.get_layer_type(5) == torch.nn.Linear

    def test_get_layer_shape(self):
        assert self.p.get_layer_shape(0) == [1, 64, 8, 8]
        assert self.p.get_layer_shape(1) == [1, 192, 4, 4]
        assert self.p.get_layer_shape(2) == [1, 384, 2, 2]
        assert self.p.get_layer_shape(3) == [1, 256, 2, 2]
        assert self.p.get_layer_shape(4) == [1, 256, 2, 2]
        assert self.p.get_layer_shape(5) == [1, 10]

    def test_get_layer_dim(self):
        assert self.p.get_layer_dim(3) == 4

    def test_get_total_layers(self):
        assert self.p.get_total_layers() == 6

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

    def test_print_func(self):
        output = self.p.print_pytorchfi_layer_summary()

        assert "PYTORCHFI INIT SUMMARY" in output
        assert "Shape of input into the model: (3 32 32 )" in output
        assert "Batch Size: 4" in output
        assert "Conv2d" in output
        assert "Linear" in output
        assert "CUDA Enabled: False" in output

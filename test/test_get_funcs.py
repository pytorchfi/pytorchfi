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
        layer_types = [torch.nn.Conv2d]
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
        ]
        if self.p.get_output_size() != shape:
            raise AssertionError

    def test_get_total_batches(self):
        if self.p.get_total_batches() != 4:
            raise AssertionError

    def test_get_inj_layer_types(self):
        if self.p.get_inj_layer_types() != [torch.nn.Conv2d]:
            raise AssertionError

    def test_get_layer_type(self):
        if self.p.get_layer_type(3) != torch.nn.Conv2d:
            raise AssertionError

    def test_get_layer_shape(self):
        if self.p.get_layer_shape(0) != [1, 64, 8, 8]:
            raise AssertionError
        if self.p.get_layer_shape(1) != [1, 192, 4, 4]:
            raise AssertionError
        if self.p.get_layer_shape(2) != [1, 384, 2, 2]:
            raise AssertionError
        if self.p.get_layer_shape(3) != [1, 256, 2, 2]:
            raise AssertionError
        if self.p.get_layer_shape(4) != [1, 256, 2, 2]:
            raise AssertionError

    def test_get_layer_dim(self):
        if self.p.get_layer_dim(3) != 4:
            raise AssertionError

    def test_get_total_layers(self):
        if self.p.get_total_layers() != 5:
            raise AssertionError

    def test_get_fmap_num(self):
        if self.p.get_fmaps_num(0) != 64:
            raise AssertionError
        if self.p.get_fmaps_num(1) != 192:
            raise AssertionError
        if self.p.get_fmaps_num(2) != 384:
            raise AssertionError
        if self.p.get_fmaps_num(3) != 256:
            raise AssertionError
        if self.p.get_fmaps_num(4) != 256:
            raise AssertionError

    def test_get_fmaps_H(self):
        if self.p.get_fmaps_H(0) != 8:
            raise AssertionError
        if self.p.get_fmaps_H(1) != 4:
            raise AssertionError
        if self.p.get_fmaps_H(2) != 2:
            raise AssertionError
        if self.p.get_fmaps_H(3) != 2:
            raise AssertionError
        if self.p.get_fmaps_H(4) != 2:
            raise AssertionError

    def test_get_fmaps_W(self):
        if self.p.get_fmaps_W(0) != 8:
            raise AssertionError
        if self.p.get_fmaps_W(1) != 4:
            raise AssertionError
        if self.p.get_fmaps_W(2) != 2:
            raise AssertionError
        if self.p.get_fmaps_W(3) != 2:
            raise AssertionError
        if self.p.get_fmaps_W(4) != 2:
            raise AssertionError

    def test_get_fmap_HW(self):
        if self.p.get_fmap_HW(0) != (8, 8):
            raise AssertionError
        if self.p.get_fmap_HW(1) != (4, 4):
            raise AssertionError
        if self.p.get_fmap_HW(2) != (2, 2):
            raise AssertionError
        if self.p.get_fmap_HW(3) != (2, 2):
            raise AssertionError
        if self.p.get_fmap_HW(4) != (2, 2):
            raise AssertionError

    def test_print_func(self):
        outputString = self.p.print_pytorchfi_layer_summary()

        if "PYTORCHFI INIT SUMMARY" not in outputString:
            raise AssertionError
        if "Shape of input into the model: (3 32 32 )" not in outputString:
            raise AssertionError
        if "Batch Size: 4" not in outputString:
            raise AssertionError
        if "Conv2d" not in outputString:
            raise AssertionError
        if "CUDA Enabled: False" not in outputString:
            raise AssertionError
        if (
            "================================================================"
            not in outputString
        ):
            raise AssertionError

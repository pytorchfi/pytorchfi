import pytest
import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import CIFAR10_set_up_custom


class TestSingleNeuron:
    """Testing focuses on neuron perturbations with a single batch element."""

    def setup_class(self):
        self.batch_size = 1
        workers = 1
        self.channels = 3
        self.img_size = 32

        self.model, self.dataset = CIFAR10_set_up_custom(self.batch_size, workers)
        self.dataiter = iter(self.dataset)
        self.model.eval()

        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        self.output = self.model(self.images)

    def test_init_cpu(self):
        # TODO: More comprehensive test
        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        )

    def test_orig_model_cpu(self):
        p = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        )

        if p.get_original_model() is not self.model:
            raise AssertionError

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_init_gpu(self):
        # TODO: More comprehensive test
        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_orig_model_gpu(self):
        p = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        )

        if p.get_original_model() is not self.model:
            raise AssertionError


class TestDtypes:
    """Testing focuses on using different model datatypes"""

    def setup_class(self):
        self.batch_size = 1
        workers = 1
        self.channels = 3
        self.img_size = 32

        self.model, dataset = CIFAR10_set_up_custom(self.batch_size, workers)
        dataiter = iter(dataset)
        self.images, self.labels = dataiter.next()

    def test_fp32_cpu(self):
        # TODO: More comprehensive test

        self.model.to("cpu")
        self.model.eval()

        torch.no_grad()
        self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_fp32_gpu(self):
        # TODO: More comprehensive test

        self.model.to("cuda")
        self.model.eval()

        torch.no_grad()
        self.images = self.images.cuda()
        self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_fp16_gpu(self):
        # TODO: More comprehensive test

        self.model.to("cuda")
        self.model.half()
        self.model.eval()

        torch.no_grad()
        self.images = self.images.cuda()
        self.output = self.model(self.images.half())

        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        )

    @pytest.mark.skip(reason="Experimental in PyTorch")
    def test_INT8_cpu(self):
        # TODO: More comprehensive test

        self.model.to("cpu")
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()

        with torch.no_grad():
            self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        )

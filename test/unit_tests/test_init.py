import pytest
import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10


class TestNeuronCPUSingle:
    """
    Testing focuses on neuron perturbations on the CPU with a single batch element.
    """

    def setup_class(self):
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()

        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)
        if self.USE_GPU:
            self.model.cuda()
        self.dataiter = iter(self.dataset)
        self.model.eval()

        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

    def test_init_cpu(self):
        """
        #TODO: More comprehensive test
        """
        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_orig_model_cpu(self):
        p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

        self.faulty_model = p.get_original_model()
        assert self.faulty_model is self.model


class TestNeuronGPUSingle:
    """
    Testing focuses on neuron perturbations on the GPU with a single batch element.
    """

    def setup_class(self):
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()

        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)
        if self.USE_GPU:
            self.model.cuda()
        self.dataiter = iter(self.dataset)
        self.model.eval()

        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_init_gpu(self):
        """
        TODO: More comprehensive test
        """
        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_orig_model_gpu(self):
        p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

        self.faulty_model = p.get_original_model()
        assert self.faulty_model is self.model


class TestDtypes:
    """
    Testing focuses on using different model datatypes
    """

    def setup_class(self):
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()

        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_fp32_gpu(self):
        """
        TODO: More comprehensive test
        """
        if self.USE_GPU:
            self.model.cuda()
        self.model.eval()

        torch.no_grad()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_fp32_cpu(self):
        """
        TODO: More comprehensive test
        """
        if self.USE_GPU:
            self.model.cuda()
        self.model.eval()

        torch.no_grad()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_fp16_gpu(self):
        """
        TODO: More comprehensive test
        """
        if self.USE_GPU:
            self.model.cuda()

        self.model.half()
        self.model.eval()

        torch.no_grad()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images.half())

        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    @pytest.mark.skip(reason="Currently failing")
    def test_INT8_cpu(self):
        """
        TODO: More comprehensive test
        """
        if self.USE_GPU:
            self.model.cuda()

        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

        self.model.eval()

        torch.no_grad()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

        pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

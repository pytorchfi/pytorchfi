import pytest
import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10_same


class TestNeuronFIgpu:
    """Testing focuses on neuron perturbations on GPU with batch = 1."""

    def setup_class(self):
        torch.manual_seed(0)

        self.BATCH_SIZE = 1
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = True

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.model.cuda()

        self.images, self.labels = self.dataiter.next()
        self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuronFI_singleElement(self):
        batch_i = [0]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]

        inj_value_i = [10000.0]

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1.eq(self.output)):
            raise AssertionError

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0],
        )

        self.inj_model.eval()
        with torch.no_grad():
            uncorrupted_output = self.inj_model(self.images)

        if not torch.all(uncorrupted_output.eq(self.output)):
            raise AssertionError

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i * 2,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_2 = self.inj_model(self.images)

        if torch.all(corrupted_output_2.eq(self.output)):
            raise AssertionError
        if not torch.all(corrupted_output_2.eq(corrupted_output_2)):
            raise AssertionError


class TestNeuronFIcpu:
    """Testing focuses on neuron perturbations on CPU with batch = 1."""

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 1
        workers = 1
        channels = 3
        img_size = 32

        model, dataset = helper_setUp_CIFAR10_same(batch_size, workers)
        dataiter = iter(dataset)
        self.images, _ = dataiter.next()

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=False,
        )

        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

    def test_neuronFI_singleElement(self):
        batch_i = [0]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]

        inj_value_i = [10000.0]

        uncorrupted_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0],
        )

        uncorrupted_model.eval()
        with torch.no_grad():
            uncorrupted_output = uncorrupted_model(self.images)

        if not torch.all(uncorrupted_output.eq(self.golden_output)):
            raise AssertionError

        corrupt_model_1 = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model_1.eval()
        with torch.no_grad():
            corrupt_output_1 = corrupt_model_1(self.images)

        if torch.all(corrupt_output_1.eq(self.golden_output)):
            raise AssertionError

        corrupt_model_2 = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i * 2,
        )

        corrupt_model_2.eval()
        with torch.no_grad():
            corrupt_output_2 = corrupt_model_2(self.images)

        if torch.all(corrupt_output_2.eq(self.golden_output)):
            raise AssertionError


class TestNeuronFIgpuBatch:
    """Testing focuses on neuron perturbations on GPU with batch = N."""

    def setup_class(self):
        torch.manual_seed(0)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
        self.img_size = 32
        self.USE_GPU = True

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.model.cuda()

        self.images, self.labels = self.dataiter.next()
        self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuronFI_batch_1(self):

        batch_i = [2]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]

        inj_value_i = [10000.0]

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if not torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuronFI_batch_2(self):

        batch_i = [0, 2, 3]
        layer_i = [1, 2, 4]
        c_i = [3, 1, 1]
        h_i = [1, 0, 1]
        w_i = [0, 1, 0]

        inj_value_i = [10000.0, 10000.0, 10000.0]

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError


class TestNeuronFIcpuBatch:
    """Testing focuses on neuron perturbations on cpu with batch = N."""

    def setup_class(self):
        torch.manual_seed(0)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.channels = 3
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
            self.BATCH_SIZE,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=self.USE_GPU,
        )

    def test_neuronFI_batch_1(self):
        batch_i = [2]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]

        inj_value_i = [10000.0]

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if not torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if not torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

    def test_neuronFI_batch_2(self):
        batch_i = [0, 2, 3]
        layer_i = [1, 2, 4]
        c_i = [3, 1, 1]
        h_i = [1, 0, 1]
        w_i = [0, 1, 0]

        inj_value_i = [10000.0, 10000.0, 10000.0]

        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        if torch.all(corrupted_output_1[0].eq(self.output[0])):
            raise AssertionError
        if not torch.all(corrupted_output_1[1].eq(self.output[1])):
            raise AssertionError
        if torch.all(corrupted_output_1[2].eq(self.output[2])):
            raise AssertionError
        if torch.all(corrupted_output_1[3].eq(self.output[3])):
            raise AssertionError

import pytest
import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10_same


class TestNeuronFIgpu:
    """Testing focuses on neuron perturbations on GPU with batch = 1."""

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 1
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = True

        model, dataset = helper_setUp_CIFAR10_same(batch_size, workers)
        dataiter = iter(dataset)
        self.images, self.labels = dataiter.next()
        self.images = self.images.cuda()

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=use_gpu,
        )

        model.cuda()
        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

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

        uncorrupt_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0],
        )

        uncorrupt_model.eval()
        with torch.no_grad():
            uncorrupted_output = uncorrupt_model(self.images)

        if not torch.all(uncorrupted_output.eq(self.golden_output)):
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
            corrupted_output_2 = corrupt_model_2(self.images)

        if torch.all(corrupted_output_2.eq(self.golden_output)):
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
        use_gpu = False

        model, dataset = helper_setUp_CIFAR10_same(batch_size, workers)
        dataiter = iter(dataset)
        self.images, self.labels = dataiter.next()

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=use_gpu,
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

        uncorrupt_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0],
        )

        uncorrupt_model.eval()
        with torch.no_grad():
            uncorrupted_output = uncorrupt_model(self.images)

        if not torch.all(uncorrupted_output.eq(self.golden_output)):
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

        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = True

        model, self.dataset = helper_setUp_CIFAR10_same(batch_size, workers)
        dataiter = iter(self.dataset)
        self.images, self.labels = dataiter.next()
        self.images = self.images.cuda()

        model.cuda()
        model.eval()
        with torch.no_grad():
            self.golden_output = model(self.images)

        self.p = pfi_core(
            model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            use_cuda=use_gpu,
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
            corrupt_output_1 = self.inj_model(self.images)

        if not torch.all(corrupt_output_1[0].eq(self.golden_output[0])):
            raise AssertionError
        if not torch.all(corrupt_output_1[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output_1[2].eq(self.golden_output[2])):
            raise AssertionError
        if not torch.all(corrupt_output_1[3].eq(self.golden_output[3])):
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
            corrupt_output_1 = self.inj_model(self.images)

        if torch.all(corrupt_output_1[0].eq(self.golden_output[0])):
            raise AssertionError
        if not torch.all(corrupt_output_1[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output_1[2].eq(self.golden_output[2])):
            raise AssertionError
        if torch.all(corrupt_output_1[3].eq(self.golden_output[3])):
            raise AssertionError


class TestNeuronFIcpuBatch:
    """Testing focuses on neuron perturbations on cpu with batch = N."""

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 4
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = False

        model, self.dataset = helper_setUp_CIFAR10_same(batch_size, workers)
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

    def test_neuronFI_batch_1(self):
        batch_i = [2]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]

        inj_value_i = [10000.0]

        corrupt_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output_1 = corrupt_model(self.images)

        if not torch.all(corrupt_output_1[0].eq(self.golden_output[0])):
            raise AssertionError
        if not torch.all(corrupt_output_1[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output_1[2].eq(self.golden_output[2])):
            raise AssertionError
        if not torch.all(corrupt_output_1[3].eq(self.golden_output[3])):
            raise AssertionError

    def test_neuronFI_batch_2(self):
        batch_i = [0, 2, 3]
        layer_i = [1, 2, 4]
        c_i = [3, 1, 1]
        h_i = [1, 0, 1]
        w_i = [0, 1, 0]

        inj_value_i = [10000.0, 10000.0, 10000.0]

        corrupt_model = self.p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output_1 = corrupt_model(self.images)

        if torch.all(corrupt_output_1[0].eq(self.golden_output[0])):
            raise AssertionError
        if not torch.all(corrupt_output_1[1].eq(self.golden_output[1])):
            raise AssertionError
        if torch.all(corrupt_output_1[2].eq(self.golden_output[2])):
            raise AssertionError
        if torch.all(corrupt_output_1[3].eq(self.golden_output[3])):
            raise AssertionError

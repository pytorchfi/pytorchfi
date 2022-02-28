import pytest
import torch

from pytorchfi.core import FaultInjection as pfi_core

from .util_test import CIFAR10_set_up_custom


class TestNeuronFi:
    """Testing focuses on neuron perturbations on with batch = 1."""

    def setup_class(self):
        torch.manual_seed(0)

        self.batch_size = 1
        self.workers = 1
        self.channels = 3
        self.img_size = 32

        self.model, dataset = CIFAR10_set_up_custom(self.batch_size, self.workers)
        dataiter = iter(dataset)
        self.images, self.labels = dataiter.next()

    def test_neuron_single_fi_cpu(self):
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images)

        batch_i = [0]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]
        inj_value_i = [10000.0]

        p = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        )

        corrupt_model_1 = p.declare_neuron_fi(
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

        assert not torch.all(corrupt_output_1.eq(golden_output))

        uncorrupt_model = p.declare_neuron_fi(
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

        assert torch.all(uncorrupted_output.eq(golden_output))

        corrupt_model_2 = p.declare_neuron_fi(
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

        assert not torch.all(corrupt_output_2.eq(golden_output))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuron_single_fi_gpu(self):
        self.images_gpu = self.images.cuda()
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images_gpu)

        batch_i = [0]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]
        inj_value_i = [10000.0]

        p = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        )

        corrupt_model_1 = p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model_1.eval()
        with torch.no_grad():
            corrupt_output_1 = corrupt_model_1(self.images_gpu)

        assert not torch.all(corrupt_output_1.eq(golden_output))

        uncorrupt_model = p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0],
        )

        uncorrupt_model.eval()
        with torch.no_grad():
            uncorrupted_output = uncorrupt_model(self.images_gpu)

        assert torch.all(uncorrupted_output.eq(golden_output))

        corrupt_model_2 = p.declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i * 2,
        )

        corrupt_model_2.eval()
        with torch.no_grad():
            corrupted_output_2 = corrupt_model_2(self.images_gpu)

        assert not torch.all(corrupted_output_2.eq(golden_output))


class TestNeuronFiBatch:
    """Testing focuses on neuron perturbations with batch = N."""

    def setup_class(self):
        torch.manual_seed(0)

        self.batch_size = 4
        self.workers = 1
        self.channels = 3
        self.img_size = 32
        self.use_gpu = True

        self.model, self.dataset = CIFAR10_set_up_custom(self.batch_size, self.workers)
        dataiter = iter(self.dataset)
        self.images, self.labels = dataiter.next()

    def test_neuron_fi_batch_cpu_1(self):
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images)

        batch_i = [2]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]
        inj_value_i = [10000.0]

        corrupt_model = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        ).declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert torch.all(corrupt_output[0].eq(golden_output[0]))
        assert torch.all(corrupt_output[1].eq(golden_output[1]))
        assert not torch.all(corrupt_output[2].eq(golden_output[2]))
        assert torch.all(corrupt_output[3].eq(golden_output[3]))

    def test_neuron_fi_batch_cpu_2(self):
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images)

        batch_i = [0, 2, 3]
        layer_i = [1, 2, 4]
        c_i = [3, 1, 1]
        h_i = [1, 0, 1]
        w_i = [0, 1, 0]
        inj_value_i = [10000.0, 10000.0, 10000.0]

        corrupt_model = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=False,
        ).declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output[0].eq(golden_output[0]))
        assert torch.all(corrupt_output[1].eq(golden_output[1]))
        assert not torch.all(corrupt_output[2].eq(golden_output[2]))
        assert not torch.all(corrupt_output[3].eq(golden_output[3]))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuron_fi_batch_gpu_1(self):
        self.images_gpu = self.images.cuda()
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images_gpu)

        batch_i = [2]
        layer_i = [4]
        c_i = [0]
        h_i = [1]
        w_i = [1]
        inj_value_i = [10000.0]

        corrupt_model = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        ).declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images_gpu)

        assert torch.all(corrupt_output[0].eq(golden_output[0]))
        assert torch.all(corrupt_output[1].eq(golden_output[1]))
        assert not torch.all(corrupt_output[2].eq(golden_output[2]))
        assert torch.all(corrupt_output[3].eq(golden_output[3]))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not supported on this machine"
    )
    def test_neuron_fi_batch_gpu_2(self):
        self.images_gpu = self.images.cuda()
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            golden_output = self.model(self.images_gpu)

        batch_i = [0, 2, 3]
        layer_i = [1, 2, 4]
        c_i = [3, 1, 1]
        h_i = [1, 0, 1]
        w_i = [0, 1, 0]

        inj_value_i = [10000.0, 10000.0, 10000.0]

        corrupt_model = pfi_core(
            self.model,
            self.batch_size,
            input_shape=[self.channels, self.img_size, self.img_size],
            use_cuda=True,
        ).declare_neuron_fi(
            batch=batch_i,
            layer_num=layer_i,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i,
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images_gpu)

        assert not torch.all(corrupt_output[0].eq(golden_output[0]))
        assert torch.all(corrupt_output[1].eq(golden_output[1]))
        assert not torch.all(corrupt_output[2].eq(golden_output[2]))
        assert not torch.all(corrupt_output[3].eq(golden_output[3]))

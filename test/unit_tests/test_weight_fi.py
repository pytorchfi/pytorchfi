import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import helper_setUp_CIFAR10_same


class TestWeightFIcpu:
    """
    Testing focuses on weight perturbations.
    """

    def setup_class(self):
        torch.manual_seed(0)

        self.BATCH_SIZE = 1
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

    def test_neuronFI_singleElement(self):
        layer_i = 1
        k = 15
        c_i = 20
        h_i = 2
        w_i = 3

        inj_value_i = 10000.0

        self.inj_model = self.p.declare_weight_fi(
            layer_num=layer_i, k=k, dim1=c_i, dim2=h_i, dim3=w_i, value=inj_value_i
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

        self.inj_model = self.p.declare_weight_fi(
            layer_num=layer_i,
            k=k,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=0.01388985849916935,
        )

        self.inj_model.eval()
        with torch.no_grad():
            uncorrupted_output = self.inj_model(self.images)

        assert torch.all(uncorrupted_output.eq(self.output))

        self.inj_model = self.p.declare_weight_fi(
            layer_num=layer_i,
            k=k,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=inj_value_i * 2,
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_2 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_2.eq(self.output))
        assert torch.all(corrupted_output_2.eq(corrupted_output_2))

    def test_neuronFI_singleElement_noErr(self):
        layer_i = 4
        k = 153
        c_i = 254
        h_i = 0
        w_i = 0

        inj_value_i = 10000.0

        self.inj_model = self.p.declare_weight_fi(
            layer_num=layer_i, k=k, dim1=c_i, dim2=h_i, dim3=w_i, value=inj_value_i
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert torch.all(corrupted_output_1.eq(self.output))

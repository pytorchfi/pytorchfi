import torch
import torchvision.models as models
from pytorchfi.core import fault_injection
import pytest


class TestLayers:
    """
    Testing PyTorchFI.Core example client.
    """

    def setup_class(self):
        torch.manual_seed(5)

        self.Cin = 3
        self.Hin = 224
        self.Win = 224
        self.BATCH_SIZE = 4

        self.IMAGE = torch.rand((self.BATCH_SIZE, self.Cin, self.Hin, self.Win))

        self.USE_GPU = False

        self.softmax = torch.nn.Softmax(dim=1)

        self.model = models.alexnet(pretrained=True)
        self.model.eval()

        # Error free inference to gather golden value
        self.output = self.model(self.IMAGE)
        self.golden_softmax = self.softmax(self.output)
        self.golden_label = list(torch.argmax(self.golden_softmax, dim=1))[0].item()

    def test_golden_inference(self):
        assert self.golden_label == 556

    def test_single_conv_neuron(self):

        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Conv2d],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = ([0], [3], [4], [2], [4], [10000])
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        if inj_label != 578:
            raise AssertionError

    def test_single_linear_layer(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        if p.get_total_layers() != 3:
            raise AssertionError
        if p.get_layer_dim(2) != 2:
            raise AssertionError
        if p.get_layer_type(2) != torch.nn.Linear:
            raise AssertionError

    def test_inj_all_layers(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=["all"],
            use_cuda=self.USE_GPU,
        )

        if p.get_total_layers() != 21:
            raise AssertionError
        if p.get_layer_dim(2) != 4:
            raise AssertionError
        if p.get_layer_type(2) != torch.nn.MaxPool2d:
            raise AssertionError
        if p.get_layer_type(20) != torch.nn.Linear:
            raise AssertionError

    def test_inj_all_layers_injections(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=["all"],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = (
            [0,1,2,3], [0,1,2,20], [4,4,0,4], [2, 2, 2, None], [4, 2, 2, None],
            [10000, 10000, 10000, 1000])

        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label_0 = list(torch.argmax(inj_softmax, dim=1))[0].item()
        inj_label_1 = list(torch.argmax(inj_softmax, dim=1))[1].item()
        inj_label_2 = list(torch.argmax(inj_softmax, dim=1))[2].item()
        inj_label_3 = list(torch.argmax(inj_softmax, dim=1))[3].item()

        if inj_label_0 != 722:
            raise AssertionError
        if inj_label_1 != 723:
            raise AssertionError
        if inj_label_2 != 968:
            raise AssertionError
        if inj_label_3 != 4:
            raise AssertionError

    def test_single_linear_neuron_inj(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = (0, 2, 888, None, None, 10000)
        inj = p.declare_neuron_fi(
            batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        if inj_label != 888:
            raise AssertionError

    def test_combo_layers(self):
        p = fault_injection(
            self.model,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Conv2d, torch.nn.Linear],
            # layer_types=[torch.nn.Conv2d],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = (
            [0, 1],
            [1, 7],
            [5, 888],
            [5, None],
            [3, None],
            [20000, 10000],
        )
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label_1 = list(torch.argmax(inj_softmax, dim=1))[0].item()
        inj_label_2 = list(torch.argmax(inj_softmax, dim=1))[1].item()

        if p.get_total_layers() != 8:
            raise AssertionError
        if inj_label_1 != 695:
            raise AssertionError
        if inj_label_2 != 888:
            raise AssertionError

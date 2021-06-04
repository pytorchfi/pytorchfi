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

        self.H = 224
        self.W = 224
        self.BATCH_SIZE = 4

        self.IMAGE = torch.rand((self.BATCH_SIZE, 3, self.H, self.W))

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
            self.H,
            self.W,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Conv2d],
            use_cuda=self.USE_GPU,
        )


        (b, layer, C, H, W, err_val) = (0, 3, 4, 2, 4, 10000)
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        assert inj_label == 578

    @pytest.mark.skip(reason="Under development")
    def test_single_linear_neuron(self):
        p = fault_injection(
            self.model,
            self.H,
            self.W,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = (0, 3, 4, 2, 4, 10000)
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        assert inj_label == 578

    @pytest.mark.skip(reason="Under development")
    def test_combo_layers(self):
        p = fault_injection(
            self.model,
            self.H,
            self.W,
            self.BATCH_SIZE,
            layer_types=[torch.nn.Conv2d, torch.nn.Linear],
            use_cuda=self.USE_GPU,
        )

        (b, layer, C, H, W, err_val) = (
            [0, 0],
            [1, 3],
            [5, 4],
            [5, 2],
            [3, 4],
            [20000, 10000],
        )
        inj = p.declare_neuron_fi(
            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
        )
        inj_output = inj(self.IMAGE)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        assert inj_label == 843

#    def test_multiple_conv_neuron(self):
#
#        p = fault_injection(
#            self.model,
#            self.H,
#            self.W,
#            self.BATCH_SIZE,
#            layer_types=[torch.nn.Conv2d],
#            use_cuda=self.USE_GPU,
#        )
#
#        (b, layer, C, H, W, err_val) = (
#            [0, 0],
#            [1, 3],
#            [5, 4],
#            [5, 2],
#            [3, 4],
#            [20000, 10000],
#        )
#        inj = p.declare_neuron_fi(
#            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
#        )
#        inj_output = inj(self.IMAGE)
#        inj_softmax = self.softmax(inj_output)
#        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()
#
#        assert inj_label == 843
#
#    def test_multiple_linear_neuron(self):
#        p = fault_injection(
#            self.model,
#            self.H,
#            self.W,
#            self.BATCH_SIZE,
#            layer_types=[torch.nn.Linear],
#            use_cuda=self.USE_GPU,
#        )
#
#        (b, layer, C, H, W, err_val) = (
#            [0, 0],
#            [1, 3],
#            [5, 4],
#            [5, 2],
#            [3, 4],
#            [20000, 10000],
#        )
#        inj = p.declare_neuron_fi(
#            batch=b, layer_num=layer, c=C, h=H, w=W, value=err_val
#        )
#        inj_output = inj(self.IMAGE)
#        inj_softmax = self.softmax(inj_output)
#        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()
#
#        assert inj_label == 843
#
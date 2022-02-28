import torch
import torchvision.models as models

from pytorchfi.core import fault_injection


class TestCoreExampleClient:
    """Testing PyTorchFI.Core example client."""

    def setup_class(self):
        torch.manual_seed(5)

        c = 3
        h = 224
        w = 224
        batch_size = 4

        self.image = torch.rand((batch_size, c, h, w))
        self.softmax = torch.nn.Softmax(dim=1)

        self.model = models.alexnet(pretrained=True)
        self.model.eval()

        # Error free inference to gather golden value
        self.output = self.model(self.image)
        self.golden_softmax = self.softmax(self.output)
        self.golden_label = list(torch.argmax(self.golden_softmax, dim=1))[0].item()

        self.p = fault_injection(
            self.model,
            batch_size,
            input_shape=[c, h, w],
            use_cuda=False,
        )

    def test_golden_inference(self):
        assert self.golden_label == 556

    def test_single_specified_neuron(self):
        (b, layer, C, H, W, err_val) = ([0], [3], [4], [2], [4], [10000])
        inj = self.p.declare_neuron_fi(
            batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val
        )
        inj_output = inj(self.image)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        assert inj_label == 578

    def test_multiple_specified_neuron(self):
        (b, layer, C, H, W, err_val) = (
            [0, 0],
            [1, 3],
            [5, 4],
            [5, 2],
            [3, 4],
            [20000, 10000],
        )
        inj = self.p.declare_neuron_fi(
            batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val
        )
        inj_output = inj(self.image)
        inj_softmax = self.softmax(inj_output)
        inj_label = list(torch.argmax(inj_softmax, dim=1))[0].item()

        assert inj_label == 843

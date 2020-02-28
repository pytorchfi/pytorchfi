import torch
import torchvision.models as models
import pickle

from pytorchfi import core

# deterministic rand input
torch.random.manual_seed(5)
image = torch.rand((10, 3, 224, 224))
h = 224
w = 224
batch_size = 1

torch.no_grad()

# load model
model = models.resnet50(pretrained=True)

# create corrupt model
fi_model = core.fault_injection(model, h, w, batch_size)

# TODO check if GPU works
# if CUDA_en():
#    model = model.cuda()
#    label = labels.cuda()
#    image = image.cuda()
# if half_precision:
#    model = model.half()
#    image = image.half()

model.eval()

golden_output = model(image)

with open("golden_output", "rb") as f:
    golden = pickle.load(f)
with open("error_output_weight", "rb") as f:
    error_weight = pickle.load(f)
with open("error_output_neuron", "rb") as f:
    error_neuron = pickle.load(f)


# General tests
assert torch.equal(golden_output, golden), "sanity check model working"

fi_model.fi_reset()
assert (
    fi_model.get_original_model() == model
), "original model not stored or overwritten"

# Basic injection tests
fi_model.declare_weight_fi(value=100000, index=(0), layer=1)
error_output_weight = fi_model.CORRUPTED_MODEL(image)
assert not torch.equal(
    golden_output, error_output_weight
), "weight error output identical to golden"
assert torch.equal(error_output_weight, error_weight), "weight injection not correct"

fi_model.fi_reset()
# assert fi_model.CORRUPTED_MODEL == None, "fi_reset not resetting model"

fi_model.declare_neuron_fi(value=100000, conv_num=0, batch=0, c=0, h=5, w=5)
error_output_neuron = fi_model.CORRUPTED_MODEL(image)
assert not torch.equal(
    golden_output, error_output_neuron
), "neuron error output identical to golden"
assert torch.equal(error_output_neuron, error_neuron), "neuron injection not correct"

fi_model.fi_reset()
# assert fi_model.CORRUPTED_MODEL == None, "fi_reset not resetting model"

# advanced injection tests

print("All core tests passed")

import torch
#import pytorchfi.core as core
import core
import torchvision.models as models
import pickle

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
fi_model = core.core(model, h, w, batch_size)

# TODO check if GPU works
#if CUDA_en():
#    model = model.cuda()
#    label = labels.cuda()
#    image = image.cuda()
#if half_precision:
#    model = model.half()
#    image = image.half()

model.eval()

# General tests
golden_output = model(image)
with open("golden_output", "ab") as f:
    pickle.dump(golden_output, f)

fi_model.fi_reset()
assert fi_model.CORRUPTED_MODEL == None, "fi_reset not resetting model"

assert fi_model.get_original_model() == model, "original model not stored or overwritten"

# Basic injection tests
fi_model.declare_weight_fi(value=100000, index=(0), layer=1)
error_output = fi_model.CORRUPTED_MODEL(image)
assert not torch.equal(golden_output, error_output), "weight error output identical to golden"
with open("error_output_weight", "ab") as f:
    pickle.dump(error_output, f)

fi_model.fi_reset()
#assert fi_model.CORRUPTED_MODEL == None, "fi_reset not resetting model"

fi_model.declare_neuron_fi(value=100000, conv_num=0, batch=0, c=0, h=5, w=5)
error_output = fi_model.CORRUPTED_MODEL(image)
assert not torch.equal(golden_output, error_output), "neuron error output identical to golden"
with open("error_output_neuron", "ab") as f:
    pickle.dump(error_output, f)

fi_model.fi_reset()
#assert fi_model.CORRUPTED_MODEL == None, "fi_reset not resetting model"

# advanced injection tests

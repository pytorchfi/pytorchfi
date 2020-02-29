import torch
import torchvision.models as models
import pickle

from pytorchfi import core
from neuron_unit_tests import *
from weight_unit_tests import *

torch.no_grad()

# deterministic rand input
torch.random.manual_seed(5)

# Variables
h = 224
w = 224
c = 3
batch_size = 1

# load model
image = torch.rand(batch_size, c, h, w)
model = models.alexnet(pretrained=True)
model.eval()

# golden inference
golden_output = model(image)

# create corrupt model
fi_model = core.fault_injection(model, h, w, batch_size)

###########################
###### General tests ######
###########################
fi_model.fi_reset()
assert (fi_model.get_original_model() == model), "Original model stored incorrectly."

# TODO check if GPU works
# if CUDA_en():
#    model = model.cuda()
#    label = labels.cuda()
#    image = image.cuda()
# if half_precision:
#    model = model.half()
#    image = image.half()


###########################
##### neuron injection ####
###########################
neuron_inj(fi_model, image, golden_output)

# TODO check injection occured at proper place


###########################
##### weight injection ####
###########################
weight_inj(fi_model, image, golden_output)

# TODO check injection occured at proper place


###########################
###### Advanced Tests #####
###########################

print("All core tests passed")

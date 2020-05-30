#========================================#
# Commonly used functions during testing
#========================================#
import torch, os
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# helper function to set up the environment. Returns a model and dataset
def helper_setUp(batchsize, workers, dataset_path):

    # Dataset prep
    valdir = os.path.join(dataset_path + '/imagenet/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size = batchsize,
        shuffle=False,
        num_workers=workers,
        )

    # Model prep
    model = models.alexnet(pretrained=True)

    return model, val_loader



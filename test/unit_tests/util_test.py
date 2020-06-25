# ========================================#
# Commonly used functions during testing
# ========================================#
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


#####################################################################################
#################                CIFAR ALEXNET                              #########
#####################################################################################
"""AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


#####################################################################################


# helper function to set up the environment. Returns a model and dataset
def helper_setUp_IMAGENET(batchsize, workers, dataset_path):

    # Dataset prep
    valdir = os.path.join(dataset_path + "/imagenet/", "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batchsize,
        shuffle=False,
        num_workers=workers,
    )

    # Model prep
    model = models.alexnet(pretrained=True)

    return model, val_loader


def helper_setUp_CIFAR10(batchsize, workers):

    # Dataset prep
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False, num_workers=workers
    )

    # Model prep
    model = alexnet(num_classes=10)
    # ckpt_name = "model_best_pth.tar"
    # checkpoint = torch.load(ckpt_name)
    # model.load_state_dict(checkpoint)

    return model, val_loader

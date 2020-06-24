#=================================#
# PyTorchFI Unit Tests
#=================================#

import unittest
import torch, os
from .util_test import *

from pytorchfi.core import fault_injection as pfi_core


class TestNeuronCPUSingle(unittest.TestCase):
    """
    Testing focuses on *neuron* perturbations on the *CPU* with a *single* batch element.
    """

    def setUp(self):
        # parameters
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.DATASETS = os.environ['ML_DATASETS']
        self.img_size = 32
        self.USE_GPU = False

        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS, self.DATASETS)
        if self.USE_GPU: self.model.cuda()
        self.dataiter = iter(self.dataset)
        self.model.eval()

        # golden output
        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True: self.images = self.images.cuda()
        self.output = self.model(self.images)

    def test_init(self):
        """
        Test PytorchFI init()
        """
        p = pfi_core(self.model, self.img_size, self.img_size, self.BATCH_SIZE, use_cuda=self.USE_GPU, FP16 = False)

        self.assertTrue(True)

    def test_orig_model(self):
        """
        Test PytorchFI get_original_model()
        """
        p = pfi_core(self.model, self.img_size, self.img_size, self.BATCH_SIZE, use_cuda=self.USE_GPU, FP16 = False)
        self.faulty_model = p.get_original_model()

        self.assertTrue(self.faulty_model is self.model)


    def test2(self):
        self.assertTrue(True)


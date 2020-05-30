#==========================p======#
# PyTorchFI Testing Environment
# Use:
#   $python test.py
#=================================#

import unittest
import torch, os
from .util_test import *

import pytorchfi.core as pfi_core


class TestNeuronSingle(unittest.TestCase):

    def setUp(self):
        # parameters
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.DATASETS = os.environ['ML_DATASETS']
        self.USE_GPU = True

        # get model and dataset
        self.model, self.dataset = helper_setUp(self.BATCH_SIZE, self.WORKERS, self.DATASETS)
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
        self.img_size = 224
        pfi_core.fault_injection(self.model, self.img_size, self.img_size, self.BATCH_SIZE, use_cuda=self.USE_GPU, FP16 = False)

        self.assertTrue(True)

    def test2(self):
        self.assertTrue(True)


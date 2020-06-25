# =================================#
# PyTorchFI Unit Tests
# =================================#

import unittest

import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import *


class TestCoreGetFuncs(unittest.TestCase):
    """
    Testing focuses on neuron perturbations on the CPU with a single batch element.
    """

    def setUp(self):
        # parameters
        self.BATCH_SIZE = 1024
        self.WORKERS = 64
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()

        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(self.BATCH_SIZE, self.WORKERS)
        if self.USE_GPU:
            self.model.cuda()
        self.dataiter = iter(self.dataset)
        self.model.eval()

        # golden output
        torch.no_grad()
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()
        self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_get_output_size(self):
        """
        Test PytorchFI get_output_size() function
        """
        shape = [
            [1, 64, 8, 8],
            [1, 192, 4, 4],
            [1, 384, 2, 2],
            [1, 256, 2, 2],
            [1, 256, 2, 2],
        ]
        self.assertTrue(self.p.get_output_size() == shape)  # for AlexNet

    def test_get_total_batches(self):
        """
        Test PytorchFI get_total_batches() function
        """
        self.assertTrue(self.p.get_total_batches() == self.BATCH_SIZE)

    def test_get_total_conv(self):
        """
        Test PytorchFI get_total_conv() function
        """
        self.assertTrue(self.p.get_total_conv() == 5)  # for AlexNet

    def test_get_fmap_num(self):
        """
        Test PytorchFI get_fmap_num(...) function
        """
        allPass = (
            self.p.get_fmaps_num(0) == 64
            and self.p.get_fmaps_num(1) == 192
            and self.p.get_fmaps_num(2) == 384
            and self.p.get_fmaps_num(3) == 256
            and self.p.get_fmaps_num(4) == 256
        )

        self.assertTrue(allPass)  # for AlexNet

    def test_get_fmaps_H(self):
        """
        Test PytorchFI get_fmaps_H(...) function
        """
        allPass = (
            self.p.get_fmaps_H(0) == 8
            and self.p.get_fmaps_H(1) == 4
            and self.p.get_fmaps_H(2) == 2
            and self.p.get_fmaps_H(3) == 2
            and self.p.get_fmaps_H(4) == 2
        )

        self.assertTrue(allPass)  # for AlexNet on CIFAR

    def test_get_fmaps_W(self):
        """
        Test PytorchFI get_fmaps_W(...) function
        """
        allPass = (
            self.p.get_fmaps_W(0) == 8
            and self.p.get_fmaps_W(1) == 4
            and self.p.get_fmaps_W(2) == 2
            and self.p.get_fmaps_W(3) == 2
            and self.p.get_fmaps_W(4) == 2
        )

        self.assertTrue(allPass)  # for AlexNet on CIFAR

    def test_get_fmap_HW(self):
        """
        Test PytorchFI get_fmap_HW(...) function
        """
        allPass = (
            self.p.get_fmap_HW(0) == (8, 8)
            and self.p.get_fmap_HW(1) == (4, 4)
            and self.p.get_fmap_HW(2) == (2, 2)
            and self.p.get_fmap_HW(3) == (2, 2)
            and self.p.get_fmap_HW(4) == (2, 2)
        )

        self.assertTrue(allPass)  # for AlexNet on CIFAR

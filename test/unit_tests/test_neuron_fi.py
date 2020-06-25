# =================================#
# PyTorchFI Unit Tests
# =================================#

import os
import unittest

import torch
from pytorchfi.core import fault_injection as pfi_core

from .util_test import *


class TestNeuronFIgpu(unittest.TestCase):
    """
    Testing focuses on *neuron* perturbations on *gpu* with *batch = 1*.
    """

    def setUp(self):
        torch.manual_seed(0)  # for reproducability in testing

        # parameters
        self.BATCH_SIZE = 1
        self.WORKERS = 1
        self.DATASETS = "./data"
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()


        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(
            self.BATCH_SIZE, self.WORKERS, self.DATASETS
        )
        self.dataiter = iter(self.dataset)

        if self.USE_GPU:
            self.model.cuda()

        # golden output
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_neuronFI_singleElement(self):
        """
        Test PytorchFI declare_neuron_fi() function
        """
        # perturbation location
        batch_i = 0  # batch
        conv_i = 4  # layer
        c_i = 0  # fmap number in layer
        h_i = 1  # h coordinate in fmap
        w_i = 1  # w coordinate in fmap

        # perturbation value
        inj_value_i = 10000.0

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_1.eq(self.output)))

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            conv_num=conv_i,
            c=c_i,
            h=h_i,
            w=w_i,
            value=0,  # original (uncorrupted value) was 0
        )

        self.inj_model.eval()
        with torch.no_grad():
            uncorrupted_output = self.inj_model(self.images)

        self.assertTrue(torch.all(uncorrupted_output.eq(self.output)))

        ### check override works
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i * 2
        )

        # check output
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_2 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_2.eq(self.output)))
        self.assertTrue(torch.all(corrupted_output_2.eq(corrupted_output_2)))


class TestNeuronFIcpu(unittest.TestCase):
    """
    Testing focuses on *neuron* perturbations on *cpu* with *batch = 1*.
    """

    def setUp(self):
        torch.manual_seed(0)  # for reproducability in testing

        # parameters
        self.BATCH_SIZE = 1
        self.WORKERS = 1
        self.DATASETS = "./data"
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()


        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(
            self.BATCH_SIZE, self.WORKERS, self.DATASETS
        )
        self.dataiter = iter(self.dataset)

        if self.USE_GPU:
            self.model.cuda()

        # golden output
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_neuronFI_singleElement(self):
        """
        Test PytorchFI declare_neuron_fi() function
        """
        # perturbation location
        batch_i = 0  # batch
        conv_i = 4  # layer
        c_i = 0  # fmap number in layer
        h_i = 1  # h coordinate in fmap
        w_i = 1  # w coordinate in fmap

        # perturbation value
        inj_value_i = 10000.0

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_1.eq(self.output)))

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i,
            conv_num=conv_i,
            c=c_i,
            h=h_i,
            w=w_i,
            value=0,  # original (uncorrupted value) was 0
        )

        self.inj_model.eval()
        with torch.no_grad():
            uncorrupted_output = self.inj_model(self.images)

        self.assertTrue(torch.all(uncorrupted_output.eq(self.output)))

        ### check override works
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i * 2
        )

        # check output
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_2 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_2.eq(self.output)))
        self.assertTrue(torch.all(corrupted_output_2.eq(corrupted_output_2)))


class TestNeuronFIgpuBatch(unittest.TestCase):
    """
    Testing focuses on *neuron* perturbations on *gpu* with *batch = N*.
    """

    def setUp(self):
        torch.manual_seed(0)  # for reproducability in testing

        # parameters
        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.DATASETS = "./data"
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()


        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(
            self.BATCH_SIZE, self.WORKERS, self.DATASETS
        )
        self.dataiter = iter(self.dataset)

        if self.USE_GPU:
            self.model.cuda()

        # golden output
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_neuronFI_batch_1(self):
        """
        Test PytorchFI declare_neuron_fi() with a batch; only one element in the batch corrupted
        """
        # perturbation location
        batch_i = 2  # batch
        conv_i = 4  # layer
        c_i = 0  # fmap number in layer
        h_i = 1  # h coordinate in fmap
        w_i = 1  # w coordinate in fmap

        # perturbation value
        inj_value_i = 10000.0

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(torch.all(corrupted_output_1[0].eq(self.output[0])))
        self.assertTrue(torch.all(corrupted_output_1[1].eq(self.output[1])))
        self.assertTrue(not torch.all(corrupted_output_1[2].eq(self.output[2])))
        self.assertTrue(torch.all(corrupted_output_1[3].eq(self.output[3])))

    def test_neuronFI_batch_2(self):
        """
        Test PytorchFI declare_neuron_fi() with a batch; multiple elements in the batch corrupted
        """
        # perturbation location
        batch_i = [0, 2, 3]  # batch
        conv_i = [1, 2, 4]  # layer
        c_i = [3, 1, 1]  # fmap number in layer
        h_i = [1, 0, 1]  # h coordinate in fmap
        w_i = [0, 1, 0]  # w coordinate in fmap

        # perturbation value
        inj_value_i = [10000.0, 10000.0, 10000.0]

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_1[0].eq(self.output[0])))
        self.assertTrue(torch.all(corrupted_output_1[1].eq(self.output[1])))
        self.assertTrue(not torch.all(corrupted_output_1[2].eq(self.output[2])))
        self.assertTrue(not torch.all(corrupted_output_1[3].eq(self.output[3])))


class TestNeuronFIcpuBatch(unittest.TestCase):
    """
    Testing focuses on *neuron* perturbations on *cpu* with *batch = N*.
    """

    def setUp(self):
        torch.manual_seed(0)  # for reproducability in testing

        # parameters
        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.DATASETS = "./data"
        self.img_size = 32
        self.USE_GPU = torch.cuda.is_available()


        # get model and dataset
        self.model, self.dataset = helper_setUp_CIFAR10(
            self.BATCH_SIZE, self.WORKERS, self.DATASETS
        )
        self.dataiter = iter(self.dataset)

        if self.USE_GPU:
            self.model.cuda()

        # golden output
        self.images, self.labels = self.dataiter.next()
        if self.USE_GPU is True:
            self.images = self.images.cuda()

        self.model.eval()
        with torch.no_grad():
            self.output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            self.img_size,
            self.img_size,
            self.BATCH_SIZE,
            use_cuda=self.USE_GPU,
        )

    def test_neuronFI_batch_1(self):
        """
        Test PytorchFI declare_neuron_fi() with a batch; only one element in the batch corrupted
        """
        # perturbation location
        batch_i = 2  # batch
        conv_i = 4  # layer
        c_i = 0  # fmap number in layer
        h_i = 1  # h coordinate in fmap
        w_i = 1  # w coordinate in fmap

        # perturbation value
        inj_value_i = 10000.0

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(torch.all(corrupted_output_1[0].eq(self.output[0])))
        self.assertTrue(torch.all(corrupted_output_1[1].eq(self.output[1])))
        self.assertTrue(not torch.all(corrupted_output_1[2].eq(self.output[2])))
        self.assertTrue(torch.all(corrupted_output_1[3].eq(self.output[3])))

    def test_neuronFI_batch_2(self):
        """
        Test PytorchFI declare_neuron_fi() with a batch; multiple elements in the batch corrupted
        """
        # perturbation location
        batch_i = [0, 2, 3]  # batch
        conv_i = [1, 2, 4]  # layer
        c_i = [3, 1, 1]  # fmap number in layer
        h_i = [1, 0, 1]  # h coordinate in fmap
        w_i = [0, 1, 0]  # w coordinate in fmap

        # perturbation value
        inj_value_i = [10000.0, 10000.0, 10000.0]

        # declare neuron fi
        self.inj_model = self.p.declare_neuron_fi(
            batch=batch_i, conv_num=conv_i, c=c_i, h=h_i, w=w_i, value=inj_value_i
        )

        ### check output matches (corruption is benign)
        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        self.assertTrue(not torch.all(corrupted_output_1[0].eq(self.output[0])))
        self.assertTrue(torch.all(corrupted_output_1[1].eq(self.output[1])))
        self.assertTrue(not torch.all(corrupted_output_1[2].eq(self.output[2])))
        self.assertTrue(not torch.all(corrupted_output_1[3].eq(self.output[3])))

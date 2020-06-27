import torch
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.errormodels import error_models as em

from .util_test import helper_setUp_CIFAR10_same


class TestNeuronErrorModels:
    """
    Testing neuron perturbation error models.
    """

    def setup_class(self):
        torch.manual_seed(0)
        # random.seed(0)

        self.BATCH_SIZE = 4
        self.WORKERS = 1
        self.img_size = 32
        self.USE_GPU = False

        self.model, self.dataset = helper_setUp_CIFAR10_same(
            self.BATCH_SIZE, self.WORKERS
        )
        self.dataiter = iter(self.dataset)

        self.images, self.labels = self.dataiter.next()

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

    def test_random_neuron_inj(self):
        # TODO make sure only one batch element is different
        self.inj_model = em.random_neuron_inj(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_neuron_inj_batched_locTrue_valTrue(self):
        # TODO make sure only all batch elements are different
        self.inj_model = em.random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_neuron_inj_batched_locFalse_valTrue(self):
        # TODO make better test
        self.inj_model = em.random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_neuron_inj_batched_locTrue_valFalse(self):
        # TODO make better test
        self.inj_model = em.random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_neuron_inj_batched_locFalse_valFalse(self):
        # TODO make better test
        self.inj_model = em.random_neuron_inj_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_inj_per_layer(self):
        # TODO make better test
        self.inj_model = em.random_inj_per_layer(self.p, min_val=10000, max_val=20000)

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_inj_per_layer_batched_locTrue_valTrue(self):
        # TODO make better test
        self.inj_model = em.random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=True, randVal=True
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_inj_per_layer_batched_locFalse_valTrue(self):
        # TODO make better test
        self.inj_model = em.random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=True
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_inj_per_layer_batched_locTrue_valFalse(self):
        # TODO make better test
        self.inj_model = em.random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=True, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

    def test_random_inj_per_layer_batched_locFalse_valFalse(self):
        # TODO make better test
        self.inj_model = em.random_inj_per_layer_batched(
            self.p, min_val=10000, max_val=20000, randLoc=False, randVal=False
        )

        self.inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = self.inj_model(self.images)

        assert not torch.all(corrupted_output_1.eq(self.output))

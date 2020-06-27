"""
pytorchfi.util contains utility functions to help the user generate fault
injections and determine their impact.
"""

import random
import time

import torch
import torch.nn as nn
from pytorchfi import core


class util(core.fault_injection):
    def compare_golden(self, input_data):
        softmax = nn.Softmax(dim=1)

        model = self.get_original_model()
        golden_output = model(input_data)
        golden_output_softmax = softmax(golden_output)
        golden = list(torch.argmax(golden_output_softmax, dim=1))

        corrupted_model = self.get_corrupted_model()
        corrupted_output = corrupted_model(input_data)
        corrupted_output_softmax = softmax(corrupted_output)
        corrupted = list(torch.argmax(corrupted_output_softmax, dim=1))

        return [golden, corrupted]

    def time_model(self, model, input_data, iterations=100):
        start_time = time.time()
        for _ in range(iterations):
            model(input_data)
        end_time = time.time()
        return (end_time - start_time) / iterations

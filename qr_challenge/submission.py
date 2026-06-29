import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Baseline: cuSOLVER batched geqrf via PyTorch. Correctness reference path.
    return torch.geqrf(data)

from typing import Tuple
import torch

def enable_torch2trt(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor], batch_size: int):
    from torch2trt import torch2trt
    return torch2trt(model, example_inputs, max_batch_size=batch_size, strict_type_constraints=True)
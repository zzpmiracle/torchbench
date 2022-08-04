from builtins import breakpoint
from charset_normalizer import logging
import torchdynamo
from torchdynamo.optimizations.backends import create_backend
import torch
import torch_blade
from contextlib import contextmanager
from torch_blade import optimize as blade_optimize
from typing import Tuple

@contextmanager
def opt_disc_config():
    torch_config = torch_blade.config.Config()
    try:
        with torch_config:
             yield
    finally:
        pass

@create_backend
def blade_optimize_dynamo(subgraph):
    # print(subgraph.model.graph)
    with opt_disc_config(), torch.no_grad():
        optimized_model = blade_optimize(
            subgraph.model.eval(),
            allow_tracing=True,
            model_inputs=tuple(subgraph.example_inputs),
        )
    if torch_blade.mlir.num_engines(optimized_model) == 0:
        logging.warning("blade none fusion group")

    if not torchdynamo.utils.counters["blade_clusters"]:
        torchdynamo.utils.counters["blade_clusters"] = []
    torchdynamo.utils.counters["blade_clusters"].append(torch_blade.mlir.num_engines(optimized_model))
        # return subgraph.model

    # print(f"\nsubgraph clusters: {torch_blade.mlir.num_engines(optimized_model)}")
    
    # with open(f'model.code.py', 'a') as writer:
    #     writer.write(str(optimized_model.code))
    # with open(f'model.graph.txt', 'a') as writer:
    #     writer.write(str(optimized_model.graph))
    # print(optimized_model.graph)
    # breakpoint()
    return optimized_model

def blade_optimize_script(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor], ):
    with opt_disc_config(), torch.no_grad():
        optimized_model = blade_optimize(
            model.eval(),
            allow_tracing=True,
            model_inputs=tuple(example_inputs),
        )
        
    return optimized_model
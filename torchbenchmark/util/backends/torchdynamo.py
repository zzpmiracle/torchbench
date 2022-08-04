"""
Support TorchDynamo(https://github.com/facebookresearch/torchdynamo) backends
"""
import argparse
import functools
from typing import List
import torchdynamo
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
from .blade import blade_optimize_dynamo

EXTRA_BACKENDS = {
    "aot_autograd_speedup_strategy": aot_autograd_speedup_strategy,
}

TORCHDYNAMO_ROUNDS = 3

def parse_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', dyamo_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    available_backends = torchdynamo.list_backends()
    available_backends.extend(EXTRA_BACKENDS.keys())
    parser.add_argument(
        "--torchdynamo", choices=available_backends, help="Specify torchdynamo backends"
    )
    args = parser.parse_args(dyamo_args)
    return args


def apply_torchdynamo_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace, precision: str):
    torchdynamo.config.raise_on_backend_error = False
    torchdynamo.reset()
    torchdynamo.utils.counters.clear()

    if args.torchdynamo in EXTRA_BACKENDS:
        model.add_context(functools.partial(torchdynamo.optimize, EXTRA_BACKENDS[args.torchdynamo]))
    elif args.torchdynamo == "fx2trt" and precision == "fp16":
        model.add_context(functools.partial(torchdynamo.optimize, torchdynamo.optimizations.backends.fx2trt_compiler_fp16))
    else:
        model.add_context(functools.partial(torchdynamo.optimize, args.torchdynamo))
    
    for _ in range(TORCHDYNAMO_ROUNDS):
        model.invoke()
        print(torchdynamo.utils.counters["stats"]["unique_graphs"])
    model.run_contexts.pop()
    model.add_context(torchdynamo.run)
    print(model.run_contexts)

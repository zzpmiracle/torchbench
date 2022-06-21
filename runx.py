import torch
import time
import gc
import numpy as np
from torchbenchmark import load_model_by_name
import argparse

def synchronize():
    torch.cuda.synchronize()

def timed(model, example_inputs, times=1, dynamo=False):
    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.time_ns()
    # Dont collect outputs to correctly measure timing
    if dynamo:
        with torchdynamo.optimize("eager"):
            result = model(*example_inputs)
    else:
        result = model(*example_inputs)
    synchronize()
    t1 = time.time_ns()
    return (t1 - t0) / 1_000_000

def speedup_experiment(model, example_inputs, dynamo=False):
    repeat = 100
    timings = np.zeros((repeat, 2), np.float64)
    for rep in range(repeat):
        # interleave the runs to handle frequency scaling and load changes
        timings[rep, 0] = timed(model, example_inputs)
        if dynamo:
            timings[rep, 1] = timed(model, example_inputs, dynamo=True)
    median = np.median(timings, axis=0)
    print(f"Eager Latency: {median[0]} ms")
    if dynamo:
        print(f"TorchDynamo Eager latency: {median[1]} ms")
        print(f"speedup: {median[0]/median[1]} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchdynamo", action="store_true", help="load torchdynamo library")
    args = parser.parse_args()
    if args.torchdynamo:
        import torchdynamo
        optimize_ctx = torchdynamo.optimize("eager")
        with optimize_ctx:
            pass
    Model = load_model_by_name("hf_Bart")
    m = Model(device="cuda", test="eval", jit=False)
    model, example_inputs = m.get_module()
    speedup_experiment(model, example_inputs, dynamo=args.torchdynamo)

"""
Entry to all FLOPS counting methods
"""
from . import dispatch_flops
from typing import Optional

SUPPORTED_FLOPS_COUNTER = {
    "dispatch": dispatch_flops.get_flops 
}

def get_flops(flops_counter: str, model: 'torchbenchmark.util.model.BenchmarkModel') -> Optional[float]:
    assert flops_counter in SUPPORTED_FLOPS_COUNTER, f"Currently we only support {SUPPORTED_FLOPS_COUNTER}," \
                                                     f"but user is asking for {flops_counter}."
    counter = SUPPORTED_FLOPS_COUNTER[flops_counter]
    return counter(model)
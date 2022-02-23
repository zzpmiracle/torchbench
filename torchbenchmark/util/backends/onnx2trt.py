from typing import Tuple
import torch
import numpy as np

_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.longlong,
    torch.bool: np.bool_,
}

def onnxrt_common(subgraph, provider, onnx_filename=None):
    import onnxruntime

    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(
        onnx_filename or subgraph.onnx_filename, providers=[provider]
    )
    input_names = subgraph.input_names
    output_names = subgraph.output_names
    create_outputs = subgraph.empty_outputs_factory()
    is_cpu = subgraph.is_cpu

    def _call(*args):
        binding = session.io_binding()
        args = [a.contiguous() for a in args]
        for name, value in zip(input_names, args):
            dev = value.device
            binding.bind_input(
                name,
                dev,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        outputs = create_outputs()
        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev,
                dev.index or 0,
                _NP_DTYPE[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        session.run_with_iobinding(binding)
        if is_cpu:
            binding.copy_outputs_to_cpu()
        return outputs

    return _call

def enable_onnx2trt(model: torch.nn.Module):
    return onnxrt_common(model, provider="TensorrtExecutionProvider")
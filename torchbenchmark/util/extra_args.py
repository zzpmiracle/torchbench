import argparse
from typing import List
from torchbenchmark.util.backends.fuser import enable_fuser
from torchbenchmark.util.backends.fx2trt import enable_fx2trt
from torchbenchmark.util.backends.torch2trt import enable_torch2trt
from torchbenchmark.util.backends.torch_trt import enable_torchtrt
from torchbenchmark.util.backends.onnx2trt import enable_onnx2trt
from torchbenchmark.util.env_check import correctness_check
from torchbenchmark.util.framework.vision.args import enable_fp16

def add_bool_arg(parser: argparse.ArgumentParser, name: str, default_value: bool=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default_value})

def is_torchvision_model(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL

def allow_fp16(model: 'torchbenchmark.util.model.BenchmarkModel') -> bool:
    return is_torchvision_model(model) and model.test == 'eval' and model.device == 'cuda'

# Dispatch arguments based on model type
def parse_args(model: 'torchbenchmark.util.model.BenchmarkModel', extra_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorrt", choices=['fx2trt', 'torch_tensorrt', 'torch2trt', 'onnx2trt'],
                        help="enable TensorRT with one of the lowering libraries: fx2trt, torch_tensorrt, torch2trt, onnx2trt")
    parser.add_argument("--fuser", type=str, default="", help="enable fuser")
    # TODO: Enable fp16 for all model inference tests
    # fp16 is only True for torchvision models running CUDA inference tests
    # otherwise, it is False
    fp16_default_value = False
    if allow_fp16(model):
        fp16_default_value = True
    add_bool_arg(parser, "fp16", fp16_default_value)
    args = parser.parse_args(extra_args)
    args.device = model.device
    args.jit = model.jit
    args.test = model.test
    args.batch_size = model.batch_size
    if args.device == "cpu":
        args.fuser = None
    if not allow_fp16(model) and args.fp16:
        raise NotImplementedError("fp16 is only implemented for torchvision models inference tests on CUDA.")
    if not (model.device == "cuda" and model.test == "eval"):
        if args.tensorrt:
            raise NotImplementedError("TensorRT only works for CUDA inference tests.")
    if hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL:
        args.cudagraph = False
    return args

def apply_args(model: 'torchbenchmark.util.model.BenchmarkModel', args: argparse.Namespace):
    if args.fuser:
        enable_fuser(args.fuser)
    if args.fp16:
        assert allow_fp16(model), "Eval fp16 is only available on CUDA for torchvison models."
        model.model, model.example_inputs = enable_fp16(model.model, model.example_inputs)
    if args.tensorrt:
        module, exmaple_inputs = model.get_module()
        # get the output tensor of eval
        model.eager_output = model.eval()
        if args.tensorrt == "fx2trt":
            if args.jit:
                raise NotImplementedError("fx2trt with JIT is not available.")
            model.set_module(enable_fx2trt(args.batch_size, fp16=args.fp16, model=module, example_inputs=exmaple_inputs))
        if args.tensorrt == "torch_tensorrt":
            precision = 'fp16' if args.fp16 else 'fp32'
            model.set_module(enable_torchtrt(precision=precision, model=module, example_inputs=exmaple_inputs))
        if args.tensorrt == "torch2trt":
            model.set_module(enable_torch2trt(model=module, example_inputs=exmaple_inputs, batch_size=model.batch_size))
        if args.tensorrt == "onnx2trt":
            model.set_module(enable_onnx2trt(model=module))
        model.output = model.eval()
        model.correctness = correctness_check(model.eager_output, model.output)


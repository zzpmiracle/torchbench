"""
Script that runs torchbench with a benchmarking config.
The configs are located within the configs/ directory.
For example, the default config we use is `torchdynamo/eager-overhead`
"""
import re
import sys
import os
import yaml
import argparse
import subprocess
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bmutils import add_path
from bmutils.summarize import analyze_result
REPO_DIR = str(Path(__file__).parent.parent.parent.resolve())

with add_path(REPO_DIR):
    from torchbenchmark import _list_model_paths

@dataclass
class BenchmarkModelConfig:
    models: Optional[List[str]]
    device: str
    test: str
    batch_size: Optional[int]
    precision: Optional[str]
    args: List[str]
    rewritten_option: str

def rewrite_option(option: List[str]) -> str:
    out = []
    for x in option:
        out.append(x.replace("--", ""))
    if option == ['']:
        return "eager"
    else:
        return "-".join(out)

def get_models(config) -> Optional[str]:
    # if the config doesn't specify the 'models' key,
    # returns None (means running all models)
    if not "models" in config:
        return None
    # get list of models
    models = list(map(lambda x: os.path.basename(x), _list_model_paths()))
    enabled_models = []
    for model_pattern in config["models"]:
        r = re.compile(model_pattern)
        matched_models = list(filter(lambda x: r.match(x), models))
        enabled_models.extend(matched_models)
    assert enabled_models, f"The model patterns you specified {config['models']} does not match any model. Please double check."
    return enabled_models

def get_subrun_key(device, test, batch_size=None, precision=None):
    key = f"{test}-{device}"
    if batch_size:
        key += f"-bsize_{batch_size}"
    if precision:
        key += f"-{precision}"
    return key

def get_tests(config):
    if not "test" in config:
        return ["train", "eval"]
    return config["test"]

def get_devices(config):
    if not "device" in config:
        return ["cpu", "cuda"]
    return config["device"]

def get_batch_sizes(config):
    if not "batch_size" in config:
        return [None]
    return config["batch_size"]

def get_precisions(config):
    if not "precision" in config:
        return [""]
    return config["precision"]

def parse_bmconfigs(repo_path: Path, config_name: str) -> List[BenchmarkModelConfig]:
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_file = repo_path.joinpath("configs").joinpath(*config_name.split("/"))
    if not config_file.exists():
        raise RuntimeError(f"Benchmark model config {config_file} does not exist.")
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    out = {}

    models = get_models(config)
    devices = get_devices(config)
    tests = get_tests(config)
    batch_sizes = get_batch_sizes(config)
    precisions = get_precisions(config)

    bm_matrix = [devices, tests, batch_sizes, precisions]
    for device, test, batch_size, precision in itertools.product(*bm_matrix):
        subrun = (device, test, batch_size, precision)
        out[subrun] = []
        for args in config["args"]:
            out[subrun].append(BenchmarkModelConfig(models=models, device=device, test=test, \
                               batch_size=batch_size, precision=precision, args=args.split(" "), \
                               rewritten_option=rewrite_option(args.split(" "))))
    return out

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test]
    if config.batch_size:
        cmd.append("-b")
        cmd.append(str(config.batch_size))
    if config.models:
        cmd.append("-m")
        cmd.extend(config.models)
    if config.precision:
        cmd.append("--precision")
        cmd.append(config.precision)
    if config.args != ['']:
        cmd.extend(config.args)
    output_dir = output_path.joinpath("json")
    output_dir.mkdir(exist_ok=True, parents=True)
    cmd.extend(["-o", os.path.join(output_dir.absolute(), f"{config.rewritten_option}.json")])
    print(f"Now running benchmark command: {cmd}.", flush=True)
    if dryrun:
        return
    subprocess.check_call(cmd, cwd=repo_path)

def run_bmconfig_profiling(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    nsys_path = "/opt/nvidia/nsight-systems/2022.2.1/bin/nsys"
    profiling_cmd = [nsys_path, "profile", "-o", "", "-f", "true", "-c", "cudaProfilerApi", sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test, "--is-profiling"]
    stats_cmd = [nsys_path, "stats", "--report", "gputrace", "-f", "csv", "-o"]
    
    if config.batch_size:
        profiling_cmd.append("-b")
        profiling_cmd.append(str(config.batch_size))

    if config.precision:
        profiling_cmd.append("--precision")
        profiling_cmd.append(config.precision)
    if config.args != ['']:
        profiling_cmd.extend(config.args)
    output_dir = output_path.joinpath("profiling")
    output_dir.mkdir(exist_ok=True, parents=True)
    models = config.models or [os.path.basename(model_path) for model_path in _list_model_paths()]
    profiling_cmd.append("-m")
    for model in models:
        
        model_profiling_dir = output_dir.joinpath(model).absolute()
        model_profiling_dir.mkdir(exist_ok=True, parents=True)
        model_prefix = os.path.join(model_profiling_dir, f"{config.rewritten_option}")

        profiling_cmd[3] = model_prefix
        profiling_cmd.append(model)

        stats_cmd.append(model_prefix)
        stats_cmd.append(model_prefix + ".nsys-rep")
        parse_cmd = [sys.executable, "parse_nsys_result.py", model_prefix + "_gputrace.csv"]

        try:
            print(f"Now profiling benchmark command: {profiling_cmd}.", flush=True)
            subprocess.run(profiling_cmd, cwd=repo_path)
            print(f"Now stats benchmark command: {stats_cmd}.", flush=True)
            subprocess.check_call(stats_cmd, cwd=repo_path)
            print(f"Now parse benchmark command: {parse_cmd}.", flush=True)
            with open(model_prefix + ".log", "w") as fd:
                subprocess.check_call(parse_cmd, cwd=repo_path, stdout=fd)
        except subprocess.CalledProcessError:
            pass

        profiling_cmd.pop()
        stats_cmd = stats_cmd[:-2]
 

def gen_output_csv(output_path: Path, base_key: str):
    result = analyze_result(output_path.joinpath("json").absolute(), base_key=base_key)
    with open(output_path.joinpath("summary.csv"), "w") as sw:
        sw.write(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Specify benchmark config to run.")
    parser.add_argument("--benchmark-repo", "-b", required=True, help="Specify the pytorch/benchmark repository location.")
    parser.add_argument("--output-dir", "-o", required=True, help="Specify the directory to save the outputs.")
    parser.add_argument("--dryrun", action="store_true", help="Dry run the script and don't run the benchmark.")
    args = parser.parse_args()
    repo_path = Path(args.benchmark_repo)
    assert repo_path.exists(), f"Path {args.benchmark_repo} doesn't exist. Exit."
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    total_run = parse_bmconfigs(repo_path, args.config)
    assert len(total_run), "Size of the BenchmarkModel list must be larger than zero."
    for subrun in total_run:
        subrun_key = get_subrun_key(*subrun)
        bmconfigs = total_run[subrun]
        assert len(bmconfigs), f"Size of subrun {subrun} must be larger than zero."
        subrun_path = output_path.joinpath(subrun_key)
        subrun_path.mkdir(exist_ok=True, parents=True)
        for bm in bmconfigs:
            run_bmconfig(bm, repo_path, subrun_path, args.dryrun)
        if not args.dryrun:
            gen_output_csv(subrun_path, base_key=bmconfigs[0].rewritten_option)
        
        for bm in bmconfigs:
            run_bmconfig_profiling(bm, repo_path, subrun_path, args.dryrun)

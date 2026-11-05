#!/usr/bin/env python3
"""Build unchanged upstream CUDA sources for the local GPU only."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    import torch
    import torch.utils.cpp_extension as extension

    arch = ".".join(map(str, torch.cuda.get_device_capability()))
    compact = arch.replace(".", "")
    flag = f"-gencode=arch=compute_{compact},code=sm_{compact}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    original = extension.CUDAExtension

    def local_extension(*pos, **kw):
        compile_args = kw.setdefault("extra_compile_args", {})
        compile_args.pop("cores", None)  # Upstream metadata; not a compiler option.
        compile_args["nvcc"] = [
            v
            for v in compile_args.get("nvcc", [])
            if not v.startswith("-gencode") and "arch=compute_" not in v
        ] + [flag]
        return original(*pos, **kw)

    extension.CUDAExtension = local_extension
    torch.cuda.get_gencode_flags = lambda: flag  # curope's build metadata only.
    source = args.source.resolve()
    os.chdir(source)
    sys.argv = [str(source / "setup.py"), "build_ext", "--inplace"]
    runpy.run_path(str(source / "setup.py"), run_name="__main__")


if __name__ == "__main__":
    main()

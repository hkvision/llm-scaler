"""setup_moe_int4_ptl.py — build moe_int4_ops for Arc B390 (PTL/XE3) via JIT SYCL.

Follows the build-esimd-kernels skill: uses the repo's esimd_build_extention
(now with the skill's Bug 2 Windows-SYCL fix applied), which compiles .sycl with
icx directly — NO -fsycl-host-compiler (torch's stock SyclExtension adds that,
which mixes MSVC + clang intrinsic headers and breaks on VS 14.44 / oneAPI 2025.3).

Difference vs the stock setup_moe_int4_only.py: drop the AOT flags
(-fsycl-targets=spir64_gen -Xs -device bmg) → portable JIT, since B390 is
PTL/XE3 (XeLPG, no dpas2). Builds against the installed torch (ABI-matched).

    python setup_moe_int4_ptl.py build_ext --inplace
"""
from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension

# JIT build: don't pin an AOT arch.
os.environ.pop("TORCH_XPU_ARCH_LIST", None)

from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch

torch_include = str(Path(torch.__file__).parent / "include")

setup(
    name="custom-esimd-kernels-vllm-moe-int4-ptl",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        SyclExtension(
            name="custom_esimd_kernels_vllm.moe_int4_ops",
            sources=[
                "csrc/moe_batch/moe_int4.sycl",
            ],
            include_dirs=[
                # NO shadow include dir: test the clean toolchain first. The
                # earlier "SVML conflict" was never confirmed on a clean build —
                # every failing build had a patched_include shadow interfering.
                root / "csrc" / "moe_batch",
                root / "csrc" / "xpu" / "esimd_kernels",
                root / "csrc",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "sycl": [
                    "-fsycl",
                    "-ffast-math",
                    "-fsycl-device-code-split=per_kernel",
                    # JIT: NO "-fsycl-targets=spir64_gen", NO "-Xs -device bmg".
                    f"-I{torch_include}",
                ],
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
            py_limited_api=False,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

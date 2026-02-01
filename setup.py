import os
import glob
import sys
from setuptools import setup

import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "twobyfour"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    is_windows = sys.platform == "win32"

    if is_windows:
        # MSVC compiler flags for Windows
        cxx_flags = [
            "/std:c++20",
            "/O2" if not debug_mode else "/Od",
            "/DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ]
        nvcc_flags = [
            "-std=c++20",
            "-O2" if not debug_mode else "-O0",
        ]
        if debug_mode:
            cxx_flags.append("/Zi")
            nvcc_flags.append("-g")
            extra_link_args.extend(["/DEBUG"])
    else:
        # GCC/Clang compiler flags for Unix-like systems
        cxx_flags = [
            "-std=c++20",
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ]
        nvcc_flags = [
            "-std=c++20",
            "-O3" if not debug_mode else "-O0",
        ]
        if debug_mode:
            cxx_flags.append("-g")
            nvcc_flags.append("-g")
            extra_link_args.extend(["-O0", "-g"])

    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
    if py_limited_api else {},
)

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HEAD_SIZE = 64

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="rwkv7_state_fwd_fp16",
    ext_modules=[
        CUDAExtension(
            name="rwkv7_state_fwd_fp16",
            sources=[
                os.path.join(this_dir, "cuda", "rwkv7_state_fwd_fp16.cpp"),
                os.path.join(this_dir, "cuda", "rwkv7_state_fwd_fp16.cu"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                            "-res-usage",
                            "--use_fast_math",
                            "-O3",
                            "--extra-device-vectorization",
                            "-gencode=arch=compute_120,code=sm_120",
                            "-gencode=arch=compute_86,code=sm_86",
                            "-gencode=arch=compute_89,code=sm_89",
                            f"-D_N_={HEAD_SIZE}",
                        ] + ([] if os.name == "nt" else ["-Xptxas", "-O3"]),
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

using torch::Tensor;

extern "C" __global__ void rwkv_mm_sparsity_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int K,
    int C,
    int blk_size,
    int block_size
);

Tensor rwkv_mm_sparsity_forward(Tensor k, Tensor v) {
    TORCH_CHECK(k.is_cuda(), "k must be a HIP tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a HIP tensor");
    TORCH_CHECK(k.dim() == 1, "k must be 1D");
    TORCH_CHECK(v.dim() == 2, "v must be 2D");
    TORCH_CHECK(k.size(0) == v.size(0), "k.size(0) must match v.size(0)");

    const int K = k.size(0);
    const int C = v.size(1);
    const int blk_size = 512;
    const int block_size = 64;
    const int grid_x = (C + block_size - 1) / block_size;

    Tensor out = torch::empty({C}, k.options());

    hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 grid(grid_x);
    dim3 block(block_size);

    hipLaunchKernelGGL(
        rwkv_mm_sparsity_kernel,
        grid,
        block,
        0,  // shared memory
        stream,
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        K,
        C,
        blk_size,
        block_size
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}
TORCH_LIBRARY(rwkv_mm_sparsity_hip, m) {
    m.def("rwkv_mm_sparsity", &rwkv_mm_sparsity_forward);
}
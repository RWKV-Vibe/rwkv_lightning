#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _cublas_check_status = (condition);                      \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                         \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_cublas_check_status) + " at " +   \
                             std::to_string(__LINE__));

#define CUDA_CHECK(condition)                                                  \
  for (cudaError_t _cuda_check_status = (condition);                           \
       _cuda_check_status != cudaSuccess;)                                     \
    throw std::runtime_error(                                                  \
        "CUDA error " + std::string(cudaGetErrorString(_cuda_check_status)) +  \
        " at " + std::to_string(__LINE__));

namespace {

struct CacheKey {
    const void* w_impl;
    const void* mx_impl;
    const void* rx_impl;
    const void* my_impl;
    const void* ry_impl;

    bool operator==(const CacheKey& other) const {
        return w_impl == other.w_impl &&
               mx_impl == other.mx_impl &&
               rx_impl == other.rx_impl &&
               my_impl == other.my_impl &&
               ry_impl == other.ry_impl;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const {
        size_t seed = 0;
        auto hash_combine = [&seed](size_t value) {
            seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };

        hash_combine(std::hash<const void*>()(key.w_impl));
        hash_combine(std::hash<const void*>()(key.mx_impl));
        hash_combine(std::hash<const void*>()(key.rx_impl));
        hash_combine(std::hash<const void*>()(key.my_impl));
        hash_combine(std::hash<const void*>()(key.ry_impl));
        return seed;
    }
};

struct W8A8Weight {
    torch::Tensor q;
    torch::Tensor scale;
};

std::mutex& w8a8_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<CacheKey, W8A8Weight, CacheKeyHash>& w8a8_weight_cache() {
    static std::unordered_map<CacheKey, W8A8Weight, CacheKeyHash> cache;
    return cache;
}

CacheKey make_cache_key(
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry) {
    return CacheKey{
        w.unsafeGetTensorImpl(),
        mx.unsafeGetTensorImpl(),
        rx.unsafeGetTensorImpl(),
        my.unsafeGetTensorImpl(),
        ry.unsafeGetTensorImpl(),
    };
}

W8A8Weight get_cached_w8a8_weight(
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry) {
    const auto key = make_cache_key(w, mx, rx, my, ry);

    {
        std::lock_guard<std::mutex> lock(w8a8_cache_mutex());
        auto& cache = w8a8_weight_cache();
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    auto weight = w.to(torch::kFloat).add(0.5);
    weight = weight.mul(ry.to(torch::kFloat));
    weight = weight.mul(rx.to(torch::kFloat));
    weight = weight.add(my.to(torch::kFloat));
    weight = weight.add(mx.to(torch::kFloat));

    auto scale = std::get<0>(weight.abs().max(0)).div(127.0).clamp_min(1.0e-8).contiguous();
    auto q = weight.div(scale).round().clamp(-127, 127).to(torch::kInt8).contiguous();
    W8A8Weight quantized{q, scale};

    {
        std::lock_guard<std::mutex> lock(w8a8_cache_mutex());
        auto& cache = w8a8_weight_cache();
        auto [it, inserted] = cache.emplace(key, quantized);
        if (!inserted) {
            return it->second;
        }
    }

    return quantized;
}

__global__ void quantize_rows_fp16_kernel(
    const __half* __restrict__ x,
    int x_stride,
    int8_t* __restrict__ q,
    float* __restrict__ scale,
    int B,
    int N) {
    const int row = blockIdx.x;
    __shared__ float shared[256];

    float local_max = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(x[row * x_stride + col])));
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    const float row_scale = fmaxf(shared[0] / 127.0f, 1.0e-8f);
    if (threadIdx.x == 0) {
        scale[row] = row_scale;
    }

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float v = nearbyintf(__half2float(x[row * x_stride + col]) / row_scale);
        v = fminf(127.0f, fmaxf(-127.0f, v));
        q[row * N + col] = static_cast<int8_t>(v);
    }
}

__global__ void dequant_i32_to_fp16_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    __half* __restrict__ y,
    int y_stride,
    int B,
    int M) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * M;
    if (idx >= total) {
        return;
    }
    const int row = idx / M;
    const int col = idx - row * M;
    const float value = static_cast<float>(acc[idx]) * x_scale[row] * w_scale[col];
    y[row * y_stride + col] = __float2half(value);
}

__global__ void dequant_i32_to_fp32_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    float* __restrict__ y,
    int y_stride,
    int B,
    int M) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * M;
    if (idx >= total) {
        return;
    }
    const int row = idx / M;
    const int col = idx - row * M;
    y[row * y_stride + col] = static_cast<float>(acc[idx]) * x_scale[row] * w_scale[col];
}

void gemm_w8a8_cublaslt(torch::Tensor x_q, torch::Tensor w_q, torch::Tensor acc) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x_q));
    const int32_t alpha = 1;
    const int32_t beta = 0;

    // Row-major X[B, N] @ W[N, M] has the same bytes as column-major
    // W^T[M, N] @ X^T[N, B] -> Y^T[M, B].
    const int m = w_q.size(1);
    const int n = x_q.size(0);
    const int k = x_q.size(1);
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t trans = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8I, m, k, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, k, n, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32I, m, n, ldc));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
    void* workspace = at::cuda::getCUDABlasLtWorkspace();
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned));
    if (returned == 0) {
        throw std::runtime_error("cuBLASLt found no W8A8 INT8 matmul algorithm");
    }

    CUBLAS_CHECK(cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        w_q.data_ptr(),
        a_desc,
        x_q.data_ptr(),
        b_desc,
        &beta,
        acc.data_ptr(),
        c_desc,
        acc.data_ptr(),
        c_desc,
        &heuristic.algo,
        workspace,
        workspace_size,
        at::cuda::getCurrentCUDAStream()));

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

void mm8_w8a8_int8(
    int64_t B,
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry,
    const torch::Tensor& y) {
    if (x.scalar_type() != torch::kHalf) {
        throw std::runtime_error("mm8 W8A8 only supports FP16 activations before dynamic quantization");
    }
    if ((N % 4) != 0 || (M % 4) != 0) {
        throw std::runtime_error("mm8 W8A8 requires N and M to be multiples of 4 for INT8 Tensor Core GEMM");
    }

    auto quantized_weight = get_cached_w8a8_weight(w, mx, rx, my, ry);
    auto x_q = torch::empty({B, N}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({B}, x.options().dtype(torch::kFloat));
    auto acc = torch::empty({B, M}, x.options().dtype(torch::kInt32));

    quantize_rows_fp16_kernel<<<B, 256>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x.stride(0),
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        B,
        N);
    CUDA_CHECK(cudaGetLastError());

    gemm_w8a8_cublaslt(x_q, quantized_weight.q, acc);

    const int threads = 256;
    const int blocks = (B * M + threads - 1) / threads;
    if (y.scalar_type() == torch::kHalf) {
        dequant_i32_to_fp16_kernel<<<blocks, threads>>>(
            acc.data_ptr<int32_t>(),
            x_scale.data_ptr<float>(),
            quantized_weight.scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            y.stride(0),
            B,
            M);
    } else if (y.scalar_type() == torch::kFloat) {
        dequant_i32_to_fp32_kernel<<<blocks, threads>>>(
            acc.data_ptr<int32_t>(),
            x_scale.data_ptr<float>(),
            quantized_weight.scale.data_ptr<float>(),
            y.data_ptr<float>(),
            y.stride(0),
            B,
            M);
    } else {
        throw std::runtime_error("mm8 W8A8 output must be FP16 or FP32");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void mm8_seq_cuda(
    int64_t B,
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry,
    torch::Tensor& y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    mm8_w8a8_int8(B, N, M, x, w, mx, rx, my, ry, y);
}

void mm8_one_cuda(
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry,
    torch::Tensor& y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    auto x_2d = x.view({1, x.size(0)});
    auto y_2d = y.view({1, y.size(0)});
    mm8_w8a8_int8(1, N, M, x_2d, w, mx, rx, my, ry, y_2d);
}

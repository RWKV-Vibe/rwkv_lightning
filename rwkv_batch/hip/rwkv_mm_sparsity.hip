#include <hip/hip_runtime.h>

#define WAVEFRONT_SIZE 32

extern "C" __global__ void rwkv_mm_sparsity_kernel(
    const float* __restrict__ k, // [K]
    const float* __restrict__ v, // [K * C] row-major
    float* __restrict__ out,     // [C]
    int K,
    int C,
    int blk_size
) {
    // 使用更符合AMD GPU特性的线程布局
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= C) return;

    float acc = 0.0f;
    
    // 分块处理 k 向量
    for (int base = 0; base < K; base += blk_size) {
        int len = blk_size;
        if (base + len > K) len = K - base;
        
        // 计算当前块的部分结果
        for (int i = 0; i < len; ++i) {
            int idx = base + i;
            float kv = k[idx];
            // 只有非零元素才参与计算（保持稀疏性）
            if (kv != 0.0f) {
                acc += kv * v[(size_t)idx * (size_t)C + (size_t)col_idx];
            }
        }
    }
    
    out[col_idx] = acc;
}
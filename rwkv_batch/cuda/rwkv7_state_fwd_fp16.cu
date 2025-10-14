#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <stdio.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>

#ifndef _N_
#define _N_ 64
#endif

typedef at::Half F;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;
#define rotator(_A,_B,_C) (two_to_neg_41*float(ro1*(_A)+ro2*(_B)+ro3*(_C)))
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))

// __global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
//                                 F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                 F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     const int t0 = _elapsed_t[bbb];
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
//     __half state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = static_cast<__half>(_state[j]);
//     __shared__ __half r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
//     for (int _t = 0; _t < T; _t++){
//         const int t = bbb*T*C + h*_N_ + i + _t * C;
//         __syncthreads();
//         r[i] = static_cast<__half>(_r[t]);
//         w[i] = __half(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(t0+_t));
//         k[i] = static_cast<__half>(_k[t]);
//         a[i] = static_cast<__half>(_a[t]);
//         b[i] = static_cast<__half>(_b[t]);
//         __syncthreads();
//         __half sa = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             sa += a[j] * state[j];
//         }
//         __half vv = static_cast<__half>(_v[t]);
//         __half y = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             __half& s = state[j];
//             s += s * w[j] + k[j] * vv + sa * b[j];
//             y += s * r[j];
//         }
//         _y[t] = F(y);
//     }
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = static_cast<F>(state[j]);
// }


__global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        __half2 sa2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 += a[j] * state[j];
        __half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        __half vv = _v[t];
        __half2 vv2 = {vv, vv};
        __half2 y2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
        {
            __half2 &s = state[j];
            s += s * w[j] + k[j] * vv2 + sa2 * b[j];
            y2 += s * r[j];
        }
        _y[t] = y2.x + y2.y;
    }
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

__global__ void kernel_forward_w0_fp16_dither_one(const int B, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    const int t = bbb * C + h * _N_ + i;
    // float w0 = __expf(_w[t]);
    // w0 = w0 / (w0 + 1);
    // sigmoid = 1 / (1+exp2f(-nlog2e * x))
    ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((F *)k)[i] = _k[t];
    ((F *)a)[i] = _a[t];
    ((F *)b)[i] = _b[t];
    ((F *)r)[i] = _r[t];
    __syncthreads();
    __half2 sa2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        sa2 += a[j] * state[j];
    __half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};

    __half vv = _v[t];
    __half2 vv2 = {vv, vv};
    __half2 y2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        __half2 &s = state[j];
        s += s * w[j] + k[j] * vv2 + sa2 * b[j];
        y2 += s * r[j];
    }
    _y[t] = y2.x + y2.y;

#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

// __global__ void kernel_forward_w0_fp16_dither_one_ref(const int B, const int C, const int H,
//                                 F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                 F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
//     F state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = _state[j];
//     __shared__ F r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
//     const int t = bbb*C + h*_N_ + i;
//     __syncthreads();
//     r[i] = _r[t];
//     w[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb])); 
//     k[i] = _k[t];
//     a[i] = _a[t];
//     b[i] = _b[t];
//     __syncthreads();
//     F sa = 0;
//     #pragma unroll
//     for (int j = 0; j < _N_; j++){
//         sa += a[j] * state[j];
//     }
//     F vv = F(_v[t]);
//     F y = 0;
//     #pragma unroll
//     for (int j = 0; j < _N_; j++){
//         F& s = state[j];
//         s += s * w[j] + k[j] * vv + sa * b[j];
//         y += s * r[j];
//     }
//     _y[t] = F(y);
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = state[j];    
// }

void cuda_forward_seq(int B, int T, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_seq<<<B * H, _N_, 0, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_forward_one(int B, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_one<<<B * H, _N_, 0, stream>>>(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

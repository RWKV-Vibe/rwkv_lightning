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
#define BLOCKDIM 128
#define MAXNPERBLOCK 64

typedef at::Half F;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769;
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))


__global__ void kernel_forward_w0_fp16_dither_seq(
    const int B, const int T, const int C, const int H,
    F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
    F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){

    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];
    
    __shared__ half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    for (int _t = 0; _t < T; _t++){
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        half2 sa2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 += a[j] * state[j];
        half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        half vv = _v[t];
        half2 vv2 = {vv, vv};
        half2 y2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++){
            half2 &s = state[j];
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
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

__global__ void kernel_forward_w0_fp16_dither_one(
    const int B, const int C, const int H,
    F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
    F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];
    
    __shared__ half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    const int t = bbb * C + h * _N_ + i;
    ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((F *)k)[i] = _k[t];
    ((F *)a)[i] = _a[t];
    ((F *)b)[i] = _b[t];
    ((F *)r)[i] = _r[t];
    __syncthreads();
    half2 sa2 = {0., 0.};
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        sa2 += a[j] * state[j];
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};

    half vv = _v[t];
    half2 vv2 = {vv, vv};
    half2 y2 = {0., 0.};
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++){
        half2 &s = state[j];
        s += s * w[j] + k[j] * vv2 + sa2 * b[j];
        y2 += s * r[j];
    }
    _y[t] = y2.x + y2.y;

    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}


union common128 {
    int4 I;
    struct {int x,y,z,w;} J;
    struct {float x,y,z,w;} F;
    struct {double x,y;} D;
    struct {half2 x,y,z,w;} G;
    struct {half a,b,c,d,e,f,g,h;} H;
    half h[8];
    int i[4];
    float f[4];
};

template <int N>
__device__ __forceinline__ void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr, bool cond) {
    static_assert(N == 16 || N == 8 || N == 4);
    int bytes = cond ? N : 0;
    unsigned int addr = (unsigned int)(smem_addr);
    if constexpr (N == 16) {
        asm volatile(
            #if ENABLE_L2_PREFETCH
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
            #else
            "cp.async.cg.shared.global [%0], [%1], %2, %3;"
            #endif
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    } else {
        asm volatile(
            #if ENABLE_L2_PREFETCH
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
            #else
            "cp.async.ca.shared.global [%0], [%1], %2, %3;"
            #endif
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    }
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__global__ void __launch_bounds__(BLOCKDIM, 1) spvecmatmul_noindices(
    const int C,
    const half* __restrict__ vec,
    const half* __restrict__ mat,
    half* __restrict__ out
){
    __shared__ __align__(256) half mat_row_smem[2][2*BLOCKDIM];
    __shared__ __align__(256) half vec_slice[MAXNPERBLOCK];
    __shared__ __align__(256) int nnz_ids[MAXNPERBLOCK];
    __shared__ int nnz_count;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int t = threadIdx.x;
    const int start_pos = bx * MAXNPERBLOCK;

    if (t < 32){
        *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
    }
    __syncthreads();
    if (t == 0){
        int cnt = 0;
        #pragma unroll
        for (int i=0; i<8; ++i) {
            common128 z;
            z.I = ((const int4*)vec_slice)[i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned short bits = __half_as_ushort(z.h[j]);
                if (bits != 0x0000 && bits != 0x8000) {
                    int idx = i * 8 + j;
                    nnz_ids[cnt] = idx;
                    cnt++;
                }
            }
        }
        nnz_count = cnt;
    }
    __syncthreads();

    half2 out_frag;
    *(int*)(&out_frag) = 0;
    // init
    #pragma unroll
    for(int i = 0; i < 2; i++){
        if (i < nnz_count){
            int actual_pos = start_pos + nnz_ids[i];
            cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
            cp_async_commit();
        }
    }
    // main for
    for(int i = 0; i < nnz_count-2; i++){
        // take data
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
        half vec_value = vec_slice[nnz_ids[i]];

        // store
        int actual_pos = start_pos + nnz_ids[i+2];
        cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
        cp_async_commit();

        // compute
        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }

    // end
    if (nnz_count >= 2){
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    if (nnz_count >= 1){
        cp_async_wait<0>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    atomicAdd((half2*)(out + by*(2*BLOCKDIM) + t*2), out_frag);
}


void cuda_forward_seq(int B, int T, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t){
    assert(H * _N_ == C);
    kernel_forward_w0_fp16_dither_seq<<<B * H, _N_>>>(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_forward_one(int B, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t){
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_one<<<B * H, _N_, 0, stream>>>(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_spmv_forward(int D, int C, F* vec1, F* mat, F* out) {
    assert(C % (2*BLOCKDIM) == 0);
    assert(D % MAXNPERBLOCK == 0);
    auto stream = at::cuda::getCurrentCUDAStream();
    spvecmatmul_noindices<<<dim3(D/MAXNPERBLOCK, C/(2*BLOCKDIM), 1), dim3(BLOCKDIM, 1, 1), 0, stream>>>
    (C, (const half*)vec1, (const half*)mat, (half*)out);
}

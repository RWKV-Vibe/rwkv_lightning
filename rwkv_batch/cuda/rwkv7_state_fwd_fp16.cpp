#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::Half F;

void cuda_forward_seq(int B, int T, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int* elapsed_t);
void cuda_forward_one(int B,        int C, int H, F *state, F *r, F* w, F *k, F *v, F *a, F *b, F *y, int* elapsed_t);
void cuda_spmv_forward(int D, int C, F* vec, F* mat, F* out);

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t) {
    cuda_forward_seq(B, T, C, H, state.data_ptr<F>(), r.data_ptr<F>(), w.data_ptr<F>(), k.data_ptr<F>(), v.data_ptr<F>(), a.data_ptr<F>(), b.data_ptr<F>(), y.data_ptr<F>(), elapsed_t.data_ptr<int>());
}

void forward_one(int64_t B,            int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t) {
    cuda_forward_one(B,    C, H, state.data_ptr<F>(), r.data_ptr<F>(), w.data_ptr<F>(), k.data_ptr<F>(), v.data_ptr<F>(), a.data_ptr<F>(), b.data_ptr<F>(), y.data_ptr<F>(), elapsed_t.data_ptr<int>());
}
void spmv_forward(int64_t D, int64_t C, torch::Tensor &vec, torch::Tensor &mat, torch::Tensor &out) {
    cuda_spmv_forward(D, C, vec.data_ptr<F>(), mat.data_ptr<F>(), out.data_ptr<F>());
}

TORCH_LIBRARY(rwkv7_state_fwd_fp16, m) {
    m.def("forward_one", forward_one);
    m.def("forward_seq", forward_seq);
    m.def("spmv_forward", spmv_forward);
}
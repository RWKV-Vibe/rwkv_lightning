#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::Half dtype;

void cuda_forward_seq(int B, int T, int C, int H, dtype *state, dtype *r, dtype *w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* elapsed_t);
void cuda_forward_one(int B,        int C, int H, dtype *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* elapsed_t);

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t) {
    cuda_forward_seq(B, T, C, H, state.data_ptr<dtype>(), r.data_ptr<dtype>(), w.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), a.data_ptr<dtype>(), b.data_ptr<dtype>(), y.data_ptr<dtype>(), elapsed_t.data_ptr<int>());
}
void forward_one(int64_t B,            int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &w, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t) {
    cuda_forward_one(B,    C, H, state.data_ptr<dtype>(), r.data_ptr<dtype>(), w.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), a.data_ptr<dtype>(), b.data_ptr<dtype>(), y.data_ptr<dtype>(), elapsed_t.data_ptr<int>());
}

TORCH_LIBRARY(rwkv7_state_fwd_fp16, m) {
    m.def("forward_one", forward_one);
    m.def("forward_seq", forward_seq);
}

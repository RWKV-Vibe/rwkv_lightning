#include <torch/extension.h>
#include "ATen/ATen.h"

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t);
void forward_one(int64_t B, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t);
void spmv_forward(int64_t D, int64_t C, at::Tensor &vec1, at::Tensor &mat, at::Tensor &out);
void cmix_up(int64_t indim, int64_t outdim, at::Tensor &x_0, at::Tensor &x_1, at::Tensor &x_k, at::Tensor &key, at::Tensor &out);
void copy_zero(at::Tensor &a, at::Tensor &b, at::Tensor &c);
at::Tensor setup_rand(int64_t seed, int64_t B);
at::Tensor batch_sampling_repetition_temperature_topk_topp(at::Tensor& logits, at::Tensor& penalties, at::Tensor& states, double presence_penalty, double repetition_penalty, double penalty_decay, double temperature, int64_t top_k, double top_p);
at::Tensor cmix_one(
    at::Tensor &x_0, //(4096)
    at::Tensor &x_1, //(4096)
    at::Tensor &x_k, //(4096)
    at::Tensor &key, //(16384, 4096)
    at::Tensor &val  //(16384, 4096)
){
    int64_t middim = key.size(0);
    int64_t indim = x_0.size(0);
    // std::cout << indim << ' ' << middim << std::endl;
    auto act = at::empty({middim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
    auto out = at::empty({indim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
    cmix_up(indim, middim, x_0, x_1, x_k, key, act);
    copy_zero(x_0, x_1, out);
    spmv_forward(middim, indim, act, val, out);
    return out;
}

TORCH_LIBRARY(rwkv7_state_fwd_fp16, m) {
    m.def("forward_seq", forward_seq);
    m.def("forward_one", forward_one);
    m.def("spmv_forward", spmv_forward);
    m.def("cmix_up", cmix_up);
    m.def("copy_zero", copy_zero);
    m.def("cmix_one", cmix_one);
    m.def("setup_rand", setup_rand);
    m.def("batch_sampling_repetition_temperature_topk_topp", batch_sampling_repetition_temperature_topk_topp);
}
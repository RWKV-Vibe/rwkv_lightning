#include <torch/extension.h>

#include <cassert>

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
    torch::Tensor& y);


void mm8_prequant_seq_cuda(
    int64_t B,
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& w_scale,
    torch::Tensor& y);

void mm8_prequant_one_cuda(
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& w_scale,
    torch::Tensor& y);

void mm8_one_cuda(
    int64_t N,
    int64_t M,
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& mx,
    const torch::Tensor& rx,
    const torch::Tensor& my,
    const torch::Tensor& ry,
    torch::Tensor& y);

namespace {

void mm8_seq(
    int64_t B,
    int64_t N,
    int64_t M,
    torch::Tensor& x,
    torch::Tensor& w,
    torch::Tensor& mx,
    torch::Tensor& rx,
    torch::Tensor& my,
    torch::Tensor& ry,
    torch::Tensor& y) {
    assert(x.is_cuda());
    assert(w.is_cuda());
    assert(mx.is_cuda() && rx.is_cuda());
    assert(my.is_cuda() && ry.is_cuda());
    assert(y.is_cuda());
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    assert(x.scalar_type() == torch::kHalf);
    mm8_seq_cuda(B, N, M, x, w, mx, rx, my, ry, y);
}

void mm8_one(
    int64_t N,
    int64_t M,
    torch::Tensor& x,
    torch::Tensor& w,
    torch::Tensor& mx,
    torch::Tensor& rx,
    torch::Tensor& my,
    torch::Tensor& ry,
    torch::Tensor& y) {
    assert(x.is_cuda());
    assert(w.is_cuda());
    assert(mx.is_cuda() && rx.is_cuda());
    assert(my.is_cuda() && ry.is_cuda());
    assert(y.is_cuda());
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    assert(x.scalar_type() == torch::kHalf);
    mm8_one_cuda(N, M, x, w, mx, rx, my, ry, y);
}


void mm8_prequant_seq(
    int64_t B,
    int64_t N,
    int64_t M,
    torch::Tensor& x,
    torch::Tensor& w_q,
    torch::Tensor& w_scale,
    torch::Tensor& y) {
    assert(x.is_cuda());
    assert(w_q.is_cuda());
    assert(w_scale.is_cuda());
    assert(y.is_cuda());
    assert(x.stride(1) == 1);
    assert(w_q.stride(1) == 1);
    assert(w_scale.stride(0) == 1);
    assert(y.stride(1) == 1);
    assert(x.scalar_type() == torch::kHalf);
    assert(w_q.scalar_type() == torch::kInt8);
    assert(w_scale.scalar_type() == torch::kFloat);
    mm8_prequant_seq_cuda(B, N, M, x, w_q, w_scale, y);
}

void mm8_prequant_one(
    int64_t N,
    int64_t M,
    torch::Tensor& x,
    torch::Tensor& w_q,
    torch::Tensor& w_scale,
    torch::Tensor& y) {
    assert(x.is_cuda());
    assert(w_q.is_cuda());
    assert(w_scale.is_cuda());
    assert(y.is_cuda());
    assert(x.stride(0) == 1);
    assert(w_q.stride(1) == 1);
    assert(w_scale.stride(0) == 1);
    assert(y.stride(0) == 1);
    assert(x.scalar_type() == torch::kHalf);
    assert(w_q.scalar_type() == torch::kInt8);
    assert(w_scale.scalar_type() == torch::kFloat);
    mm8_prequant_one_cuda(N, M, x, w_q, w_scale, y);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_one", &mm8_one, "mm8 one");
    m.def("mm8_prequant_seq", &mm8_prequant_seq, "mm8 prequant seq");
    m.def("mm8_prequant_one", &mm8_prequant_one, "mm8 prequant one");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
    m.def("mm8_prequant_seq", mm8_prequant_seq);
    m.def("mm8_prequant_one", mm8_prequant_one);
}

#include <torch/extension.h>

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
    TORCH_CHECK(x.is_cuda(), "mm8_seq: x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "mm8_seq: w must be CUDA");
    TORCH_CHECK(mx.is_cuda() && rx.is_cuda(), "mm8_seq: mx/rx must be CUDA");
    TORCH_CHECK(my.is_cuda() && ry.is_cuda(), "mm8_seq: my/ry must be CUDA");
    TORCH_CHECK(y.is_cuda(), "mm8_seq: y must be CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == B && x.size(1) == N, "mm8_seq: x shape must be [B, N]");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == N && w.size(1) == M, "mm8_seq: w shape must be [N, M]");
    TORCH_CHECK(mx.numel() == M && rx.numel() == M, "mm8_seq: mx/rx numel must be M");
    TORCH_CHECK(my.numel() == N && ry.numel() == N, "mm8_seq: my/ry numel must be N");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == B && y.size(1) == M, "mm8_seq: y shape must be [B, M]");
    TORCH_CHECK(x.stride(1) == 1, "mm8_seq: x last dimension must be contiguous");
    TORCH_CHECK(w.stride(1) == 1, "mm8_seq: w last dimension must be contiguous");
    TORCH_CHECK(mx.stride(0) == 1 && rx.stride(0) == 1, "mm8_seq: mx/rx must be contiguous");
    TORCH_CHECK(my.stride(0) == 1 && ry.stride(0) == 1, "mm8_seq: my/ry must be contiguous");
    TORCH_CHECK(y.stride(1) == 1, "mm8_seq: y last dimension must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "mm8_seq: x must be FP16");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8, "mm8_seq: w must be uint8");
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
    TORCH_CHECK(x.is_cuda(), "mm8_one: x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "mm8_one: w must be CUDA");
    TORCH_CHECK(mx.is_cuda() && rx.is_cuda(), "mm8_one: mx/rx must be CUDA");
    TORCH_CHECK(my.is_cuda() && ry.is_cuda(), "mm8_one: my/ry must be CUDA");
    TORCH_CHECK(y.is_cuda(), "mm8_one: y must be CUDA");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == N, "mm8_one: x shape must be [N]");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == N && w.size(1) == M, "mm8_one: w shape must be [N, M]");
    TORCH_CHECK(mx.numel() == M && rx.numel() == M, "mm8_one: mx/rx numel must be M");
    TORCH_CHECK(my.numel() == N && ry.numel() == N, "mm8_one: my/ry numel must be N");
    TORCH_CHECK(y.dim() == 1 && y.size(0) == M, "mm8_one: y shape must be [M]");
    TORCH_CHECK(x.stride(0) == 1, "mm8_one: x must be contiguous");
    TORCH_CHECK(w.stride(1) == 1, "mm8_one: w last dimension must be contiguous");
    TORCH_CHECK(mx.stride(0) == 1 && rx.stride(0) == 1, "mm8_one: mx/rx must be contiguous");
    TORCH_CHECK(my.stride(0) == 1 && ry.stride(0) == 1, "mm8_one: my/ry must be contiguous");
    TORCH_CHECK(y.stride(0) == 1, "mm8_one: y must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "mm8_one: x must be FP16");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8, "mm8_one: w must be uint8");
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
    TORCH_CHECK(x.is_cuda(), "mm8_prequant_seq: x must be CUDA");
    TORCH_CHECK(w_q.is_cuda(), "mm8_prequant_seq: w_q must be CUDA");
    TORCH_CHECK(w_scale.is_cuda(), "mm8_prequant_seq: w_scale must be CUDA");
    TORCH_CHECK(y.is_cuda(), "mm8_prequant_seq: y must be CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == B && x.size(1) == N, "mm8_prequant_seq: x shape must be [B, N]");
    TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == N && w_q.size(1) == M, "mm8_prequant_seq: w_q shape must be [N, M]");
    TORCH_CHECK(w_scale.numel() == M, "mm8_prequant_seq: w_scale numel must be M");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == B && y.size(1) == M, "mm8_prequant_seq: y shape must be [B, M]");
    TORCH_CHECK(x.stride(1) == 1, "mm8_prequant_seq: x last dimension must be contiguous");
    TORCH_CHECK(w_q.stride(1) == 1, "mm8_prequant_seq: w_q last dimension must be contiguous");
    TORCH_CHECK(w_scale.stride(0) == 1, "mm8_prequant_seq: w_scale must be contiguous");
    TORCH_CHECK(y.stride(1) == 1, "mm8_prequant_seq: y last dimension must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "mm8_prequant_seq: x must be FP16");
    TORCH_CHECK(w_q.scalar_type() == torch::kInt8, "mm8_prequant_seq: w_q must be int8");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat, "mm8_prequant_seq: w_scale must be FP32");
    mm8_prequant_seq_cuda(B, N, M, x, w_q, w_scale, y);
}

void mm8_prequant_one(
    int64_t N,
    int64_t M,
    torch::Tensor& x,
    torch::Tensor& w_q,
    torch::Tensor& w_scale,
    torch::Tensor& y) {
    TORCH_CHECK(x.is_cuda(), "mm8_prequant_one: x must be CUDA");
    TORCH_CHECK(w_q.is_cuda(), "mm8_prequant_one: w_q must be CUDA");
    TORCH_CHECK(w_scale.is_cuda(), "mm8_prequant_one: w_scale must be CUDA");
    TORCH_CHECK(y.is_cuda(), "mm8_prequant_one: y must be CUDA");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == N, "mm8_prequant_one: x shape must be [N]");
    TORCH_CHECK(w_q.dim() == 2 && w_q.size(0) == N && w_q.size(1) == M, "mm8_prequant_one: w_q shape must be [N, M]");
    TORCH_CHECK(w_scale.numel() == M, "mm8_prequant_one: w_scale numel must be M");
    TORCH_CHECK(y.dim() == 1 && y.size(0) == M, "mm8_prequant_one: y shape must be [M]");
    TORCH_CHECK(x.stride(0) == 1, "mm8_prequant_one: x must be contiguous");
    TORCH_CHECK(w_q.stride(1) == 1, "mm8_prequant_one: w_q last dimension must be contiguous");
    TORCH_CHECK(w_scale.stride(0) == 1, "mm8_prequant_one: w_scale must be contiguous");
    TORCH_CHECK(y.stride(0) == 1, "mm8_prequant_one: y must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "mm8_prequant_one: x must be FP16");
    TORCH_CHECK(w_q.scalar_type() == torch::kInt8, "mm8_prequant_one: w_q must be int8");
    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat, "mm8_prequant_one: w_scale must be FP32");
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

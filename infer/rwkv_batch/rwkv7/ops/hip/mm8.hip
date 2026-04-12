#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

typedef at::Half fp16;

template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);
template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<fp16>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<fp16>(), y.stride(0));
        break;
    case c10::ScalarType::Float:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<float>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>(), y.stride(0));
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}
void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_one(
            N, M,
            x.data_ptr<fp16>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>());
        break;
    case c10::ScalarType::Float:
        cuda_mm8_one(
            N, M,
            x.data_ptr<float>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

using torch::Tensor;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_one", &mm8_one, "mm8 one");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
}

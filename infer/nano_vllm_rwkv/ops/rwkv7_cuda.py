import os
from pathlib import Path

import torch
import torch.library
from torch.utils.cpp_extension import load

_LOADED = False
_HEAD_SIZE = None
_EXT_NAME = "nanovllm_rwkv7_state_fwd_fp16"


def ensure_loaded(head_size: int):
    global _LOADED, _HEAD_SIZE
    if _LOADED:
        if _HEAD_SIZE != head_size:
            raise RuntimeError(f"RWKV7 CUDA extension was built with head_size={_HEAD_SIZE}, got {head_size}")
        return
    if hasattr(torch.ops, "nanovllm_rwkv7_state_fwd_fp16") and hasattr(torch.ops.nanovllm_rwkv7_state_fwd_fp16, "forward_one"):
        _LOADED = True
        _HEAD_SIZE = head_size
        return
    cur = Path(__file__).resolve().parent
    src_cpp = str(cur / "cuda" / "rwkv7_state_fwd_fp16.cpp")
    src_cu = str(cur / "cuda" / "rwkv7_state_fwd_fp16.cu")
    load(
        name=_EXT_NAME,
        sources=[src_cpp, src_cu],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}",
        ] + (["-Xptxas", "-O3"] if os.name != "nt" else []),
    )
    _LOADED = True
    _HEAD_SIZE = head_size


@torch.library.custom_op("nanovllm::rwkv7_one", mutates_args=("state_out",))
def rwkv7_one_op(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    c = r.numel()
    h = state_in.size(0)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.nanovllm_rwkv7_state_fwd_fp16.forward_one(
        1,
        c,
        h,
        state_in,
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


@rwkv7_one_op.register_fake
def _(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(r)


@torch.library.custom_op("nanovllm::rwkv7_one_batch", mutates_args=("state_out",))
def rwkv7_one_batch_op(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    batch, c = r.shape
    h = state_in.size(1)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.nanovllm_rwkv7_state_fwd_fp16.forward_one(
        batch,
        c,
        h,
        state_in.contiguous(),
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


@rwkv7_one_batch_op.register_fake
def _(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(r)


@torch.library.custom_op("nanovllm::rwkv7_seq", mutates_args=("state_out",))
def rwkv7_seq_op(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    t, c = r.shape
    h = state_in.size(0)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.nanovllm_rwkv7_state_fwd_fp16.forward_seq(
        1,
        t,
        c,
        h,
        state_in,
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


@rwkv7_seq_op.register_fake
def _(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(r)


@torch.library.custom_op("nanovllm::rwkv7_seq_batch", mutates_args=("state_out",))
def rwkv7_seq_batch_op(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    batch, t, c = r.shape
    h = state_in.size(1)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.nanovllm_rwkv7_state_fwd_fp16.forward_seq(
        batch,
        t,
        c,
        h,
        state_in.contiguous(),
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


@rwkv7_seq_batch_op.register_fake
def _(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(r)


def wkv7_one(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: int,
) -> torch.Tensor:
    # Inputs are [C], state is [H, N, N].
    elapsed = torch.tensor([elapsed_t], device=r.device, dtype=torch.int32)
    return rwkv7_one_op(state_in, state_out, r, w, k, v, a, b, elapsed)


def wkv7_one_batch(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return rwkv7_one_batch_op(state_in, state_out, r, w, k, v, a, b, elapsed_t)


def wkv7_seq(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: int,
) -> torch.Tensor:
    elapsed = torch.tensor([elapsed_t], device=r.device, dtype=torch.int32)
    return rwkv7_seq_op(state_in, state_out, r, w, k, v, a, b, elapsed)


def wkv7_seq_batch(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    return rwkv7_seq_batch_op(state_in, state_out, r, w, k, v, a, b, elapsed_t)


def cmix_one(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    x_k: torch.Tensor,
    key: torch.Tensor,
    val: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.nanovllm_rwkv7_state_fwd_fp16.cmix_one(
        x_0.contiguous(),
        x_1,
        x_k.contiguous(),
        key.contiguous(),
        val.contiguous(),
    )

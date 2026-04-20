import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def tmix_shift_batch_t1_kernel(
        x_ptr,
        prev_ptr,
        x_r_ptr,
        x_w_ptr,
        x_k_ptr,
        x_v_ptr,
        x_a_ptr,
        x_g_ptr,
        xr_ptr,
        xw_ptr,
        xk_ptr,
        xv_ptr,
        xa_ptr,
        xg_ptr,
        c_size,
        n_elements,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        cols = offsets % c_size

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        prev = tl.load(prev_ptr + offsets, mask=mask, other=0.0)
        xx = prev - x

        x_r = tl.load(x_r_ptr + cols, mask=mask, other=0.0)
        x_w = tl.load(x_w_ptr + cols, mask=mask, other=0.0)
        x_k = tl.load(x_k_ptr + cols, mask=mask, other=0.0)
        x_v = tl.load(x_v_ptr + cols, mask=mask, other=0.0)
        x_a = tl.load(x_a_ptr + cols, mask=mask, other=0.0)
        x_g = tl.load(x_g_ptr + cols, mask=mask, other=0.0)

        tl.store(xr_ptr + offsets, x + xx * x_r, mask=mask)
        tl.store(xw_ptr + offsets, x + xx * x_w, mask=mask)
        tl.store(xk_ptr + offsets, x + xx * x_k, mask=mask)
        tl.store(xv_ptr + offsets, x + xx * x_v, mask=mask)
        tl.store(xa_ptr + offsets, x + xx * x_a, mask=mask)
        tl.store(xg_ptr + offsets, x + xx * x_g, mask=mask)


    @triton.jit
    def cmix_shift_batch_t1_kernel(
        x_ptr,
        prev_ptr,
        x_k_ptr,
        out_ptr,
        c_size,
        n_elements,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        cols = offsets % c_size

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        prev = tl.load(prev_ptr + offsets, mask=mask, other=0.0)
        x_k = tl.load(x_k_ptr + cols, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + (prev - x) * x_k, mask=mask)


def _tmix_shift_batch_t1_impl(x, prev, x_r, x_w, x_k, x_v, x_a, x_g):
    if not TRITON_AVAILABLE or x.device.type != "cuda":
        xx = prev - x
        return torch.stack(
            (
                torch.addcmul(x, xx, x_r),
                torch.addcmul(x, xx, x_w),
                torch.addcmul(x, xx, x_k),
                torch.addcmul(x, xx, x_v),
                torch.addcmul(x, xx, x_a),
                torch.addcmul(x, xx, x_g),
            ),
            dim=0,
        )

    outputs = [torch.empty_like(x) for _ in range(6)]
    n_elements = x.numel()
    block_size = 512
    grid = (triton.cdiv(n_elements, block_size),)
    tmix_shift_batch_t1_kernel[grid](
        x,
        prev,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        x.shape[-1],
        n_elements,
        block_size=block_size,
    )
    return torch.stack(outputs, dim=0)


def _cmix_shift_batch_t1_impl(x, prev, x_k):
    if not TRITON_AVAILABLE or x.device.type != "cuda":
        return torch.addcmul(x, prev - x, x_k)

    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 512
    grid = (triton.cdiv(n_elements, block_size),)
    cmix_shift_batch_t1_kernel[grid](
        x,
        prev,
        x_k,
        out,
        x.shape[-1],
        n_elements,
        block_size=block_size,
    )
    return out


@torch.library.custom_op("mylib::TMIX_SHIFT_BATCH_T1_OP", mutates_args=())
def TMIX_SHIFT_BATCH_T1_OP(
    x: torch.Tensor,
    prev: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> torch.Tensor:
    return _tmix_shift_batch_t1_impl(x, prev, x_r, x_w, x_k, x_v, x_a, x_g)


@TMIX_SHIFT_BATCH_T1_OP.register_fake
def _(x, prev, x_r, x_w, x_k, x_v, x_a, x_g):
    return torch.empty((6, *x.shape), device=x.device, dtype=x.dtype)


@torch.library.custom_op("mylib::CMIX_SHIFT_BATCH_T1_OP", mutates_args=())
def CMIX_SHIFT_BATCH_T1_OP(
    x: torch.Tensor,
    prev: torch.Tensor,
    x_k: torch.Tensor,
) -> torch.Tensor:
    return _cmix_shift_batch_t1_impl(x, prev, x_k)


@CMIX_SHIFT_BATCH_T1_OP.register_fake
def _(x, prev, x_k):
    return torch.empty_like(x)

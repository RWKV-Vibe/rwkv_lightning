import torch.library
import os

ROCm_flag = torch.version.hip is not None
current_path = os.path.dirname(os.path.abspath(__file__))
########################################################################################################

from torch.utils.cpp_extension import load
HEAD_SIZE = 64
DTYPE = torch.half

############################################### Load kernels ###########################################
if ROCm_flag == True:
    load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/hip/rwkv7_state_fwd_fp16_op.hip", f"{current_path}/hip/rwkv7_state_fwd_fp16.hip"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics', f"-D_N_={HEAD_SIZE}"])
else:
    load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"] + (["-Xptxas -O3"] if os.name != "nt" else []))


class SPMV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vec, mat):
        D, C = mat.size()
        out = torch.zeros((C,), device=vec.device, dtype=DTYPE, requires_grad=False)
        torch.ops.rwkv7_state_fwd_fp16.spmv_forward(D, C, vec, mat, out)
        return out

@torch.library.custom_op("mylib::SPMV_OP", mutates_args=())
# @MyDisable
def SPMV_OP(vec:torch.Tensor, mat:torch.Tensor) -> torch.Tensor:
    return SPMV.apply(vec, mat)
@SPMV_OP.register_fake
def _(vec:torch.Tensor, mat:torch.Tensor) -> torch.Tensor:
    D, C = mat.size()
    return torch.zeros((C,), device=vec.device, dtype=DTYPE, requires_grad=False)


############################################### gems ###################################################
if ROCm_flag == False:
    try:
        # import flag_gems # type: ignore
        # import torch.ops.flag_gems.rwkv_mm_sparsity as rwkv_mm_sparsity # type: ignore
        rwkv_mm_sparsity = SPMV_OP
    except:
        print("flag_gems is not installed. Using triton kernel directly instead.")
        from .rwkv_mm_op_triton import rwkv_mm_sparsity
else:
    from .rwkv_mm_op_triton import rwkv_mm_sparsity

class WKV_7_ONE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            C = r.size()[0]
            H = C // HEAD_SIZE
            y = torch.empty((C,), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one(1, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_ONE_OP", mutates_args=())
def RWKV7_ONE_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_ONE.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_ONE_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

class WKV_7_SEQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq(1, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_SEQ_OP", mutates_args=())
def RWKV7_SEQ_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_SEQ.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_SEQ_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

class WKV_7_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_ONE_BATCH_OP", mutates_args=())
# @MyDisable
def RWKV7_ONE_BATCH_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_ONE_BATCH_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)


class WKV_7_SEQ_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y

@torch.library.custom_op("mylib::RWKV7_BATCH_OP", mutates_args=())
# @MyDisable
def RWKV7_BATCH_OP(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_SEQ_BATCH.apply(state, r, w, k, v, a, b, elapsed_t)
@RWKV7_BATCH_OP.register_fake
def _(state:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

########################################################################################################
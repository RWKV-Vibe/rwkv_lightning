########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

from typing import List
import os
current_path = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.library
from torch.library import register_fake
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F
from .rwkv_pointwise_op import CMIX_SHIFT_BATCH_T1_OP, TMIX_SHIFT_BATCH_T1_OP

if os.getenv("RWKV_USE_JIT", "0") == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = nn.Module
    MyFunction = torch.compile(mode='max-autotune-no-cudagraphs')
    MyStatic = torch.compile(mode='max-autotune-no-cudagraphs')
MyDisable = torch.compiler.disable

DTYPE = torch.half
ROCm_flag = torch.version.hip is not None

########################################################################################################

from torch.utils.cpp_extension import load
HEAD_SIZE = 64

if ROCm_flag == True:
    load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/hip/rwkv7_state_fwd_fp16_op.hip", f"{current_path}/hip/rwkv7_state_fwd_fp16.hip"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics', f"-D_N_={HEAD_SIZE}"])
else:
    load(name="rwkv7_state_fwd_fp16", sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_path}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
                    verbose=True,
                    extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", "-gencode=arch=compute_120,code=sm_120", "-gencode=arch=compute_120,code=compute_120"] + (["-Xptxas -O3"] if os.name != "nt" else []))

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
# @MyDisable
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
# @MyDisable
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


class WKV_7_ONE_BATCH_PAGED(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_pool, page_table, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_one_paged(B, C, H, state_pool, page_table, r, w, k, v, a, b, y, elapsed_t)
            return y


@torch.library.custom_op("mylib::RWKV7_ONE_BATCH_PAGED_OP", mutates_args=())
def RWKV7_ONE_BATCH_PAGED_OP(state_pool:torch.Tensor, page_table:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_ONE_BATCH_PAGED.apply(state_pool, page_table, r, w, k, v, a, b, elapsed_t)


@RWKV7_ONE_BATCH_PAGED_OP.register_fake
def _(state_pool:torch.Tensor, page_table:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)


class WKV_7_SEQ_BATCH_PAGED(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_pool, page_table, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward_seq_paged(B, T, C, H, state_pool, page_table, r, w, k, v, a, b, y, elapsed_t)
            return y


@torch.library.custom_op("mylib::RWKV7_BATCH_PAGED_OP", mutates_args=())
def RWKV7_BATCH_PAGED_OP(state_pool:torch.Tensor, page_table:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return WKV_7_SEQ_BATCH_PAGED.apply(state_pool, page_table, r, w, k, v, a, b, elapsed_t)


@RWKV7_BATCH_PAGED_OP.register_fake
def _(state_pool:torch.Tensor, page_table:torch.Tensor, r:torch.Tensor, w:torch.Tensor, k:torch.Tensor, v:torch.Tensor, a:torch.Tensor, b:torch.Tensor, elapsed_t:torch.Tensor) -> torch.Tensor:
    return torch.empty_like(r)

########################################################################################################

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = 64
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            kk = k.split('.')
            # if kk[0] == 'blocks' and int(kk[1]) >= 10:
            #     continue
            if 'att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or 'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or 'ffn.value.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        print(args)
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None, None]
        if bsz >= 1:
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
            state[2] = torch.zeros((bsz,), dtype=torch.int32, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
            state[2] = torch.zeros((), dtype=torch.int32, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False): # will modify state in-place
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                x = self.z['emb.weight'][idx[0]]
                return self.forward_one(x, state)
        elif type(idx) is torch.Tensor:
            return self.forward_one(idx, state)
        else:
            x = self.z['emb.weight'][idx]
            return self.forward_one(x, state)
        
    def forward_batch(self, tokens, state, full_output=False): # will modify state in-place
        if torch.is_tensor(tokens):
            return self.forward_batch_same_length(tokens, state, full_output)

        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            return self.forward_batch_same_length(tokens, state, full_output)

        bsz = len(tokens)
        pos = [0] * bsz

        if full_output == False:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda") for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] for i in active]
            batch_state = [state[0][:,:,active],state[1][:,active], state[2][active]] # state[0]=[Layer][2][Bsz][C]    state[1]=[Layer][Bsz][H][N][N]
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)
            for k, i in enumerate(active):
                if full_output == False:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                state[0][:,:,i] = batch_state[0][:,:,k]
                state[1][:,i] = batch_state[1][:,k]
                state[2][i] = batch_state[2][k]
                pos[i] += step
        return out

    def forward_batch_same_length(self, tokens, state, full_output=False):
        if torch.is_tensor(tokens):
            seq_len = int(tokens.shape[1]) if tokens.ndim >= 2 else 1
        else:
            assert type(tokens) is list
            assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
            seq_len = len(tokens[0]) if tokens else 0

        if seq_len > 128:
            return self.forward_seq_batch_chunk(tokens, state, chunk_len=128, full_output=full_output)
        return self.forward_seq_batch(tokens, state, full_output)

    @MyFunction
    def forward_one(self, x:torch.Tensor, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += 1
            return x
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += len(idx)
            return x
        
    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            idx_tensor = idxs.to(device=z['emb.weight'].device, dtype=torch.long) if torch.is_tensor(idxs) else torch.tensor(idxs, device=z['emb.weight'].device)
            x = z['emb.weight'][idx_tensor]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += int(idx_tensor.shape[1])
            return x
    @MyFunction
    def forward_seq_batch_chunk(self, idxs: List[List[int]], state: List[torch.Tensor], chunk_len: int = 64, full_output: bool = False):
        with torch.no_grad():
            z = self.z
            device = z['emb.weight'].device
            
            # 转换为 Tensor，形状为 [Batch, Total_Seq_Len]
            full_idxs = idxs.to(device=device, dtype=torch.long) if torch.is_tensor(idxs) else torch.tensor(idxs, device=device)
            batch_size, total_len = full_idxs.size()
            
            all_outputs = []

            # 按 chunk_len 进行循环处理
            for start in range(0, total_len, chunk_len):
                end = min(start + chunk_len, total_len)
                # 截取当前的 chunk
                chunk_idxs = full_idxs[:, start:end]
                
                x = z['emb.weight'][chunk_idxs] 
                v_first = torch.empty_like(x)

                for i in range(self.n_layer):
                    bbb = f'blocks.{i}.'
                    att = f'blocks.{i}.att.'
                    ffn = f'blocks.{i}.ffn.'

                    # TimeMix
                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                    
                    xx, v_first = RWKV_x070_TMix_seq_batch(
                        i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2]
                    )
                    x = x + xx

                    # ChannelMix
                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                    xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                    x = x + xx

                state[2] += (end - start)
                
                if full_output:
                    all_outputs.append(x)
                else:
                    if end == total_len:
                        all_outputs.append(x[:, -1, :])

            x = torch.cat(all_outputs, dim=1) if full_output else all_outputs[0]
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            
            return x

    def forward_batch_paged(self, tokens, state_pool, page_table, full_output=False):
        if torch.is_tensor(tokens):
            return self.forward_batch_same_length_paged(tokens, state_pool, page_table, full_output)

        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            return self.forward_batch_same_length_paged(tokens, state_pool, page_table, full_output)

        bsz = len(tokens)
        pos = [0] * bsz

        if full_output == False:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda") for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] for i in active]
            batch_pages = page_table[active]
            new_out = self.forward_batch_same_length_paged(batch_tokens, state_pool, batch_pages, full_output)
            for k, i in enumerate(active):
                if full_output == False:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                pos[i] += step
        return out

    def forward_batch_same_length_paged(self, tokens, state_pool, page_table, full_output=False):
        if torch.is_tensor(tokens):
            seq_len = int(tokens.shape[1]) if tokens.ndim >= 2 else 1
        else:
            assert type(tokens) is list
            assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
            seq_len = len(tokens[0]) if tokens else 0

        if seq_len > 128:
            return self.forward_seq_batch_chunk_paged(tokens, state_pool, page_table, chunk_len=128, full_output=full_output)
        return self.forward_seq_batch_paged(tokens, state_pool, page_table, full_output)

    @MyFunction
    def forward_seq_batch_paged(self, idxs, state_pool: List[torch.Tensor], page_table: torch.Tensor, full_output: bool = False):
        with torch.no_grad():
            z = self.z
            idx_tensor = idxs.to(device=z['emb.weight'].device, dtype=torch.long) if torch.is_tensor(idxs) else torch.tensor(idxs, device=z['emb.weight'].device)
            page_idx = page_table.to(device=z['emb.weight'].device, dtype=torch.int32)
            x = z['emb.weight'][idx_tensor]

            state0_batch = state_pool[0][:, :, page_idx.long(), :].contiguous()
            elapsed_t = state_pool[2][page_idx.long()].contiguous()
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch_paged(i, self.n_head, self.head_size, xx, state0_batch[i], v_first, state_pool[1][i], page_idx,
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], elapsed_t)
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = RWKV_x070_CMix_seq_batch(xx, state0_batch[i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            state_pool[0][:, :, page_idx.long(), :] = state0_batch
            state_pool[2][page_idx.long()] = elapsed_t + int(idx_tensor.shape[1])

            if not full_output:
                x = x[:, -1, :]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            return x

    @MyFunction
    def forward_seq_batch_chunk_paged(self, idxs, state_pool: List[torch.Tensor], page_table: torch.Tensor, chunk_len: int = 64, full_output: bool = False):
        with torch.no_grad():
            z = self.z
            device = z['emb.weight'].device
            full_idxs = idxs.to(device=device, dtype=torch.long) if torch.is_tensor(idxs) else torch.tensor(idxs, device=device)
            page_idx = page_table.to(device=device, dtype=torch.int32)
            batch_size, total_len = full_idxs.size()

            state0_batch = state_pool[0][:, :, page_idx.long(), :].contiguous()
            elapsed_t = state_pool[2][page_idx.long()].contiguous()
            all_outputs = []

            for start in range(0, total_len, chunk_len):
                end = min(start + chunk_len, total_len)
                chunk_idxs = full_idxs[:, start:end]
                x = z['emb.weight'][chunk_idxs]
                v_first = torch.empty_like(x)

                for i in range(self.n_layer):
                    bbb = f'blocks.{i}.'
                    att = f'blocks.{i}.att.'
                    ffn = f'blocks.{i}.ffn.'

                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                    xx, v_first = RWKV_x070_TMix_seq_batch_paged(
                        i, self.n_head, self.head_size, xx, state0_batch[i], v_first, state_pool[1][i], page_idx,
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'], elapsed_t
                    )
                    x = x + xx

                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                    xx = RWKV_x070_CMix_seq_batch(xx, state0_batch[i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                    x = x + xx

                elapsed_t = elapsed_t + (end - start)

                if full_output:
                    all_outputs.append(x)
                elif end == total_len:
                    all_outputs.append(x[:, -1, :])

            state_pool[0][:, :, page_idx.long(), :] = state0_batch
            state_pool[2][page_idx.long()] = elapsed_t

            x = torch.cat(all_outputs, dim=1) if full_output else all_outputs[0]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            return x

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = (
        torch.addcmul(x, xx, x_r),
        torch.addcmul(x, xx, x_w),
        torch.addcmul(x, xx, x_k),
        torch.addcmul(x, xx, x_v),
        torch.addcmul(x, xx, x_a),
        torch.addcmul(x, xx, x_g),
    )

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = torch.addcmul(k, k, (a - 1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = torch.addcmul(v, v_first - v, torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)))

    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kka, elapsed_t) # !!! using CUDA to modify state in-place !!! (faster too)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)
    xx = torch.addcmul(
        xx.view(H, N),
        (r * k * r_k).view(H, N).sum(dim=-1, keepdim=True),
        v.view(H, N),
    ).view(H * N)
    return F.linear(xx * g, O_), v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = (
        torch.addcmul(x, xx, x_r),
        torch.addcmul(x, xx, x_w),
        torch.addcmul(x, xx, x_k),
        torch.addcmul(x, xx, x_v),
        torch.addcmul(x, xx, x_a),
        torch.addcmul(x, xx, x_g),
    )

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = torch.addcmul(k, k, (a - 1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = torch.addcmul(v, v_first - v, torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)))

    xx = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kka, elapsed_t)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = torch.addcmul(
        xx.view(T, H, N),
        (r * k * r_k).view(T, H, N).sum(dim=-1, keepdim=True),
        v.view(T, H, N),
    ).view(T, H * N)
    return F.linear(xx * g, O_), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    if T == 1:
        x_single = x[:, 0, :]
        shifted = TMIX_SHIFT_BATCH_T1_OP(
            x_single, x_prev[0], x_r, x_w, x_k, x_v, x_a, x_g
        )
        x_prev[0] = x_single
        xr, xw, xk, xv, xa, xg = shifted[0], shifted[1], shifted[2], shifted[3], shifted[4], shifted[5]

        r = F.linear(xr, R_)
        w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
        k = F.linear(xk, K_)
        v = F.linear(xv, V_)
        a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
        g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

        kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
        k = torch.addcmul(k, k, (a - 1) * k_a)
        kka = kk * a

        if layer_id == 0:
            v_first = v.unsqueeze(1)
        else:
            v = torch.addcmul(
                v,
                v_first[:, 0, :] - v,
                torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)),
            )

        xx = RWKV7_ONE_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t)
        xx = F.group_norm(xx.view(B, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B, H * N)
        xx = torch.addcmul(
            xx.view(B, H, N),
            (r * k * r_k).view(B, H, N).sum(dim=-1, keepdim=True),
            v.view(B, H, N),
        ).view(B, H * N)
        return F.linear(xx * g, O_).unsqueeze(1), v_first

    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = (
        torch.addcmul(x, xx, x_r),
        torch.addcmul(x, xx, x_w),
        torch.addcmul(x, xx, x_k),
        torch.addcmul(x, xx, x_v),
        torch.addcmul(x, xx, x_a),
        torch.addcmul(x, xx, x_g),
    )

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = torch.addcmul(k, k, (a - 1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = torch.addcmul(v, v_first - v, torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)))

    # if T == 1:
    #     vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
    #     ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
    #     state = state * w.view(B,H,1,N) + state @ ab + vk
    #     xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1)).view(B*T,H*N)
    # else:
    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = torch.addcmul(
        xx.view(B, T, H, N),
        (r * k * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True),
        v.view(B, T, H, N),
    ).view(B, T, H * N)
    return F.linear(xx * g, O_), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch_paged(layer_id: int, H:int, N:int, x, x_prev, v_first, state_pool, page_table, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    if T == 1:
        x_single = x[:, 0, :]
        shifted = TMIX_SHIFT_BATCH_T1_OP(
            x_single, x_prev[0], x_r, x_w, x_k, x_v, x_a, x_g
        )
        x_prev[0] = x_single
        xr, xw, xk, xv, xa, xg = shifted[0], shifted[1], shifted[2], shifted[3], shifted[4], shifted[5]

        r = F.linear(xr, R_)
        w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
        k = F.linear(xk, K_)
        v = F.linear(xv, V_)
        a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
        g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

        kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
        k = torch.addcmul(k, k, (a - 1) * k_a)
        kka = kk * a

        if layer_id == 0:
            v_first = v.unsqueeze(1)
        else:
            v = torch.addcmul(
                v,
                v_first[:, 0, :] - v,
                torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)),
            )

        xx = RWKV7_ONE_BATCH_PAGED_OP(state_pool, page_table, r, w, k, v, -kk, kka, elapsed_t)
        xx = F.group_norm(xx.view(B, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B, H * N)
        xx = torch.addcmul(
            xx.view(B, H, N),
            (r * k * r_k).view(B, H, N).sum(dim=-1, keepdim=True),
            v.view(B, H, N),
        ).view(B, H * N)
        return F.linear(xx * g, O_).unsqueeze(1), v_first

    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = (
        torch.addcmul(x, xx, x_r),
        torch.addcmul(x, xx, x_w),
        torch.addcmul(x, xx, x_k),
        torch.addcmul(x, xx, x_v),
        torch.addcmul(x, xx, x_a),
        torch.addcmul(x, xx, x_g),
    )

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = torch.addcmul(k, k, (a - 1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = torch.addcmul(v, v_first - v, torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0)))

    xx = RWKV7_BATCH_PAGED_OP(state_pool, page_table, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)
    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = torch.addcmul(
        xx.view(B, T, H, N),
        (r * k * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True),
        v.view(B, T, H, N),
    ).view(B, T, H * N)
    return F.linear(xx * g, O_), v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = torch.addcmul(x, xx, x_k)
    k = torch.relu(F.linear(k, K_)).square_()
    kv = rwkv_mm_sparsity(k, V_)
    # kv = k @ V_
    # kv = SPMV_OP(k, V_)
    return kv

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = torch.addcmul(x, xx, x_k)
    k = torch.relu(F.linear(k, K_)).square_()
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_ # F.linear(k, V_)

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    if x.shape[1] == 1:
        x_single = x[:, 0, :]
        k = CMIX_SHIFT_BATCH_T1_OP(x_single, x_prev[1], x_k)
        x_prev[1] = x_single
        k = torch.relu(F.linear(k, K_)).square_()
        return (k @ V_).unsqueeze(1)

    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = torch.addcmul(x, xx, x_k)
    k = torch.relu(F.linear(k, K_)).square_()
    return k @ V_ # F.linear(k, V_)

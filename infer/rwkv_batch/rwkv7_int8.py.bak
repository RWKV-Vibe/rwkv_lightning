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

# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F
from .mm_int8_kernel import linear_i8, quantize_mm8_for_linear

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# MyFunction = torch.compile()
# MyStatic = torch.compile()
MyDisable = torch.compiler.disable
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop
# MyDisable = __nop

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
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        print(args)
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        # Quantize on CPU first to avoid peak CUDA memory during init.
        self._quantize_large_linear_weights()
        self._move_model_to_cuda()

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def _quantize_one_linear_weight(self, key: str):
        z = self.z
        w_q, mx, rx, my, ry = quantize_mm8_for_linear(z[key])
        z[key + '.q'] = w_q.contiguous()
        z[key + '.mx'] = mx.contiguous()
        z[key + '.rx'] = rx.contiguous()
        z[key + '.my'] = my.contiguous()
        z[key + '.ry'] = ry.contiguous()
        # Original fp16 weight is no longer used after int8 path is enabled.
        del z[key]

    def _quantize_large_linear_weights(self):
        z = self.z
        for i in range(self.n_layer):
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'
            self._quantize_one_linear_weight(att + 'receptance.weight')
            self._quantize_one_linear_weight(att + 'key.weight')
            self._quantize_one_linear_weight(att + 'value.weight')
            self._quantize_one_linear_weight(att + 'output.weight')
            self._quantize_one_linear_weight(ffn + 'key.weight')
        self._quantize_one_linear_weight('head.weight')

    def _move_model_to_cuda(self):
        z = self.z
        for k in list(z.keys()):
            t = z[k]
            if not torch.is_tensor(t):
                continue
            if t.dtype == torch.uint8:
                z[k] = t.to(device="cuda").contiguous()
            elif t.is_floating_point():
                z[k] = t.to(device="cuda", dtype=DTYPE).contiguous()
            else:
                z[k] = t.to(device="cuda").contiguous()

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
        assert type(tokens) is list
        assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
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
                    z[att+'receptance.weight.q'], z[att+'receptance.weight.mx'], z[att+'receptance.weight.rx'], z[att+'receptance.weight.my'], z[att+'receptance.weight.ry'],
                    z[att+'key.weight.q'], z[att+'key.weight.mx'], z[att+'key.weight.rx'], z[att+'key.weight.my'], z[att+'key.weight.ry'],
                    z[att+'value.weight.q'], z[att+'value.weight.mx'], z[att+'value.weight.rx'], z[att+'value.weight.my'], z[att+'value.weight.ry'],
                    z[att+'output.weight.q'], z[att+'output.weight.mx'], z[att+'output.weight.rx'], z[att+'output.weight.my'], z[att+'output.weight.ry'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(
                    xx, state[0][i], z[ffn+'x_k'],
                    z[ffn+'key.weight.q'], z[ffn+'key.weight.mx'], z[ffn+'key.weight.rx'], z[ffn+'key.weight.my'], z[ffn+'key.weight.ry'],
                    z[ffn+'value.weight']
                )
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = linear_i8(x, z['head.weight.q'], z['head.weight.mx'], z['head.weight.rx'], z['head.weight.my'], z['head.weight.ry'])
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
                    z[att+'receptance.weight.q'], z[att+'receptance.weight.mx'], z[att+'receptance.weight.rx'], z[att+'receptance.weight.my'], z[att+'receptance.weight.ry'],
                    z[att+'key.weight.q'], z[att+'key.weight.mx'], z[att+'key.weight.rx'], z[att+'key.weight.my'], z[att+'key.weight.ry'],
                    z[att+'value.weight.q'], z[att+'value.weight.mx'], z[att+'value.weight.rx'], z[att+'value.weight.my'], z[att+'value.weight.ry'],
                    z[att+'output.weight.q'], z[att+'output.weight.mx'], z[att+'output.weight.rx'], z[att+'output.weight.my'], z[att+'output.weight.ry'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(
                    xx, state[0][i], z[ffn+'x_k'],
                    z[ffn+'key.weight.q'], z[ffn+'key.weight.mx'], z[ffn+'key.weight.rx'], z[ffn+'key.weight.my'], z[ffn+'key.weight.ry'],
                    z[ffn+'value.weight']
                )
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = linear_i8(x, z['head.weight.q'], z['head.weight.mx'], z['head.weight.rx'], z['head.weight.my'], z['head.weight.ry'])
            state[2] += len(idx)
            return x
        
    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][torch.tensor(idxs, device=z['emb.weight'].device)]

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
                    z[att+'receptance.weight.q'], z[att+'receptance.weight.mx'], z[att+'receptance.weight.rx'], z[att+'receptance.weight.my'], z[att+'receptance.weight.ry'],
                    z[att+'key.weight.q'], z[att+'key.weight.mx'], z[att+'key.weight.rx'], z[att+'key.weight.my'], z[att+'key.weight.ry'],
                    z[att+'value.weight.q'], z[att+'value.weight.mx'], z[att+'value.weight.rx'], z[att+'value.weight.my'], z[att+'value.weight.ry'],
                    z[att+'output.weight.q'], z[att+'output.weight.mx'], z[att+'output.weight.rx'], z[att+'output.weight.my'], z[att+'output.weight.ry'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(
                    xx, state[0][i], z[ffn+'x_k'],
                    z[ffn+'key.weight.q'], z[ffn+'key.weight.mx'], z[ffn+'key.weight.rx'], z[ffn+'key.weight.my'], z[ffn+'key.weight.ry'],
                    z[ffn+'value.weight']
                )
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = linear_i8(x, z['head.weight.q'], z['head.weight.mx'], z['head.weight.rx'], z['head.weight.my'], z['head.weight.ry'])
            state[2] += len(idxs[0])
            return x
    @MyFunction
    def forward_seq_batch_chunk(self, idxs: List[List[int]], state: List[torch.Tensor], chunk_len: int = 64, full_output: bool = False):
        with torch.no_grad():
            z = self.z
            device = z['emb.weight'].device
            
            # 转换为 Tensor，形状为 [Batch, Total_Seq_Len]
            full_idxs = torch.tensor(idxs, device=device)
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
                        z[att+'receptance.weight.q'], z[att+'receptance.weight.mx'], z[att+'receptance.weight.rx'], z[att+'receptance.weight.my'], z[att+'receptance.weight.ry'],
                        z[att+'key.weight.q'], z[att+'key.weight.mx'], z[att+'key.weight.rx'], z[att+'key.weight.my'], z[att+'key.weight.ry'],
                        z[att+'value.weight.q'], z[att+'value.weight.mx'], z[att+'value.weight.rx'], z[att+'value.weight.my'], z[att+'value.weight.ry'],
                        z[att+'output.weight.q'], z[att+'output.weight.mx'], z[att+'output.weight.rx'], z[att+'output.weight.my'], z[att+'output.weight.ry'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2]
                    )
                    x = x + xx

                    # ChannelMix
                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                    xx = RWKV_x070_CMix_seq_batch(
                        xx, state[0][i], z[ffn+'x_k'],
                        z[ffn+'key.weight.q'], z[ffn+'key.weight.mx'], z[ffn+'key.weight.rx'], z[ffn+'key.weight.my'], z[ffn+'key.weight.ry'],
                        z[ffn+'value.weight']
                    )
                    x = x + xx

                state[2] += (end - start)
                
                if full_output:
                    all_outputs.append(x)
                else:
                    if end == total_len:
                        all_outputs.append(x[:, -1, :])

            x = torch.cat(all_outputs, dim=1) if full_output else all_outputs[0]
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = linear_i8(x, z['head.weight.q'], z['head.weight.mx'], z['head.weight.rx'], z['head.weight.my'], z['head.weight.ry'])
            
            return x

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k,
                       R_q, R_mx, R_rx, R_my, R_ry, K_q, K_mx, K_rx, K_my, K_ry, V_q, V_mx, V_rx, V_my, V_ry, O_q, O_mx, O_rx, O_my, O_ry,
                       ln_w, ln_b, elapsed_t):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = linear_i8(xr, R_q, R_mx, R_rx, R_my, R_ry)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = linear_i8(xk, K_q, K_mx, K_rx, K_my, K_ry)
    v = linear_i8(xv, V_q, V_mx, V_rx, V_my, V_ry)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kka, elapsed_t) # !!! using CUDA to modify state in-place !!! (faster too)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return linear_i8((xx * g), O_q, O_mx, O_rx, O_my, O_ry), v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k,
                       R_q, R_mx, R_rx, R_my, R_ry, K_q, K_mx, K_rx, K_my, K_ry, V_q, V_mx, V_rx, V_my, V_ry, O_q, O_mx, O_rx, O_my, O_ry,
                       ln_w, ln_b, elapsed_t):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = linear_i8(xr, R_q, R_mx, R_rx, R_my, R_ry)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = linear_i8(xk, K_q, K_mx, K_rx, K_my, K_ry)
    v = linear_i8(xv, V_q, V_mx, V_rx, V_my, V_ry)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kka, elapsed_t)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return linear_i8((xx * g), O_q, O_mx, O_rx, O_my, O_ry), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k,
                             R_q, R_mx, R_rx, R_my, R_ry, K_q, K_mx, K_rx, K_my, K_ry, V_q, V_mx, V_rx, V_my, V_ry, O_q, O_mx, O_rx, O_my, O_ry,
                             ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = linear_i8(xr, R_q, R_mx, R_rx, R_my, R_ry)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = linear_i8(xk, K_q, K_mx, K_rx, K_my, K_ry)
    v = linear_i8(xv, V_q, V_mx, V_rx, V_my, V_ry)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    # if T == 1:
    #     vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
    #     ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
    #     state = state * w.view(B,H,1,N) + state @ ab + vk
    #     xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1)).view(B*T,H*N)
    # else:
    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return linear_i8((xx * g), O_q, O_mx, O_rx, O_my, O_ry), v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_q, K_mx, K_rx, K_my, K_ry, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(linear_i8(k, K_q, K_mx, K_rx, K_my, K_ry)) ** 2
    kv = rwkv_mm_sparsity(k, V_)
    # kv = k @ V_
    # kv = SPMV_OP(k, V_)
    return kv

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_q, K_mx, K_rx, K_my, K_ry, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(linear_i8(k, K_q, K_mx, K_rx, K_my, K_ry)) ** 2
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_ # F.linear(k, V_)

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_q, K_mx, K_rx, K_my, K_ry, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(linear_i8(k, K_q, K_mx, K_rx, K_my, K_ry)) ** 2
    return k @ V_ # F.linear(k, V_)

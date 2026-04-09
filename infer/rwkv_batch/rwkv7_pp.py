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

MyModule = nn.Module
def MyFunction(fn):
    return fn
def MyStatic(fn):
    return fn
# MyModule = torch.jit.ScriptModule
# MyFunction = torch.jit.script_method
# MyStatic = torch.jit.script
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
    def __init__(self, args, devices):
        super().__init__()
        if not devices:
            raise ValueError("pp devices must be a non-empty list")
        if not torch.cuda.is_available():
            raise RuntimeError("pipeline parallelism requires CUDA")

        self.args = args
        args.head_size = 64
        self.eval()

        self.stage_devices = [self._normalize_device(device_id) for device_id in devices]
        self.stage_device_names = [str(device) for device in self.stage_devices]
        self.emb_device = self.stage_devices[0]
        self.out_device = self.stage_devices[-1]
        self.sample_device = self.out_device
        self.state_restore_device = list(self.stage_device_names)

        self.z = torch.load(args.MODEL_NAME + ".pth", map_location="cpu")
        z = self.z
        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        max_layer = -1
        for key in list(z.keys()):
            key_parts = key.split(".")
            if (
                "att.g1" in key
                or "att.g2" in key
                or "att.a1" in key
                or "att.a2" in key
                or "att.w1" in key
                or "att.w2" in key
                or "att.v1" in key
                or "att.v2" in key
                or "ffn.value.weight" in key
            ):
                z[key] = z[key].t()
            z[key] = z[key].squeeze()
            if key.endswith("att.r_k"):
                z[key] = z[key].flatten()
            if key_parts[0] == "blocks":
                max_layer = max(max_layer, int(key_parts[1]))

        args.n_layer = max_layer + 1
        self.n_layer, self.n_embd = args.n_layer, args.n_embd
        self.layer_ranges = self._build_layer_ranges(args.n_layer, len(self.stage_devices))
        self.layer_to_stage = {}
        for stage_idx, (start_layer, end_layer) in enumerate(self.layer_ranges):
            for layer_id in range(start_layer, end_layer):
                self.layer_to_stage[layer_id] = stage_idx

        for key in list(z.keys()):
            target_device = self._device_for_key(key)
            z[key] = z[key].to(dtype=DTYPE, device=target_device).contiguous()

        z["emb.weight"] = F.layer_norm(
            z["emb.weight"],
            (args.n_embd,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"],
        )
        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

        print(args)
        print(f"[PP] devices={self.stage_device_names} layer_ranges={self.layer_ranges}")

    @staticmethod
    def _normalize_device(device_id):
        if isinstance(device_id, torch.device):
            return device_id
        if isinstance(device_id, int):
            return torch.device(f"cuda:{device_id}")
        if isinstance(device_id, str):
            device_id = device_id.strip()
            if device_id.isdigit():
                return torch.device(f"cuda:{device_id}")
            return torch.device(device_id)
        raise TypeError(f"unsupported device id: {device_id!r}")

    @staticmethod
    def _build_layer_ranges(num_layers, num_stages):
        base, remainder = divmod(num_layers, num_stages)
        ranges = []
        start = 0
        for stage_idx in range(num_stages):
            width = base + (1 if stage_idx < remainder else 0)
            end = start + width
            ranges.append((start, end))
            start = end
        return ranges

    def _device_for_key(self, key):
        if key.startswith("blocks."):
            layer_id = int(key.split(".")[1])
            return self.stage_devices[self.layer_to_stage[layer_id]]
        if key.startswith("ln_out.") or key.startswith("head."):
            return self.out_device
        return self.emb_device

    def _move_to_stage(self, tensor, stage_idx):
        device = self.stage_devices[stage_idx]
        if tensor.device == device:
            return tensor
        return tensor.to(device, non_blocking=True)

    def _increment_elapsed(self, state, amount):
        for counter in state[2]:
            counter.add_(amount)

    def _select_batch_state(self, state, indices):
        batch_indices = []
        for stage_idx, device in enumerate(self.stage_devices):
            batch_indices.append(
                torch.tensor(indices, device=device, dtype=torch.long)
            )

        return [
            [
                shard.index_select(2, batch_indices[stage_idx])
                for stage_idx, shard in enumerate(state[0])
            ],
            [
                shard.index_select(1, batch_indices[stage_idx])
                for stage_idx, shard in enumerate(state[1])
            ],
            [
                shard.index_select(0, batch_indices[stage_idx])
                for stage_idx, shard in enumerate(state[2])
            ],
            list(state[3]),
        ]

    def _scatter_batch_state(self, state, batch_state, indices):
        if not indices:
            return state
        for stage_idx, device in enumerate(self.stage_devices):
            batch_indices = torch.tensor(indices, device=device, dtype=torch.long)
            state[0][stage_idx][:, :, batch_indices, :] = batch_state[0][stage_idx]
            state[1][stage_idx][:, batch_indices, :, :, :] = batch_state[1][stage_idx]
            state[2][stage_idx][batch_indices] = batch_state[2][stage_idx]
        return state

    def reset_state_slot(self, state, slot):
        for stage_idx in range(len(self.stage_devices)):
            state[0][stage_idx][:, :, slot, :].zero_()
            state[1][stage_idx][:, slot, :, :, :].zero_()
            state[2][stage_idx][slot].zero_()
        return state

    def remove_state_slots(self, state, slots):
        if not slots:
            return state
        total_slots = state[2][0].shape[0]
        keep_slots = [slot for slot in range(total_slots) if slot not in set(slots)]

        for stage_idx, device in enumerate(self.stage_devices):
            index = torch.tensor(keep_slots, device=device, dtype=torch.long)
            state[0][stage_idx] = state[0][stage_idx].index_select(2, index)
            state[1][stage_idx] = state[1][stage_idx].index_select(1, index)
            state[2][stage_idx] = state[2][stage_idx].index_select(0, index)
        return state

    def generate_zero_state(self, bsz):
        args = self.args
        state0 = []
        state1 = []
        state2 = []
        n_head = args.n_embd // args.head_size

        for stage_idx, (start_layer, end_layer) in enumerate(self.layer_ranges):
            local_layers = end_layer - start_layer
            device = self.stage_devices[stage_idx]
            if bsz >= 1:
                state0.append(
                    torch.zeros(
                        (local_layers, 2, bsz, args.n_embd),
                        dtype=DTYPE,
                        requires_grad=False,
                        device=device,
                    )
                )
                state1.append(
                    torch.zeros(
                        (local_layers, bsz, n_head, args.head_size, args.head_size),
                        dtype=DTYPE,
                        requires_grad=False,
                        device=device,
                    )
                )
                state2.append(
                    torch.zeros(
                        (bsz,),
                        dtype=torch.int32,
                        requires_grad=False,
                        device=device,
                    )
                )
            else:
                state0.append(
                    torch.zeros(
                        (local_layers, 2, args.n_embd),
                        dtype=DTYPE,
                        requires_grad=False,
                        device=device,
                    )
                )
                state1.append(
                    torch.zeros(
                        (local_layers, n_head, args.head_size, args.head_size),
                        dtype=DTYPE,
                        requires_grad=False,
                        device=device,
                    )
                )
                state2.append(
                    torch.zeros(
                        (),
                        dtype=torch.int32,
                        requires_grad=False,
                        device=device,
                    )
                )
        return [state0, state1, state2, list(self.stage_device_names)]

    def forward(self, idx, state, full_output=False):
        if isinstance(idx, list):
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            x = self.z["emb.weight"][idx[0]]
            return self.forward_one(x, state)

        if torch.is_tensor(idx):
            if idx.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                idx = idx.to(self.emb_device)
                x = self.z["emb.weight"][idx]
                if x.dim() == 1:
                    return self.forward_one(x, state)
                if x.dim() == 2:
                    return self._forward_hidden(x, state, "seq", full_output)
                if x.dim() == 3:
                    return self._forward_hidden(x, state, "seq_batch", full_output)
            if idx.dim() == 1:
                return self.forward_one(idx, state)
            if idx.dim() == 2:
                return self._forward_hidden(idx, state, "seq", full_output)
            if idx.dim() == 3:
                return self._forward_hidden(idx, state, "seq_batch", full_output)
            raise ValueError(f"unsupported tensor input shape: {tuple(idx.shape)}")

        x = self.z["emb.weight"][idx]
        return self.forward_one(x, state)

    def forward_batch(self, tokens, state, full_output=False):
        assert isinstance(tokens, list)
        lengths = [len(item) for item in tokens]
        if len(set(lengths)) == 1:
            return self.forward_batch_same_length(tokens, state, full_output)

        batch_size = len(tokens)
        pos = [0] * batch_size
        if not full_output:
            out = torch.empty(
                (batch_size, self.args.vocab_size),
                dtype=DTYPE,
                requires_grad=False,
                device=self.out_device,
            )
        else:
            out = [
                torch.empty(
                    (0, self.args.vocab_size),
                    dtype=DTYPE,
                    requires_grad=False,
                    device=self.out_device,
                )
                for _ in range(batch_size)
            ]

        while True:
            active = [idx for idx in range(batch_size) if pos[idx] < lengths[idx]]
            if not active:
                break
            step = min(lengths[idx] - pos[idx] for idx in active)
            batch_tokens = [tokens[idx][pos[idx] : pos[idx] + step] for idx in active]
            batch_state = self._select_batch_state(state, active)
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)

            for local_idx, batch_idx in enumerate(active):
                if not full_output:
                    out[batch_idx] = new_out[local_idx]
                else:
                    out[batch_idx] = torch.cat([out[batch_idx], new_out[local_idx]], dim=0)
                pos[batch_idx] += step

            self._scatter_batch_state(state, batch_state, active)

        return out

    def forward_batch_same_length(self, tokens, state, full_output=False):
        assert isinstance(tokens, list)
        assert len(set(len(item) for item in tokens)) == 1, "all sequences must have the same length"
        return self.forward_seq_batch(tokens, state, full_output)

    def _forward_hidden(self, x, state, mode, full_output=False):
        with torch.no_grad():
            if mode == "one":
                tmix_fn = RWKV_x070_TMix_one
                cmix_fn = RWKV_x070_CMix_one
                increment = 1
            elif mode == "seq":
                tmix_fn = RWKV_x070_TMix_seq
                cmix_fn = RWKV_x070_CMix_seq
                increment = x.shape[0]
            elif mode == "seq_batch":
                tmix_fn = RWKV_x070_TMix_seq_batch
                cmix_fn = RWKV_x070_CMix_seq_batch
                increment = x.shape[1]
            else:
                raise ValueError(f"unsupported forward mode: {mode}")

            z = self.z
            v_first = None

            for stage_idx, (start_layer, end_layer) in enumerate(self.layer_ranges):
                stage_device = self.stage_devices[stage_idx]
                with torch.cuda.device(stage_device):
                    x = self._move_to_stage(x, stage_idx)
                    if v_first is None:
                        v_first = torch.empty_like(x)
                    else:
                        v_first = self._move_to_stage(v_first, stage_idx)

                    elapsed_t = state[2][stage_idx]
                    for local_layer_idx, layer_id in enumerate(range(start_layer, end_layer)):
                        bbb = f"blocks.{layer_id}."
                        att = f"blocks.{layer_id}.att."
                        ffn = f"blocks.{layer_id}.ffn."

                        xx = F.layer_norm(
                            x,
                            (self.n_embd,),
                            weight=z[bbb + "ln1.weight"],
                            bias=z[bbb + "ln1.bias"],
                        )
                        xx, v_first = tmix_fn(
                            layer_id,
                            self.n_head,
                            self.head_size,
                            xx,
                            state[0][stage_idx][local_layer_idx],
                            v_first,
                            state[1][stage_idx][local_layer_idx],
                            z[att + "x_r"],
                            z[att + "x_w"],
                            z[att + "x_k"],
                            z[att + "x_v"],
                            z[att + "x_a"],
                            z[att + "x_g"],
                            z[att + "w0"],
                            z[att + "w1"],
                            z[att + "w2"],
                            z[att + "a0"],
                            z[att + "a1"],
                            z[att + "a2"],
                            z[att + "v0"],
                            z[att + "v1"],
                            z[att + "v2"],
                            z[att + "g1"],
                            z[att + "g2"],
                            z[att + "k_k"],
                            z[att + "k_a"],
                            z[att + "r_k"],
                            z[att + "receptance.weight"],
                            z[att + "key.weight"],
                            z[att + "value.weight"],
                            z[att + "output.weight"],
                            z[att + "ln_x.weight"],
                            z[att + "ln_x.bias"],
                            elapsed_t,
                        )
                        x = x + xx

                        xx = F.layer_norm(
                            x,
                            (self.n_embd,),
                            weight=z[bbb + "ln2.weight"],
                            bias=z[bbb + "ln2.bias"],
                        )
                        xx = cmix_fn(
                            xx,
                            state[0][stage_idx][local_layer_idx],
                            z[ffn + "x_k"],
                            z[ffn + "key.weight"],
                            z[ffn + "value.weight"],
                        )
                        x = x + xx

            x = x.to(self.out_device, non_blocking=True)
            if mode == "seq" and not full_output:
                x = x[-1, :]
            elif mode == "seq_batch" and not full_output:
                x = x[:, -1, :]

            x = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z["ln_out.weight"],
                bias=z["ln_out.bias"],
            )
            x = F.linear(x, z["head.weight"])
            self._increment_elapsed(state, increment)
            return x

    @MyFunction
    def forward_one(self, x: torch.Tensor, state):
        return self._forward_hidden(x, state, "one")

    @MyFunction
    def forward_seq(self, idx: List[int], state, full_output: bool = False):
        x = self.z["emb.weight"][torch.tensor(idx, device=self.emb_device)]
        return self._forward_hidden(x, state, "seq", full_output)

    @MyFunction
    def forward_seq_batch(self, idxs: List[List[int]], state, full_output: bool = False):
        token_tensor = torch.tensor(idxs, device=self.emb_device)
        x = self.z["emb.weight"][token_tensor]
        return self._forward_hidden(x, state, "seq_batch", full_output)

    @MyFunction
    def forward_seq_batch_chunk(
        self,
        idxs: List[List[int]],
        state,
        chunk_len: int = 64,
        full_output: bool = False,
    ):
        outputs = []
        last_output = None
        total_len = len(idxs[0])

        for start in range(0, total_len, chunk_len):
            end = min(start + chunk_len, total_len)
            chunk = [tokens[start:end] for tokens in idxs]
            chunk_output = self.forward_seq_batch(chunk, state, full_output=full_output)
            if full_output:
                outputs.append(chunk_output)
            else:
                last_output = chunk_output

        if full_output:
            return torch.cat(outputs, dim=1)
        return last_output

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
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
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
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
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
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
    return F.linear((xx * g), O_), v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    kv = rwkv_mm_sparsity(k, V_)
    # kv = k @ V_
    # kv = SPMV_OP(k, V_)
    return kv

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_ # F.linear(k, V_)

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    return k @ V_ # F.linear(k, V_)

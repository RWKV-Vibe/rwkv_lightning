import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch._C._jit_set_autocast_mode(False)

HEAD_SIZE = 64
DTYPE = torch.float16
THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"
HIP_DIR = THIS_DIR / "hip"

CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"

_EXTENSIONS_LOADED = False
ROCM_FLAG = torch.version.hip is not None

@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def _resolve_model_path(model_name: str) -> str:
    return model_name if model_name.endswith(".pth") else f"{model_name}.pth"


def load_extensions() -> None:
    global _EXTENSIONS_LOADED
    if _EXTENSIONS_LOADED:
        return
    
    fast_cuda_flags = [
        "-O3",
        "--use_fast_math",
        "--extra-device-vectorization",
    ] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    state_cuda_flags = [
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        f"-D_N_={HEAD_SIZE}",
    ] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])

    load(
        name="rwkv7_fast_ops_fp16",
        sources=[
            str(CUDA_DIR / "rwkv7_fast_ops_fp16.cpp"),
            str(CUDA_DIR / "rwkv7_fast_ops_fp16.cu"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=fast_cuda_flags,
    )
    load(
        name="rwkv7_state_fwd_fp16",
        sources=[
            str(CUDA_DIR / "rwkv7_state_fwd_fp16.cpp"),
            str(CUDA_DIR / "rwkv7_state_fwd_fp16.cu"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=state_cuda_flags,
    )
    _EXTENSIONS_LOADED = True

def load_extensions_rocm() -> None:
    global _EXTENSIONS_LOADED
    if _EXTENSIONS_LOADED:
        return

    fast_cuda_flags = [
        '-fopenmp', 
        '-ffast-math', 
        '-O3', 
        '-munsafe-fp-atomics',
    ]
    state_cuda_flags = [
        '-fopenmp', 
        '-ffast-math', 
        '-O3', 
        '-munsafe-fp-atomics',
        f"-D_N_={HEAD_SIZE}",
    ]

    load(
        name="rwkv7_fast_ops_fp16",
        sources=[
            str(HIP_DIR / "rwkv7_fast_ops_fp16_op.hip"),
            str(HIP_DIR / "rwkv7_fast_ops_fp16.hip"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=fast_cuda_flags,
    )
    load(
        name="rwkv7_state_fwd_fp16",
        sources=[
            str(HIP_DIR / "rwkv7_state_fwd_fp16_op.hip"),
            str(HIP_DIR / "rwkv7_state_fwd_fp16.hip"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=state_cuda_flags,
    )
    _EXTENSIONS_LOADED = True


class RWKV_x070(nn.Module):
    def __init__(self, args):
        super().__init__()
        if ROCM_FLAG:
            load_extensions_rocm()
        else:
            load_extensions()

        self.args = args
        self.args.head_size = HEAD_SIZE
        self.emb_device = os.environ.get("EMB_DEVICE", _get_arg(args, "emb_device", "gpu"))
        self.rkv_mode = _get_arg(args, "batched_rkv", "off")
        self.cmix_sparse = _get_arg(args, "cmix_sparse", "no-fc")
        self.eval()

        model_path = _resolve_model_path(args.MODEL_NAME)
        z = torch.load(model_path, map_location="cpu", mmap=True)
        self.z = z

        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
        self.args.n_embd = self.n_head * self.head_size
        self.args.vocab_size = z["emb.weight"].shape[0]

        assert self.head_size == HEAD_SIZE
        assert self.args.head_size == HEAD_SIZE
        print(args, "emb_device: ", self.emb_device)
        emb_cpu = z["emb.weight"].squeeze() if self.emb_device == "cpu" else None
        max_layer = -1
        transpose_keys = (
            "att.receptance.weight",
            "att.key.weight",
            "att.value.weight",
            "att.output.weight",
            "ffn.key.weight",
            "ffn.value.weight",
            "head.weight",
        )

        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue

            value = z[key].squeeze()
            if key.endswith("ffn.key.weight") and self.cmix_sparse not in ("off", "no-fc"):
                z[f"{key}.fc"] = value.to(device="cuda", dtype=DTYPE).contiguous()
            if any(name in key for name in transpose_keys):
                value = value.t()
            value = value.to(device="cuda", dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            z[key] = value

            parts = key.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))

        self.args.n_layer = max_layer + 1
        self.n_layer = self.args.n_layer
        self.n_embd = self.args.n_embd

        if emb_cpu is None:
            z["emb.weight"] = F.layer_norm(
                z["emb.weight"],
                (self.n_embd,),
                weight=z["blocks.0.ln0.weight"],
                bias=z["blocks.0.ln0.bias"],
            ).contiguous()
        else:
            emb = torch.empty(
                (self.args.vocab_size, self.n_embd),
                dtype=DTYPE,
                pin_memory=True,
            )
            for start in range(0, self.args.vocab_size, 4096):
                end = min(start + 4096, self.args.vocab_size)
                chunk = emb_cpu[start:end].to(device="cuda", dtype=DTYPE)
                chunk = F.layer_norm(
                    chunk,
                    (self.n_embd,),
                    weight=z["blocks.0.ln0.weight"],
                    bias=z["blocks.0.ln0.bias"],
                )
                emb[start:end].copy_(chunk)
            z["emb.weight"] = emb

        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

        if self.rkv_mode != "off":
            for layer in range(self.n_layer):
                prefix = f"blocks.{layer}.att."
                z[f"{prefix}rkv.weight"] = torch.stack(
                    (
                        z[f"{prefix}receptance.weight"],
                        z[f"{prefix}key.weight"],
                        z[f"{prefix}value.weight"],
                    )
                ).contiguous()

        self.emb_cpu = emb_cpu is not None
        self.emb_cache = {}
        self.decode_step_cache = {}

    def generate_zero_state(self, bsz):
        if bsz >= 1:
            return [
                torch.zeros((self.n_layer, 2, bsz, self.n_embd), dtype=DTYPE, device="cuda",),
                torch.zeros((self.n_layer, bsz, self.n_head, HEAD_SIZE, HEAD_SIZE), dtype=DTYPE, device="cuda",),
                torch.zeros((bsz,), dtype=torch.int32, device="cuda"),
            ]
        return [
            torch.zeros((self.n_layer, 2, self.n_embd),dtype=DTYPE,device="cuda",),
            torch.zeros((self.n_layer, self.n_head, HEAD_SIZE, HEAD_SIZE), dtype=DTYPE, device="cuda",),
            torch.zeros((), dtype=torch.int32, device="cuda"), 
        ]

    def zero_state(self, bsz: int):
        return self.generate_zero_state(bsz)

    def forward(self, idx, state, full_output: bool = False):
        if isinstance(idx, list):
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            return self._forward_single_token_id(int(idx[0]), state)

        if torch.is_tensor(idx):
            if idx.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                if idx.dim() == 0:
                    return self._forward_single_token_id(int(idx.item()), state)
                if idx.dim() == 1:
                    return self.forward_seq(idx.detach().cpu().tolist(), state, full_output)
                if idx.dim() == 2:
                    return self.forward_seq_batch(idx.detach().cpu().tolist(), state, full_output)
            return self.forward_one(idx, state)

        return self._forward_single_token_id(int(idx), state)

    def forward_batch(self, tokens, state, full_output: bool = False):
        assert isinstance(tokens, list)
        if tokens and all(isinstance(x, list) and len(x) == 1 for x in tokens) and not full_output:
            return self._forward_batch_one_step([int(x[0]) for x in tokens], state)

        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and not full_output:
            return self.forward_batch_same_length(tokens, state, full_output)

        bsz = len(tokens)
        pos = [0] * bsz
        if not full_output:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, device="cuda",)
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, device="cuda")for _ in range(bsz)]

        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break

            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i] : pos[i] + step] for i in active]
            batch_state = [
                state[0][:, :, active],
                state[1][:, active],
                state[2][active],
            ]
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)

            for k, i in enumerate(active):
                if not full_output:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                state[0][:, :, i] = batch_state[0][:, :, k]
                state[1][:, i] = batch_state[1][:, k]
                state[2][i] = batch_state[2][k]
                pos[i] += step
        return out

    def forward_batch_same_length(self, tokens, state, full_output: bool = False):
        assert isinstance(tokens, list)
        assert len(set(len(x) for x in tokens)) == 1
        return self.forward_seq_batch(tokens, state, full_output)

    def forward_one(self, x: torch.Tensor, state: List[torch.Tensor]):
        with torch.no_grad():
            x = x.to(device="cuda", dtype=DTYPE).view(1, 1, self.n_embd).contiguous()
            return self._forward_x(x, state, False).squeeze(0)

    def forward_seq(self, idx: List[int], state: List[torch.Tensor], full_output: bool = False):
        with torch.no_grad():
            tokens = torch.tensor([idx], dtype=torch.long, device=self._token_device())
            out = self._forward_tokens(tokens, state, full_output)
            return out.squeeze(0)

    def forward_seq_batch(self, idxs: List[List[int]], state: List[torch.Tensor], full_output: bool = False):
        with torch.no_grad():
            tokens = torch.tensor(idxs, dtype=torch.long, device=self._token_device())
            return self._forward_tokens(tokens, state, full_output)

    def forward_seq_batch_chunk(
        self,
        idxs: List[List[int]],
        state: List[torch.Tensor],
        chunk_len: int = 64,
        full_output: bool = False,
    ):
        with torch.no_grad():
            tokens = torch.tensor(idxs, dtype=torch.long, device=self._token_device())
            total_len = tokens.size(1)
            outputs = []
            last_out = None

            for start in range(0, total_len, chunk_len):
                end = min(start + chunk_len, total_len)
                chunk = tokens[:, start:end]
                chunk_out = self._forward_tokens(chunk, state, full_output)
                if full_output:
                    outputs.append(chunk_out)
                else:
                    last_out = chunk_out

            return torch.cat(outputs, dim=1) if full_output else last_out

    def _token_device(self) -> str:
        return "cpu" if self.emb_cpu else "cuda"

    def _embed_single_token_id(self, token_id: int) -> torch.Tensor:
        x = self.z["emb.weight"][token_id]
        if x.device.type != "cuda":
            x = x.to(device="cuda", non_blocking=True)
        return x.contiguous()

    def _forward_single_token_id(self, token_id: int, state):
        x = self._embed_single_token_id(token_id)
        return self.forward_one(x, state)

    def _forward_batch_one_step(self, token_ids: List[int], state):
        bsz = len(token_ids)
        if self.emb_cpu:
            tokens = torch.tensor(token_ids, dtype=torch.long, device=self._token_device())
            x = self._embed_tokens(tokens.unsqueeze(1))
        else:
            cache = self.decode_step_cache.get(bsz)
            if cache is None:
                cache = {
                    "tokens": torch.empty((bsz,), dtype=torch.long, device="cuda"),
                    "x": torch.empty((bsz, self.n_embd), dtype=DTYPE, device="cuda"),
                }
                self.decode_step_cache[bsz] = cache
            cache["tokens"].copy_(torch.tensor(token_ids, dtype=torch.long))
            torch.index_select(self.z["emb.weight"], 0, cache["tokens"], out=cache["x"])
            x = cache["x"].view(bsz, 1, self.n_embd)
        return self._forward_x(x, state, False)

    def _select_path(self, bsz: int, seq_len: int) -> PathConfig:
        rows = bsz * seq_len
        if self.cmix_sparse == "off":
            cmix_mode = CMIX_DENSE
        elif self.cmix_sparse == "no-fc":
            cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if rows == 2 else CMIX_DENSE)
        elif rows == 1:
            cmix_mode = CMIX_B1T1_SPARSE
        elif rows == 2:
            cmix_mode = CMIX_ROWS2_SPARSE
        else:
            cmix_mode = CMIX_DENSE

        if self.rkv_mode == "auto":
            use_batched_rkv = 4 <= rows <= 64
        elif self.rkv_mode == "on":
            use_batched_rkv = True
        else:
            use_batched_rkv = False
        return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        if not self.emb_cpu:
            return self.z["emb.weight"][tokens.to(device="cuda")]

        bsz, seq_len = tokens.shape
        host, dev = self.emb_cache.get((bsz, seq_len), (None, None))
        if host is None:
            host = torch.empty((bsz * seq_len, self.n_embd), dtype=DTYPE, pin_memory=True)
            dev = torch.empty((bsz, seq_len, self.n_embd), dtype=DTYPE, device="cuda")
            self.emb_cache[(bsz, seq_len)] = (host, dev)

        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(bsz, seq_len, self.n_embd), non_blocking=True)
        return dev

    def _forward_tokens(self, tokens: torch.Tensor, state, full_output: bool):
        x = self._embed_tokens(tokens)
        return self._forward_x(x, state, full_output)

    def _forward_x(self, x: torch.Tensor, state, full_output: bool):
        z = self.z
        bsz, seq_len, _ = x.shape
        path = self._select_path(bsz, seq_len)
        single_state = state[0].dim() == 3
        if single_state:
            assert bsz == 1, "single state only supports one sequence"

        v_first = torch.empty_like(x)
        for layer in range(self.n_layer):
            block = f"blocks.{layer}."
            att = f"{block}att."
            ffn = f"{block}ffn."

            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[f"{block}ln1.weight"],
                bias=z[f"{block}ln1.bias"],
            )
            shift_state = state[0][layer, 0].unsqueeze(0) if single_state else state[0][layer, 0]
            wkv_state = state[1][layer]
            elapsed_t = state[2].view(1) if single_state else state[2]
            xx, v_first = self._tmix(layer, xx, shift_state, wkv_state, elapsed_t, v_first, att, path)
            x = x + xx

            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[f"{block}ln2.weight"],
                bias=z[f"{block}ln2.bias"],
            )
            cmix_state = state[0][layer, 1].unsqueeze(0) if single_state else state[0][layer, 1]
            x = x + self._cmix(xx, cmix_state, ffn, path)

        if full_output:
            x = F.layer_norm(x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
            out = x @ z["head.weight"]
        else:
            x = x[:, -1, :]
            x = F.layer_norm(x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
            out = x @ z["head.weight"]

        if single_state:
            state[2] += seq_len
        else:
            state[2].add_(seq_len)
        return out

    def _tmix(self, layer: int, x, shift_state, wkv_state, elapsed_t, v_first, prefix: str, path: PathConfig):
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        bsz, seq_len, _ = x.shape

        xr, xw, xk, xv, xa, xg = ops.tmix_mix6(
            bsz,
            seq_len,
            self.n_embd,
            x.contiguous(),
            shift_state.contiguous(),
            z[f"{prefix}x_r"],
            z[f"{prefix}x_w"],
            z[f"{prefix}x_k"],
            z[f"{prefix}x_v"],
            z[f"{prefix}x_a"],
            z[f"{prefix}x_g"],
        )

        if path.use_batched_rkv:
            flat = torch.stack((xr.reshape(-1, self.n_embd), xk.reshape(-1, self.n_embd), xv.reshape(-1, self.n_embd)))
            rkv = torch.bmm(flat, z[f"{prefix}rkv.weight"])
            r, k, v = (item.view(bsz, seq_len, self.n_embd) for item in rkv.unbind(0))
        else:
            r = xr @ z[f"{prefix}receptance.weight"]
            k = xk @ z[f"{prefix}key.weight"]
            v = xv @ z[f"{prefix}value.weight"]

        w = ops.act_tanh((xw @ z[f"{prefix}w1"]).contiguous()) @ z[f"{prefix}w2"]
        a12 = (xa @ z[f"{prefix}a1"]) @ z[f"{prefix}a2"]
        k, neg_kk, kka = ops.tmix_kk_a_gate(
            bsz,
            seq_len,
            self.n_embd,
            self.n_head,
            k.contiguous(),
            z[f"{prefix}k_k"],
            z[f"{prefix}a0"],
            a12.contiguous(),
            z[f"{prefix}k_a"],
        )
        g = ops.act_sigmoid((xg @ z[f"{prefix}g1"]).contiguous()) @ z[f"{prefix}g2"]

        if layer == 0:
            v_first = v
        else:
            v12 = (xv @ z[f"{prefix}v1"]) @ z[f"{prefix}v2"]
            v = ops.tmix_vres_gate(
                bsz,
                seq_len,
                self.n_embd,
                v.contiguous(),
                v_first.contiguous(),
                z[f"{prefix}v0"],
                v12.contiguous(),
            )

        w = ops.add_vec(self.n_embd, w.contiguous(), z[f"{prefix}w0"])
        y = self._run_wkv(bsz, seq_len, wkv_state, r, w, k, v, neg_kk, kka, elapsed_t)
        y = ops.tmix_lnx_rkvres_xg(
            bsz,
            seq_len,
            self.n_embd,
            self.n_head,
            y.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            z[f"{prefix}r_k"],
            z[f"{prefix}ln_x.weight"],
            z[f"{prefix}ln_x.bias"],
            g.contiguous(),
        )
        return y @ z[f"{prefix}output.weight"], v_first

    def _run_wkv(self, bsz: int, seq_len: int, wkv_state, r, w, k, v, a, b, elapsed_t):
        if seq_len == 1:
            out = torch.empty((bsz, self.n_embd), dtype=DTYPE, device="cuda")
            torch.ops.rwkv7_state_fwd_fp16.forward_one(
                bsz,
                self.n_embd,
                self.n_head,
                wkv_state,
                r[:, 0].contiguous(),
                w[:, 0].contiguous(),
                k[:, 0].contiguous(),
                v[:, 0].contiguous(),
                a[:, 0].contiguous(),
                b[:, 0].contiguous(),
                out,
                elapsed_t,
            )
            return out.view(bsz, 1, self.n_embd)

        out = torch.empty((bsz, seq_len, self.n_embd), dtype=DTYPE, device="cuda")
        torch.ops.rwkv7_state_fwd_fp16.forward_seq(
            bsz,
            seq_len,
            self.n_embd,
            self.n_head,
            wkv_state,
            r.contiguous(),
            w.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            b.contiguous(),
            out,
            elapsed_t,
        )
        return out

    def _cmix(self, x, shift_state, prefix: str, path: PathConfig):
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        bsz, seq_len, _ = x.shape
        key_fc = z.get(f"{prefix}key.weight.fc")

        if path.cmix_mode == CMIX_B1T1_SPARSE and key_fc is not None:
            return ops.cmix_sparse_one(
                self.n_embd,
                key_fc.size(0),
                x.contiguous(),
                shift_state.contiguous(),
                z[f"{prefix}x_k"],
                key_fc,
                z[f"{prefix}value.weight"],
            )
        if path.cmix_mode == CMIX_ROWS2_SPARSE and key_fc is not None:
            return ops.cmix_sparse_rows(
                bsz,
                seq_len,
                self.n_embd,
                key_fc.size(0),
                x.contiguous(),
                shift_state.contiguous(),
                z[f"{prefix}x_k"],
                key_fc,
                z[f"{prefix}value.weight"],
            )

        mixed = ops.cmix_mix(
            bsz,
            seq_len,
            self.n_embd,
            x.contiguous(),
            shift_state.contiguous(),
            z[f"{prefix}x_k"],
        )
        hid = mixed @ z[f"{prefix}key.weight"]
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return ops.cmix_sparse_down_relu_one(
                self.n_embd,
                z[f"{prefix}value.weight"].size(0),
                hid.view(-1).contiguous(),
                z[f"{prefix}value.weight"],
            ).view(1, 1, self.n_embd)
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            return ops.cmix_sparse_down_relu_rows(
                bsz,
                seq_len,
                self.n_embd,
                z[f"{prefix}value.weight"].size(0),
                hid.contiguous(),
                z[f"{prefix}value.weight"],
            )
        return ops.relu_square(hid.contiguous()) @ z[f"{prefix}value.weight"]


RWKV7 = RWKV_x070

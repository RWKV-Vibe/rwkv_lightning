from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List

import torch

from . import rwkv7_fast_v3a as v3a


DEFAULT_RKV_MODE = "off"
DEFAULT_CMIX_SPARSE = "no-fc"
DEFAULT_LOWRANK_WEIGHT = "transpose"
DEFAULT_ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}


def _model_path_from_args(args: Any) -> str:
    model_name = getattr(args, "MODEL_NAME", None) or getattr(args, "MODEL_PATH", None)
    if not model_name:
        raise ValueError("args.MODEL_NAME or args.MODEL_PATH is required")
    return model_name if str(model_name).endswith(".pth") else f"{model_name}.pth"


def _normalize_wkv_mode(value: str | None) -> str:
    if value is None:
        return "fp16"
    value = value.lower()
    if value in {"fp16", "float16", "half"}:
        return "fp16"
    if value in {"fp32", "fp32io16", "float32"}:
        return "fp32io16"
    raise ValueError(f"unknown wkv mode: {value}")


def _normalize_emb_device(value: str | None) -> str:
    if value is None:
        return "cpu"
    value = value.lower()
    if value in {"cpu", "gpu"}:
        return value
    if value in {"cuda", "cuda:0"}:
        return "gpu"
    raise ValueError(f"unknown embedding device: {value}")


def _normalize_pp_devices(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return v3a.parse_pp_devices(value)
    return [int(x) for x in value]


def _to_token_tensor(tokens: Any, device: str) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        if tokens.dtype.is_floating_point:
            raise TypeError("floating tensors are embeddings, not token ids")
        out = tokens.long()
        if out.dim() == 0:
            out = out.view(1, 1)
        elif out.dim() == 1:
            out = out.unsqueeze(0)
        if out.device.type != device:
            out = out.to(device, non_blocking=True)
        return out

    if isinstance(tokens, int):
        return torch.tensor([[tokens]], dtype=torch.long, device=device)

    return torch.tensor([list(tokens)], dtype=torch.long, device=device)


def _state_batch_size(state: List[torch.Tensor]) -> int:
    if isinstance(state[2], list):
        return int(state[2][0].numel())
    return int(state[2].numel())


def _copy_state(dst, src) -> None:
    if isinstance(dst[0], list):
        for d, s in zip(dst[0], src[0]):
            d.copy_(s, non_blocking=True)
        for d, s in zip(dst[1], src[1]):
            d.copy_(s, non_blocking=True)
        for d, s in zip(dst[2], src[2]):
            d.copy_(s, non_blocking=True)
        return
    dst[0].copy_(src[0], non_blocking=True)
    dst[1].copy_(src[1], non_blocking=True)
    dst[2].copy_(src[2], non_blocking=True)


def _slice_state_rows(state, rows: list[int]):
    if isinstance(state[0], list):
        return [
            [x[:, rows, :].contiguous() for x in state[0]],
            [x[rows].contiguous() for x in state[1]],
            [x[rows].contiguous() for x in state[2]],
        ]
    return [
        state[0][:, :, rows].contiguous(),
        state[1][:, rows].contiguous(),
        state[2][rows].contiguous(),
    ]


def _copy_state_rows_back(dst, rows: list[int], src) -> None:
    if isinstance(dst[0], list):
        for layer, tensor in enumerate(src[0]):
            for row, original in enumerate(rows):
                dst[0][layer][:, original].copy_(tensor[:, row])
        for layer, tensor in enumerate(src[1]):
            for row, original in enumerate(rows):
                dst[1][layer][original].copy_(tensor[row])
        for dev_idx, tensor in enumerate(src[2]):
            for row, original in enumerate(rows):
                dst[2][dev_idx][original].copy_(tensor[row])
        return
    for row, original in enumerate(rows):
        dst[0][:, :, original].copy_(src[0][:, :, row])
        dst[1][:, original].copy_(src[1][:, row])
        dst[2][original].copy_(src[2][row])


@dataclass
class _DecodeGraph:
    state: Any
    x: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None


class RWKV_x070:
    """Compatibility wrapper around rwkv7_fast_v3a.RWKV7.

    The old inference stack expects RWKV_x070(args), generate_zero_state(),
    forward(), and forward_batch().  This class keeps that surface while using
    the v3a fused kernels underneath.
    """

    def __init__(
        self,
        args: Any,
        wkv_mode: str | None = None,
        emb_device: str | None = None,
        pp_devices: Any = None,
        use_cuda_graph: bool | None = None,
    ) -> None:
        self.args = args
        model_path = _model_path_from_args(args)
        self.wkv_mode = _normalize_wkv_mode(
            wkv_mode
            or getattr(args, "WKV_MODE", None)
            or getattr(args, "wkv_mode", None)
            or getattr(args, "wkv", None)
        )
        self.emb_device = _normalize_emb_device(
            emb_device
            or getattr(args, "EMB_DEVICE", None)
            or getattr(args, "emb_device", None)
            or getattr(args, "emb", None)
        )
        self.pp_devices = _normalize_pp_devices(
            pp_devices
            if pp_devices is not None
            else (
                getattr(args, "PP_DEVICES", None)
                or getattr(args, "pp_devices", None)
                or getattr(args, "pp", None)
            )
        )
        if use_cuda_graph is None:
            use_cuda_graph = bool(getattr(args, "USE_CUDA_GRAPH", True))
        self.use_cuda_graph = use_cuda_graph
        self.decode_graphs: dict[int, _DecodeGraph] = {}
        if self.use_cuda_graph and len(self.pp_devices) > 1:
            print(
                "[rwkv7_v3a] PP decode runs in eager mode; single-GPU decode graph remains enabled.",
                flush=True,
            )

        v3a.MODEL_PATH = model_path
        v3a.WKV_MODE = self.wkv_mode
        v3a.EMB_DEVICE = self.emb_device
        v3a.RKV_MODE = DEFAULT_RKV_MODE
        v3a.CMIX_SPARSE = DEFAULT_CMIX_SPARSE
        v3a.LOWRANK_WEIGHT = DEFAULT_LOWRANK_WEIGHT
        v3a.ORIG_LINEAR_GROUPS = set(DEFAULT_ORIG_LINEAR_GROUPS)
        v3a.PP_DEVICES = list(self.pp_devices)
        v3a.load_extensions(v3a.WKV_MODE)

        self.inner = v3a.RWKV7()
        self.z = self.inner.z
        self.emb_cpu = self.inner.emb_cpu
        self.n_layer = v3a.L
        self.n_embd = v3a.C
        self.n_head = v3a.H
        self.head_size = v3a.N

        args.MODEL_NAME = os.path.splitext(model_path)[0]
        args.head_size = v3a.N
        args.n_layer = v3a.L
        args.n_embd = v3a.C
        args.vocab_size = v3a.V
        args.PP_DEVICES = list(self.pp_devices)
        args.USE_CUDA_GRAPH = self.use_cuda_graph

    def eval(self) -> "RWKV_x070":
        return self

    def generate_zero_state(self, bsz: int):
        return self.zero_state(max(1, int(bsz)))

    def zero_state(self, bsz: int):
        return self.inner.zero_state(max(1, int(bsz)))

    def prepare_state(self, state):
        if not v3a.pp_enabled():
            return state
        if not isinstance(state[0], list):
            raise ValueError("pipeline-parallel mode requires nested PP state")
        state[0] = [
            tensor.to(v3a.layer_device(layer), non_blocking=True)
            for layer, tensor in enumerate(state[0])
        ]
        state[1] = [
            tensor.to(v3a.layer_device(layer), non_blocking=True)
            for layer, tensor in enumerate(state[1])
        ]
        state[2] = [
            tensor.to(torch.device(f"cuda:{dev_id}"), non_blocking=True)
            for tensor, dev_id in zip(state[2], v3a.PP_DEVICES)
        ]
        return state

    def forward(self, idx, state, full_output: bool = False):
        if isinstance(idx, torch.Tensor) and idx.dtype.is_floating_point:
            return self.forward_from_x(idx, state, full_output=full_output)

        single = isinstance(idx, int) or (
            isinstance(idx, torch.Tensor) and idx.dim() == 0
        )
        if isinstance(idx, list) and idx and isinstance(idx[0], list):
            return self.forward_batch(idx, state, full_output=full_output)

        token_device = "cpu" if self.emb_cpu else "cuda"
        tokens = _to_token_tensor(idx if not isinstance(idx, tuple) else list(idx), token_device)
        if self._can_use_decode_graph(tokens, state, full_output):
            out = self._forward_decode_graph(tokens, state)
        elif full_output:
            out = self.inner.forward_all_logits(tokens, state)
        else:
            out = self.inner.forward(tokens, state)
        return out[0] if single or _state_batch_size(state) == 1 else out

    def forward_from_x(self, x: torch.Tensor, state, full_output: bool = False):
        if x.dim() == 1:
            x = x.view(1, 1, -1)
            squeeze = True
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = _state_batch_size(state) == 1
        else:
            squeeze = False
        path = v3a.select_path(x.size(0), x.size(1))
        if self._can_use_x_decode_graph(x, state, full_output):
            out = self._forward_x_decode_graph(x, state)
        else:
            out = self.inner.forward_from_x(x, state, path, all_logits=full_output)
        return out[0] if squeeze and not full_output else out

    def forward_batch(self, tokens, state, full_output: bool = False):
        assert isinstance(tokens, list)
        if not tokens:
            raise ValueError("tokens must not be empty")
        lengths = [len(x) if isinstance(x, Iterable) else 1 for x in tokens]
        if len(set(lengths)) == 1:
            return self.forward_batch_same_length(tokens, state, full_output)
        return self._forward_batch_variable_length(tokens, state, full_output)

    def forward_batch_same_length(self, tokens, state, full_output: bool = False):
        assert isinstance(tokens, list)
        token_device = "cpu" if self.emb_cpu else "cuda"
        token_rows = [[int(x)] if isinstance(x, int) else list(x) for x in tokens]
        idx = torch.tensor(token_rows, dtype=torch.long, device=token_device)
        if self._can_use_decode_graph(idx, state, full_output):
            return self._forward_decode_graph(idx, state)
        if full_output:
            return self.inner.forward_all_logits(idx, state)
        return self.inner.forward(idx, state)

    def forward_seq_batch(self, idxs, state, full_output: bool = False):
        return self.forward_batch_same_length(idxs, state, full_output)

    def forward_seq_batch_chunk(
        self,
        idxs,
        state,
        chunk_len: int = 128,
        full_output: bool = False,
    ):
        out = None
        all_out = []
        total_len = len(idxs[0]) if idxs else 0
        for start in range(0, total_len, chunk_len):
            chunk = [row[start : start + chunk_len] for row in idxs]
            out = self.forward_batch_same_length(chunk, state, full_output)
            if full_output:
                all_out.append(out)
        if full_output:
            return torch.cat(all_out, dim=1)
        return out

    def forward_all_logits(self, tokens: torch.Tensor, state):
        return self.inner.forward_all_logits(tokens, state)

    def forward_last_at(
        self,
        tokens: torch.Tensor,
        state,
        last_indices: torch.Tensor,
    ):
        return self.inner.forward_last_at(tokens, state, last_indices)

    def _forward_batch_variable_length(self, tokens, state, full_output: bool):
        bsz = len(tokens)
        token_rows = [[int(x)] if isinstance(x, int) else list(x) for x in tokens]
        lengths = [len(x) for x in token_rows]
        pos = [0] * bsz

        if full_output:
            out = [
                torch.empty((0, self.args.vocab_size), dtype=v3a.DTYPE, device="cuda")
                for _ in range(bsz)
            ]
        else:
            out = torch.empty((bsz, self.args.vocab_size), dtype=v3a.DTYPE, device="cuda")

        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [token_rows[i][pos[i] : pos[i] + step] for i in active]
            batch_state = _slice_state_rows(state, active)
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)
            for row, original in enumerate(active):
                if full_output:
                    out[original] = torch.cat([out[original], new_out[row]], dim=0)
                else:
                    out[original] = new_out[row]
                pos[original] += step
            _copy_state_rows_back(state, active, batch_state)
        return out

    def _can_use_decode_graph(self, tokens, state, full_output: bool) -> bool:
        return (
            self.use_cuda_graph
            and not v3a.pp_enabled()
            and not full_output
            and torch.cuda.is_available()
            and isinstance(tokens, torch.Tensor)
            and tokens.dim() == 2
            and tokens.size(1) == 1
            and _state_batch_size(state) == tokens.size(0)
        )

    def _can_use_x_decode_graph(self, x, state, full_output: bool) -> bool:
        return (
            self.use_cuda_graph
            and not v3a.pp_enabled()
            and not full_output
            and torch.cuda.is_available()
            and isinstance(x, torch.Tensor)
            and x.dim() == 3
            and x.size(1) == 1
            and _state_batch_size(state) == x.size(0)
        )

    def _forward_decode_graph(self, tokens: torch.Tensor, state):
        x = self.inner.embed(tokens)
        return self._forward_x_decode_graph(x, state)

    def _forward_x_decode_graph(self, x: torch.Tensor, state):
        bsz = x.size(0)
        graph = self._get_decode_graph(bsz)
        _copy_state(graph.state, state)
        graph.x.copy_(x.to(graph.x.device), non_blocking=True)
        graph.graph.replay()
        _copy_state(state, graph.state)
        return graph.output

    def _get_decode_graph(self, bsz: int) -> _DecodeGraph:
        cached = self.decode_graphs.get(bsz)
        if cached is not None:
            return cached
        cached = self._build_single_gpu_decode_graph(bsz)
        self.decode_graphs[bsz] = cached
        return cached

    def _build_single_gpu_decode_graph(self, bsz: int) -> _DecodeGraph:
        state = self.inner.zero_state(bsz)
        x = torch.empty((bsz, 1, v3a.C), dtype=v3a.DTYPE, device=v3a.first_device())
        path = v3a.select_path(bsz, 1)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            self.inner.forward_from_x(x, state, path)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.inner.forward_from_x(x, state, path)
        v3a.sync_all()
        return _DecodeGraph(state=state, x=x, output=output, graph=graph)


RWKV7 = RWKV_x070

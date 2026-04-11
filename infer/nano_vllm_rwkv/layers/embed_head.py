import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from infer.nano_vllm_rwkv.utils.context import get_context
from infer.nano_vllm_rwkv.layers.linear import _int8_per_channel_cublas


@torch.jit.ignore
def _all_reduce_(x: torch.Tensor):
    dist.all_reduce(x)


@torch.jit.ignore
def _gather_logits(logits: torch.Tensor, tp_size: int, tp_rank: int):
    all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if tp_rank == 0 else None
    dist.gather(logits, all_logits, 0)
    return torch.cat(all_logits, -1) if tp_rank == 0 else None


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size == 1:
            return F.embedding(x, self.weight)
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        y = mask.unsqueeze(1) * y
        _all_reduce_(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(embedding_dim, self.num_embeddings_per_partition))
        self.weight.weight_loader = self.weight_loader
        self.register_buffer("qweight", None)
        self.register_buffer("scales", None)
        self.register_buffer("scales_fp16", None)
        self.group_size = 128
        self.use_int8 = False

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size).t().contiguous()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size == 1 and x.dim() == 2:
            if self.use_int8:
                return _int8_per_channel_cublas(x, self.qweight, self.scales, self.scales_fp16, None)
            return F.linear(x, self.weight.t())
        context = get_context()
        if context.is_prefill:
            if x.dim() == 3:
                x = x[:, -1, :].contiguous()
            else:
                last_indices = context.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        if self.use_int8:
            logits = _int8_per_channel_cublas(x, self.qweight, self.scales, self.scales_fp16, None)
        else:
            logits = F.linear(x, self.weight.t())
        if self.tp_size > 1:
            logits = _gather_logits(logits, self.tp_size, self.tp_rank)
        return logits

    @torch.no_grad()
    def quantize_weight_int8(self, eps: float = 1e-8):
        weight = self.weight.data
        # Store lm_head weights in row-major [N, K] for torch._int_mm.
        weight_row = weight.t().contiguous()
        max_abs = weight_row.abs().amax(dim=1)
        scales = torch.clamp(max_abs / 127.0, min=eps)
        qweight = torch.round(weight_row / scales[:, None]).clamp(-127, 127).to(torch.int8)
        self.qweight = qweight
        self.scales = scales.to(weight.dtype)
        self.scales_fp16 = scales.to(torch.float16)
        self.use_int8 = True
        if "weight" in self._parameters:
            del self._parameters["weight"]
        self.register_parameter("weight", None)

from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    force_regular_decode: bool = False
    force_contiguous_decode: bool = False
    contiguous_decode_slot_in_start: int = -1
    contiguous_decode_slot_out_start: int = -1
    contiguous_decode_slot_count: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    slot_mapping_in: torch.Tensor | None = None
    slot_mapping_out: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    force_regular_decode=False,
    force_contiguous_decode=False,
    contiguous_decode_slot_in_start=-1,
    contiguous_decode_slot_out_start=-1,
    contiguous_decode_slot_count=0,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    slot_mapping_in=None,
    slot_mapping_out=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        force_regular_decode,
        force_contiguous_decode,
        contiguous_decode_slot_in_start,
        contiguous_decode_slot_out_start,
        contiguous_decode_slot_count,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        slot_mapping_in,
        slot_mapping_out,
        context_lens,
        block_tables,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

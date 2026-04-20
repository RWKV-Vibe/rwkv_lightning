#!/usr/bin/env python3

import sys
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

sampler_stub = ModuleType("infer.rwkv_batch.sampler")
sampler_stub.sample = SimpleNamespace(
    setup_rand=lambda seed, batch_size: None,
    batch_sampling_repetition_temperature_topk_topp=lambda *args, **kwargs: None,
)
sys.modules.setdefault("infer.rwkv_batch.sampler", sampler_stub)

from infer.inference import InferenceEngine


class DummyTokenizer:
    def encode(self, text: str):
        return [1]

    def decode(self, tokens, utf8_errors="ignore"):
        mapping = {5: "X"}
        return "".join(mapping.get(token, "") for token in tokens)


class DummyPagedModel:
    def __init__(self):
        self.vocab_size = 8
        self.z = {
            "head.weight": torch.zeros(1),
            "emb.weight": torch.zeros(8, 1),
        }
        self.page_tables = []
        self.token_steps = []

    def generate_zero_state(self, bsz):
        return [
            torch.zeros((1, 2, bsz, 1), dtype=torch.float32),
            torch.zeros((1, bsz, 1, 1, 1), dtype=torch.float32),
            torch.zeros((bsz,), dtype=torch.int32),
        ]

    def forward_batch_paged(self, tokens, state, page_table, full_output=False):
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self.page_tables.append(page_table.cpu().tolist())
        self.token_steps.append(int(tokens.shape[1]))
        state[2][page_table.long()] += int(tokens.shape[1])
        return torch.zeros((tokens.shape[0], self.vocab_size), dtype=torch.float32)


def main():
    model = DummyPagedModel()
    engine = InferenceEngine(
        model=model,
        tokenizer=DummyTokenizer(),
        args=SimpleNamespace(vocab_size=model.vocab_size),
        rocm_flag=False,
    )

    samples = iter([
        torch.tensor([0], dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    ])

    engine._sample_top_k_top_p = lambda logits, top_k, top_p, temperature=1.0: next(samples)

    results = engine._continuous_batching_sync(
        inputs=["prompt-a", "prompt-b", "prompt-c"],
        stop_tokens=[0],
        max_generate_tokens=2,
        batch_size=2,
        pad_zero=True,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        alpha_presence=0.0,
        alpha_frequency=0.0,
        alpha_decay=1.0,
    )

    if results != ["", "X", "X"]:
        raise AssertionError(f"Unexpected paged batching results: {results!r}")

    prefill_page_tables = [pages for pages, step in zip(model.page_tables, model.token_steps) if step > 1]
    decode_page_tables = [pages for pages, step in zip(model.page_tables, model.token_steps) if step == 1]

    if prefill_page_tables[0] != [0, 1]:
        raise AssertionError(
            f"Initial prefill should allocate the first free pages [0, 1], got {prefill_page_tables[0]!r}"
        )

    if [1] not in decode_page_tables:
        raise AssertionError(
            f"Expected prompt-b to keep page 1 across prefill and decode, got {decode_page_tables!r}"
        )

    if [2] not in prefill_page_tables[1:] or [2] not in decode_page_tables:
        raise AssertionError(
            f"Expected prompt-c to keep page 2 across prefill and decode, got prefill={prefill_page_tables!r}, decode={decode_page_tables!r}"
        )

    print("paged continuous batching ok")


if __name__ == "__main__":
    main()

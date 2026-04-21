#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sampler_stub = ModuleType("infer.rwkv_batch.sampler")
sampler_stub.sample = SimpleNamespace(
    setup_rand=lambda seed, batch_size: torch.zeros(
        (max(1, batch_size),), dtype=torch.int64
    ),
    batch_sampling_repetition_temperature_topk_topp=lambda *args, **kwargs: torch.tensor(
        [0], dtype=torch.long
    ),
    batch_sampling_temperature_topk_topp=lambda *args, **kwargs: torch.tensor(
        [0], dtype=torch.long
    ),
)
sys.modules.setdefault("infer.rwkv_batch.sampler", sampler_stub)

utils_stub = ModuleType("infer.rwkv_batch.utils")
utils_stub.sampler_gumbel_batch = lambda logits, temp: torch.zeros(
    (logits.shape[0], 1), dtype=torch.long
)
sys.modules.setdefault("infer.rwkv_batch.utils", utils_stub)

import infer.inference as inference_module


class DummyTokenizer:
    def encode(self, text: str):
        return [1, 2, 3]

    def decode(self, tokens, utf8_errors="ignore"):
        mapping = {11: "B"}
        return "".join(mapping.get(int(token), "") for token in tokens)


class DummyBatchModel:
    def __init__(self):
        self.vocab_size = 16
        self.z = {
            "head.weight": torch.zeros(1),
            "emb.weight": torch.zeros((self.vocab_size, 1)),
        }
        self.forward_batch_sizes = []

    def generate_zero_state(self, bsz):
        return [
            torch.zeros((1, 2, bsz, 1), dtype=torch.float32),
            torch.zeros((1, bsz, 1, 1, 1), dtype=torch.float32),
            torch.zeros((bsz,), dtype=torch.int32),
        ]

    def forward_batch(self, tokens, state, full_output=False):
        if torch.is_tensor(tokens):
            batch_size = int(tokens.shape[0]) if tokens.ndim >= 2 else 1
        else:
            batch_size = len(tokens)
        self.forward_batch_sizes.append(batch_size)
        return torch.zeros((batch_size, self.vocab_size), dtype=torch.float32)


def _make_engine():
    return inference_module.InferenceEngine(
        model=DummyBatchModel(),
        tokenizer=DummyTokenizer(),
        args=SimpleNamespace(vocab_size=16),
        rocm_flag=False,
    )


def _assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


async def _collect_stream(iterator) -> list[str]:
    chunks = []
    async for chunk in iterator:
        chunks.append(chunk)
    return chunks


def test_standard_batch_compacts_finished_rows() -> None:
    engine = _make_engine()
    sampled = iter(
        [
            torch.tensor([0, 11], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ]
    )

    engine._sample_with_repetition = lambda *args, **kwargs: next(sampled)

    result = engine.batch_generate(
        prompts=["prompt-a", "prompt-b"],
        max_length=2,
        stop_tokens=[0],
        top_k=1,
        top_p=1.0,
    )

    _assert_equal(result, ["", "B"], "standard generate output")
    _assert_equal(
        engine.model.forward_batch_sizes,
        [2, 2, 1],
        "standard generate should compact decode batch size after one row finishes",
    )
    print("[PASS] test_standard_batch_compacts_finished_rows")


async def test_standard_stream_preserves_original_indices_after_compaction() -> None:
    engine = _make_engine()
    sampled = iter(
        [
            torch.tensor([0, 11], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ]
    )

    engine._sample_with_repetition = lambda *args, **kwargs: next(sampled)

    chunks = await _collect_stream(
        engine.batch_infer_stream(
            prompts=["prompt-a", "prompt-b"],
            max_length=2,
            stop_tokens=[0],
            chunk_size=1,
            top_k=1,
            top_p=1.0,
        )
    )

    combined = "".join(chunks)
    if '"index": 1' not in combined or '"content": "B"' not in combined:
        raise AssertionError(
            "standard stream should emit surviving content under the original prompt index"
        )
    _assert_equal(
        engine.model.forward_batch_sizes,
        [2, 2, 1],
        "standard stream should compact decode batch size after one row finishes",
    )
    print("[PASS] test_standard_stream_preserves_original_indices_after_compaction")


def test_throughput_batch_compacts_finished_rows() -> None:
    engine = _make_engine()
    sampled = iter(
        [
            torch.tensor([[0], [11]], dtype=torch.long),
            torch.tensor([[0]], dtype=torch.long),
        ]
    )

    engine._sample_throughput_tokens = lambda *args, **kwargs: next(sampled)

    result = engine.big_batch_generate(
        prompts=["prompt-a", "prompt-b"],
        max_length=2,
        stop_tokens=[0],
    )

    _assert_equal(result, ["", "B"], "throughput generate output")
    _assert_equal(
        engine.model.forward_batch_sizes,
        [2, 2, 1],
        "throughput generate should compact decode batch size after one row finishes",
    )
    print("[PASS] test_throughput_batch_compacts_finished_rows")


async def test_continuous_stream_avoids_queue_polling_sleep() -> None:
    engine = _make_engine()

    def _fake_stream_sync(
        inputs,
        stop_tokens,
        max_generate_tokens,
        batch_size,
        output_queue,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
        prefix_cache_manager=None,
    ):
        output_queue.put("data: chunk\n\n")
        output_queue.put("EOF")

    original_sync = engine._continuous_batching_stream_sync
    original_sleep = inference_module.asyncio.sleep

    async def _unexpected_sleep(delay, *args, **kwargs):
        raise AssertionError(
            f"continuous stream wrapper should not poll with asyncio.sleep, got delay={delay}"
        )

    try:
        engine._continuous_batching_stream_sync = _fake_stream_sync
        inference_module.asyncio.sleep = _unexpected_sleep
        chunks = await _collect_stream(
            engine.continuous_batching_stream(
                inputs=["prompt"],
                stop_tokens=[0],
                max_generate_tokens=1,
                batch_size=1,
            )
        )
    finally:
        engine._continuous_batching_stream_sync = original_sync
        inference_module.asyncio.sleep = original_sleep

    _assert_equal(
        chunks,
        ["data: chunk\n\n", "data: [DONE]\n\n"],
        "continuous stream wrapper output",
    )
    print("[PASS] test_continuous_stream_avoids_queue_polling_sleep")


async def main() -> None:
    test_standard_batch_compacts_finished_rows()
    await test_standard_stream_preserves_original_indices_after_compaction()
    test_throughput_batch_compacts_finished_rows()
    await test_continuous_stream_avoids_queue_polling_sleep()
    print("scheduler optimizations ok")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Focused tests for InferenceEngine graph generation paths."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

class FakeTensor:
    def __init__(self):
        self.device = SimpleNamespace(type="cuda")

    def dim(self):
        return 1

    def unsqueeze(self, _dim):
        return self

    def float(self):
        return self

    def to(self, _device):
        return self

    def copy_(self, _other):
        return self

    def __getitem__(self, _index):
        return self


class DummyCUDAGraph:
    def replay(self):
        return None


@contextmanager
def dummy_cuda_graph(_graph):
    yield


cuda_stub = ModuleType("torch.cuda")
setattr(cuda_stub, "CUDAGraph", DummyCUDAGraph)
setattr(cuda_stub, "is_available", lambda: True)
setattr(cuda_stub, "graph", dummy_cuda_graph)
setattr(cuda_stub, "empty_cache", lambda: None)

torch_stub = ModuleType("torch")
setattr(torch_stub, "cuda", cuda_stub)
setattr(torch_stub, "float32", "float32")
setattr(torch_stub, "zeros", lambda *args, **kwargs: FakeTensor())
setattr(torch_stub, "empty_like", lambda _tensor, device=None: FakeTensor())

sys.modules.setdefault("torch", torch_stub)



class DummyTokenizer:
    def __init__(self):
        self.decoded = {11: "A", 12: "B", 13: "C"}

    def encode(self, text: str):
        return [1, 2, 3]

    def decode(self, tokens, utf8_errors="ignore"):
        return "".join(self.decoded.get(token, "") for token in tokens)


class DummyModel:
    def __init__(self):
        self.z = {"emb.weight": FakeTensor(), "head.weight": FakeTensor()}

    def generate_zero_state(self, index: int):
        return [FakeTensor(), FakeTensor(), FakeTensor()]

    def forward(self, tokens, state):
        return FakeTensor()


class FakeScalar:
    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


sampler_stub = ModuleType("infer.rwkv_batch.sampler")
setattr(
    sampler_stub,
    "sample",
    SimpleNamespace(
        setup_rand=lambda seed, batch_size: None,
        batch_sampling_repetition_temperature_topk_topp=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault("infer.rwkv_batch.sampler", sampler_stub)

utils_stub = ModuleType("infer.rwkv_batch.utils")
setattr(utils_stub, "sampler_gumbel_batch", lambda logits, temp: FakeScalar(0))
sys.modules.setdefault("infer.rwkv_batch.utils", utils_stub)

import infer.inference as inference_module


class FakeBatchTokens:
    def __init__(self, value: int):
        self.value = value

    def tolist(self):
        return [self.value]


def _make_engine() -> inference_module.InferenceEngine:
    return inference_module.InferenceEngine(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        args=SimpleNamespace(vocab_size=65536),
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


@contextmanager
def _patch_sampler(
    *,
    gumbel_fn=None,
    setup_rand_fn=None,
    batch_sampling_fn=None,
):
    original_sampler = inference_module.sampler_gumbel_batch
    original_setup_rand = getattr(inference_module.sample, "setup_rand")
    original_batch_sampling = getattr(
        inference_module.sample, "batch_sampling_repetition_temperature_topk_topp"
    )
    try:
        if gumbel_fn is not None:
            inference_module.sampler_gumbel_batch = gumbel_fn
        if setup_rand_fn is not None:
            setattr(inference_module.sample, "setup_rand", setup_rand_fn)
        if batch_sampling_fn is not None:
            setattr(
                inference_module.sample,
                "batch_sampling_repetition_temperature_topk_topp",
                batch_sampling_fn,
            )
        yield
    finally:
        inference_module.sampler_gumbel_batch = original_sampler
        setattr(inference_module.sample, "setup_rand", original_setup_rand)
        setattr(
            inference_module.sample,
            "batch_sampling_repetition_temperature_topk_topp",
            original_batch_sampling,
        )


@contextmanager
def _patch_cuda_available(available: bool):
    original_is_available = inference_module.torch.cuda.is_available
    try:
        inference_module.torch.cuda.is_available = lambda: available
        yield
    finally:
        inference_module.torch.cuda.is_available = original_is_available


async def test_graph_generate_respects_max_generate_tokens() -> None:
    engine = _make_engine()
    batch_tokens = iter([12, 13])

    with _patch_sampler(
        setup_rand_fn=lambda seed, batch_size: None,
        batch_sampling_fn=lambda *args, **kwargs: FakeBatchTokens(next(batch_tokens)),
    ):
        result = engine.graph_generate_state(
            prompts=["hello"],
            state=engine.model.generate_zero_state(0),
            stop_tokens=[0],
            max_length=2,
            session_id="graph-test",
        )

    _assert_equal(result, ["BC"], "graph_generate should cap output to max_generate_tokens")
    print("[PASS] test_graph_generate_respects_max_generate_tokens")


async def test_graph_infer_stream_stops_on_initial_stop_token() -> None:
    engine = _make_engine()
    sampler_calls = {"count": 0}

    def _unexpected_batch_sampling(*args, **kwargs):
        sampler_calls["count"] += 1
        if sampler_calls["count"] > 1:
            raise AssertionError("graph_infer_stream_state should not sample again after initial stop token")
        return FakeBatchTokens(0)

    with _patch_sampler(
        batch_sampling_fn=_unexpected_batch_sampling,
    ):
        chunks = await _collect_stream(
            engine.graph_infer_stream_state(
                prompts=["hello"],
                state=engine.model.generate_zero_state(0),
                stop_tokens=[0],
                max_length=3,
                chunk_size=2,
                session_id="graph-stream-test",
            )
        )

    combined = "".join(chunks)
    if '"content"' in combined:
        raise AssertionError("graph_infer_stream_state should not emit content when the first token is a stop token")
    if combined.count("[DONE]") != 1:
        raise AssertionError("graph_infer_stream_state should emit exactly one DONE marker")
    print("[PASS] test_graph_infer_stream_stops_on_initial_stop_token")


async def test_batch_infer_stream_state_respects_chunk_size() -> None:
    engine = _make_engine()
    sampled_tokens = iter([11, 12, 0])

    with _patch_sampler(
        setup_rand_fn=lambda seed, batch_size: None,
        batch_sampling_fn=lambda *args, **kwargs: FakeBatchTokens(next(sampled_tokens)),
    ):
        chunks = await _collect_stream(
            engine.batch_infer_stream_state(
                prompts=["hello"],
                state=engine.model.generate_zero_state(0),
                stop_tokens=[0],
                max_length=3,
                chunk_size=2,
            )
        )

    combined = "".join(chunks)
    if '"content": "A"' in combined:
        raise AssertionError(
            "batch_infer_stream_state should not flush the first token before chunk_size is reached"
        )
    if '"content": "AB"' not in combined:
        raise AssertionError(
            "batch_infer_stream_state should flush buffered tokens once chunk_size is reached"
        )
    if combined.count("[DONE]") != 1:
        raise AssertionError("batch_infer_stream_state should emit exactly one DONE marker")
    print("[PASS] test_batch_infer_stream_state_respects_chunk_size")


async def main() -> None:
    await test_graph_generate_respects_max_generate_tokens()
    await test_graph_infer_stream_stops_on_initial_stop_token()
    await test_batch_infer_stream_state_respects_chunk_size()
    print("All inference graph tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

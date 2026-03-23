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
        args=SimpleNamespace(),
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
        gumbel_fn=lambda logits, temp: FakeScalar(11),
        setup_rand_fn=lambda seed, batch_size: None,
        batch_sampling_fn=lambda *args, **kwargs: FakeBatchTokens(next(batch_tokens)),
    ):
        result = await engine.graph_generate(
            inputs=["hello"],
            stop_tokens=[0],
            max_generate_tokens=2,
        )

    _assert_equal(result, ["AB"], "graph_generate should cap output to max_generate_tokens")
    print("[PASS] test_graph_generate_respects_max_generate_tokens")


async def test_graph_infer_stream_stops_on_initial_stop_token() -> None:
    engine = _make_engine()

    def _unexpected_batch_sampling(*args, **kwargs):
        raise AssertionError("graph_infer_stream should not sample again after initial stop token")

    with _patch_sampler(
        gumbel_fn=lambda logits, temp: FakeScalar(0),
        batch_sampling_fn=_unexpected_batch_sampling,
    ):
        chunks = await _collect_stream(
            engine.graph_infer_stream(
                inputs=["hello"],
                stop_tokens=[0],
                max_generate_tokens=3,
                chunk_size=2,
            )
        )

    combined = "".join(chunks)
    if '"finish_reason": "stop"' not in combined:
        raise AssertionError("graph_infer_stream should report stop when the first token is a stop token")
    if '"content"' in combined:
        raise AssertionError("graph_infer_stream should not emit content when the first token is a stop token")
    if combined.count("[DONE]") != 1:
        raise AssertionError("graph_infer_stream should emit exactly one DONE marker")
    print("[PASS] test_graph_infer_stream_stops_on_initial_stop_token")


async def test_graph_infer_stream_falls_back_without_cuda() -> None:
    dynamic_calls = []

    class FallbackEngine(inference_module.InferenceEngine):
        async def dynamic_batch_infer_stream(
            self,
            prompt,
            max_generate_tokens,
            stop_tokens,
            pad_zero=True,
            temperature=1.0,
            top_k=50,
            top_p=0.3,
            alpha_presence=0.5,
            alpha_frequency=0.5,
            alpha_decay=0.996,
            chunk_size=32,
        ):
            dynamic_calls.append({"prompt": prompt, "max_generate_tokens": max_generate_tokens})
            yield {"type": "delta", "text": "fallback-stream"}
            yield {"type": "done", "finish_reason": "length"}

    engine = FallbackEngine(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        args=SimpleNamespace(),
        rocm_flag=False,
    )

    with _patch_cuda_available(False):
        chunks = await _collect_stream(
            engine.graph_infer_stream(
                inputs=["hello"],
                stop_tokens=[0],
                max_generate_tokens=2,
                chunk_size=1,
            )
        )

    combined = "".join(chunks)
    if "fallback-stream" not in combined:
        raise AssertionError("graph_infer_stream should preserve fallback delta content when CUDA is unavailable")
    if '"finish_reason": "length"' not in combined:
        raise AssertionError("graph_infer_stream should preserve fallback finish_reason when CUDA is unavailable")
    _assert_equal(
        len(dynamic_calls),
        1,
        "graph_infer_stream should delegate to dynamic_batch_infer_stream when CUDA is unavailable",
    )
    print("[PASS] test_graph_infer_stream_falls_back_without_cuda")


async def main() -> None:
    await test_graph_generate_respects_max_generate_tokens()
    await test_graph_infer_stream_stops_on_initial_stop_token()
    await test_graph_infer_stream_falls_back_without_cuda()
    print("All inference graph tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

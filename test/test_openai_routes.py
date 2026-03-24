#!/usr/bin/env python3
"""Focused tests for OpenAI route stateful/stateless branching."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class DummyResponse:
    def __init__(self, status_code: int, description: str, headers: dict | None = None):
        self.status_code = status_code
        self.description = description
        self.headers = headers or {}


class DummyStreamingResponse:
    def __init__(self, iterator, media_type: str):
        self.iterator = iterator
        self.media_type = media_type


robyn_stub = ModuleType("robyn")
setattr(robyn_stub, "Response", DummyResponse)
setattr(robyn_stub, "StreamingResponse", DummyStreamingResponse)
sys.modules.setdefault("robyn", robyn_stub)

state_pool_stub = ModuleType("state_manager.state_pool")
setattr(state_pool_stub, "get_state_manager", lambda: None)
sys.modules.setdefault("state_manager.state_pool", state_pool_stub)

import API_servers.openai_routes as openai_routes


class ChatRequest:
    def __init__(self, **kwargs):
        self.model = "rwkv7"
        self.contents = []
        self.messages = []
        self.system = None
        self.max_tokens = 128
        self.stop_tokens = [0, 261, 24281]
        self.temperature = 1.0
        self.top_k = 20
        self.top_p = 0.6
        self.stream = False
        self.pad_zero = False
        self.alpha_presence = 1.0
        self.alpha_frequency = 0.1
        self.alpha_decay = 0.996
        self.enable_think = False
        self.chunk_size = 2
        self.password = None
        self.session_id = None

        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyApp:
    def __init__(self):
        self.routes: dict[str, Callable] = {}

    def post(self, path: str):
        def decorator(func):
            self.routes[path] = func
            return func

        return decorator


class DummyRequest:
    def __init__(self, body: dict, headers: dict | None = None):
        self.body = json.dumps(body)
        self.headers = headers or {}


class DummyTokenizer:
    def encode(self, text: str):
        return list(text.encode("utf-8"))


class DummyModel:
    def generate_zero_state(self, index: int):
        return [f"state-{index}", 0, [0]]


class DummyEngine:
    def __init__(self):
        self.args = SimpleNamespace(MODEL_NAME="dummy-model")
        self.rocm_flag = False
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel()
        self.dynamic_calls: list[dict] = []
        self.state_calls: list[dict] = []
        self.dynamic_stream_calls: list[dict] = []
        self.state_stream_calls: list[dict] = []
        self.graph_stream_calls: list[dict] = []

    async def dynamic_batch_generate(self, **kwargs) -> tuple[str, str]:
        self.dynamic_calls.append(kwargs)
        return "stateless-ok", "stop"

    def batch_generate_state(self, **kwargs):
        self.state_calls.append(kwargs)
        return ["stateful-ok"]

    async def dynamic_batch_infer_stream(self, **kwargs):
        self.dynamic_stream_calls.append(kwargs)
        yield {"type": "delta", "text": "stream-stateless"}
        yield {"type": "done", "finish_reason": "stop"}

    async def batch_infer_stream_state(self, **kwargs):
        self.state_stream_calls.append(kwargs)
        yield 'data: {"choices":[{"index":0,"delta":{"content":"stream-stateful"}}]}'
        yield "data: [DONE]"

    async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
        self.graph_stream_calls.append(kwargs)
        yield 'data: {"choices":[{"index":0,"delta":{"content":"stream-graph"}}]}'
        yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        yield "data: [DONE]"


class DummyStateManager:
    def __init__(self, initial_state=None):
        self.state = initial_state
        self.get_calls: list[str] = []
        self.put_calls: list[tuple[str, object]] = []

    def get_state(self, session_id: str):
        self.get_calls.append(session_id)
        return self.state

    def put_state(self, session_id: str, state):
        self.put_calls.append((session_id, state))
        self.state = state


def _assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


async def test_openai_route_stateless_non_stream_uses_graph_aggregation() -> None:
    app = DummyApp()
    engine = DummyEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello there"}],
                "stream": False,
            }
        )
    )
    payload = json.loads(response.description)
    _assert_equal(
        payload["choices"][0]["message"]["content"],
        "stream-graph",
        "stateless response content mismatch",
    )
    _assert_equal(
        len(engine.graph_stream_calls),
        1,
        "stateless route should use graph stream aggregation",
    )
    _assert_equal(
        len(engine.dynamic_calls),
        0,
        "stateless route should not use dynamic batch generate when graph is available",
    )
    _assert_equal(
        len(engine.state_calls), 0, "stateless route should not use state generate"
    )
    print("[PASS] test_openai_route_stateless_non_stream_uses_graph_aggregation")


async def test_openai_route_stateless_non_stream_falls_back_without_graph() -> None:
    app = DummyApp()
    engine = DummyEngine()
    engine.rocm_flag = True
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello there"}],
                "stream": False,
            }
        )
    )
    payload = json.loads(response.description)
    _assert_equal(
        payload["choices"][0]["message"]["content"],
        "stateless-ok",
        "fallback stateless response content mismatch",
    )
    _assert_equal(
        len(engine.dynamic_calls),
        1,
        "stateless route should fall back to dynamic batch generate when graph is unavailable",
    )
    print("[PASS] test_openai_route_stateless_non_stream_falls_back_without_graph")


async def test_openai_route_stateful_non_stream_uses_state_cache() -> None:
    app = DummyApp()
    engine = DummyEngine()
    manager = DummyStateManager(initial_state=None)
    original_get_state_manager = openai_routes.get_state_manager
    openai_routes.get_state_manager = lambda: manager
    try:
        openai_routes.register_openai_routes(
            app, engine, password=None, chat_request_model=ChatRequest
        )
        route = app.routes["/openai/v1/chat/completions"]

        response = await route(
            DummyRequest(
                {
                    "messages": [{"role": "user", "content": "Only new turn"}],
                    "session_id": "sess-1",
                    "stream": False,
                }
            )
        )
    finally:
        openai_routes.get_state_manager = original_get_state_manager

    payload = json.loads(response.description)
    _assert_equal(
        payload["choices"][0]["message"]["content"],
        "stateful-ok",
        "stateful response content mismatch",
    )
    _assert_equal(
        len(engine.dynamic_calls),
        0,
        "stateful route should not use dynamic batch generate",
    )
    _assert_equal(
        len(engine.state_calls), 1, "stateful route should use state generate"
    )
    _assert_equal(manager.get_calls, ["sess-1"], "state manager lookup mismatch")
    _assert_equal(
        len(manager.put_calls), 2, "stateful route should initialize and persist state"
    )
    prompt = engine.state_calls[0]["prompts"][0]
    if "Only new turn" not in prompt:
        raise AssertionError("stateful prompt should include the incremental turn")
    print("[PASS] test_openai_route_stateful_non_stream_uses_state_cache")


async def test_openai_route_stateful_stream_uses_state_stream() -> None:
    app = DummyApp()
    engine = DummyEngine()
    manager = DummyStateManager(initial_state=["existing", 0, [1]])
    original_get_state_manager = openai_routes.get_state_manager
    openai_routes.get_state_manager = lambda: manager
    try:
        openai_routes.register_openai_routes(
            app, engine, password=None, chat_request_model=ChatRequest
        )
        route = app.routes["/openai/v1/chat/completions"]

        response = await route(
            DummyRequest(
                {
                    "messages": [{"role": "user", "content": "Next turn"}],
                    "session_id": "sess-stream",
                    "stream": True,
                }
            )
        )
        chunks = []
        async for chunk in response.iterator:
            chunks.append(chunk)
    finally:
        openai_routes.get_state_manager = original_get_state_manager

    combined = "".join(chunks)
    if "stream-stateful" not in combined or "[DONE]" not in combined:
        raise AssertionError(
            "stateful streaming response should contain content and DONE marker"
        )
    _assert_equal(
        len(engine.state_stream_calls), 1, "stateful streaming should use state stream"
    )
    _assert_equal(
        len(engine.dynamic_stream_calls),
        0,
        "stateful streaming should not use stateless stream",
    )
    print("[PASS] test_openai_route_stateful_stream_uses_state_stream")


async def test_openai_route_stateless_stream_uses_graph_stream() -> None:
    app = DummyApp()
    engine = DummyEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello stream"}],
                "stream": True,
            }
        )
    )
    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if "stream-graph" not in combined or "[DONE]" not in combined:
        raise AssertionError(
            "stateless streaming response should contain graph content and DONE marker"
        )
    if '"finish_reason": "stop"' not in combined:
        raise AssertionError(
            "graph streaming response should propagate stop finish_reason"
        )
    if combined.count("[DONE]") != 1:
        raise AssertionError("stateless streaming should emit a single DONE marker")
    _assert_equal(
        len(engine.graph_stream_calls),
        1,
        "stateless streaming should use graph stream when available",
    )
    _assert_equal(
        len(engine.dynamic_stream_calls),
        0,
        "stateless streaming should not use dynamic stream when graph stream is available",
    )
    _assert_equal(
        engine.graph_stream_calls[0]["chunk_size"],
        1,
        "stateless streaming should use updated default chunk_size",
    )
    print("[PASS] test_openai_route_stateless_stream_uses_graph_stream")


async def test_openai_route_stateless_stream_falls_back_when_graph_not_supported() -> (
    None
):
    app = DummyApp()
    engine = DummyEngine()
    engine.rocm_flag = True
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello fallback"}],
                "stream": True,
            }
        )
    )
    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if "stream-stateless" not in combined:
        raise AssertionError(
            "stateless streaming should fall back to dynamic stream when graph is not supported"
        )
    _assert_equal(
        len(engine.graph_stream_calls),
        0,
        "stateless streaming should not use graph stream when graph is not supported",
    )
    _assert_equal(
        len(engine.dynamic_stream_calls),
        1,
        "stateless streaming should use dynamic stream fallback when graph is not supported",
    )
    print(
        "[PASS] test_openai_route_stateless_stream_falls_back_when_graph_not_supported"
    )


async def test_openai_route_stateless_stream_propagates_graph_length_finish_reason() -> (
    None
):
    app = DummyApp()

    class GraphLengthEngine(DummyEngine):
        async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
            self.graph_stream_calls.append(kwargs)
            yield 'data: {"choices":[{"index":0,"delta":{"content":"stream-graph-length"}}]}'
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}'
            yield "data: [DONE]"

    engine = GraphLengthEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello length"}],
                "stream": True,
            }
        )
    )
    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if '"finish_reason": "length"' not in combined:
        raise AssertionError(
            "graph streaming response should propagate length finish_reason"
        )
    if '"finish_reason": "stop"' in combined:
        raise AssertionError("graph length case should not be rewritten to stop")
    print(
        "[PASS] test_openai_route_stateless_stream_propagates_graph_length_finish_reason"
    )


async def test_openai_route_stateless_stream_skips_malformed_graph_json_payload() -> (
    None
):
    app = DummyApp()

    class MalformedGraphEngine(DummyEngine):
        async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
            self.graph_stream_calls.append(kwargs)
            yield 'data: {"choices":[{"index":0,"delta":{"content":"before-bad-json"}}]}'
            yield 'data: {"choices":['
            yield 'data: {"choices":[{"index":0,"delta":{"content":"after-bad-json"}}]}'
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    engine = MalformedGraphEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello malformed"}],
                "stream": True,
            }
        )
    )
    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if "before-bad-json" not in combined or "after-bad-json" not in combined:
        raise AssertionError(
            "stateless graph stream should preserve valid chunks around malformed JSON"
        )
    if '"finish_reason": "stop"' not in combined or "[DONE]" not in combined:
        raise AssertionError(
            "stateless graph stream should still emit stop finish_reason and DONE after malformed JSON"
        )
    print(
        "[PASS] test_openai_route_stateless_stream_skips_malformed_graph_json_payload"
    )


async def test_openai_route_stateless_stream_respects_chunk_size_override() -> None:
    app = DummyApp()
    engine = DummyEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Hello override"}],
                "stream": True,
                "chunk_size": 7,
            }
        )
    )
    async for _ in response.iterator:
        pass

    _assert_equal(
        engine.graph_stream_calls[0]["chunk_size"],
        7,
        "stateless streaming should honor explicit chunk_size override",
    )
    print("[PASS] test_openai_route_stateless_stream_respects_chunk_size_override")


async def test_openai_route_non_stream_tool_compat_injects_prompt_and_parses_tool_calls() -> (
    None
):
    app = DummyApp()

    class ToolCallEngine(DummyEngine):
        async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
            self.graph_stream_calls.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": '<tool_call>{"name":"lookup_weather","arguments":{"city":"Beijing"}}</tool_call>'
                            },
                        }
                    ]
                }
            )
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    engine = ToolCallEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Use a tool if needed"}],
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "description": "Get weather.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    payload = json.loads(response.description)
    _assert_equal(response.status_code, 200, "tool compat request should succeed")
    _assert_equal(
        len(engine.graph_stream_calls), 1, "tool compat should use graph aggregation"
    )
    prompt = engine.graph_stream_calls[0]["inputs"][0]
    if "Available tools:" not in prompt or "Tool: lookup_weather" not in prompt:
        raise AssertionError(
            "tool compat should inject tools into the formatted system prompt"
        )
    message = payload["choices"][0]["message"]
    if message.get("tool_calls") is None:
        raise AssertionError("non-stream tool compat should return tool_calls")
    _assert_equal(
        payload["choices"][0]["finish_reason"],
        "tool_calls",
        "non-stream tool compat finish reason mismatch",
    )
    print(
        "[PASS] test_openai_route_non_stream_tool_compat_injects_prompt_and_parses_tool_calls"
    )


async def test_openai_route_stream_tool_compat_emits_tool_call_sse() -> None:
    app = DummyApp()

    class StreamingToolCallEngine(DummyEngine):
        async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
            self.graph_stream_calls.append(kwargs)
            for piece in [
                "<tool_",
                'call>{"name":"lookup_weather",',
                '"arguments":{"city":"Beijing"}}',
                "</tool_call>",
            ]:
                yield "data: " + json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": piece}}]}
                )
            yield "data: [DONE]"

    engine = StreamingToolCallEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Use a tool if needed"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if '"tool_calls"' not in combined or "lookup_weather" not in combined:
        raise AssertionError("stream tool compat should emit tool_calls chunks")
    if '"finish_reason": "tool_calls"' not in combined:
        raise AssertionError("stream tool compat should finish with tool_calls")
    if "<tool_call>" in combined:
        raise AssertionError("raw tool_call tags should not leak into SSE output")
    print("[PASS] test_openai_route_stream_tool_compat_emits_tool_call_sse")


async def test_openai_route_stream_tool_compat_preserves_trailing_text_graph() -> None:
    app = DummyApp()

    class StreamingToolCallTrailingEngine(DummyEngine):
        async def graph_infer_stream(self, **kwargs) -> AsyncIterator[str]:
            self.graph_stream_calls.append(kwargs)
            for piece in [
                "<tool_call>",
                '{"name":"lookup_weather","arguments":{"city":"Beijing"}}',
                "</tool",
                "_call>",
                " trailing explanation",
            ]:
                yield "data: " + json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": piece}}]}
                )
            yield "data: [DONE]"

    engine = StreamingToolCallTrailingEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Use a tool"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if "trailing explanation" not in combined:
        raise AssertionError("trailing text should be preserved after tool call")
    if '"tool_calls"' not in combined or "lookup_weather" not in combined:
        raise AssertionError("tool compat should emit tool_calls chunks")
    if '"finish_reason": "tool_calls"' not in combined:
        raise AssertionError("tool compat should finish with tool_calls")
    if "<tool_call>" in combined:
        raise AssertionError("raw tool_call tags should not leak into SSE output")
    print("[PASS] test_openai_route_stream_tool_compat_preserves_trailing_text_graph")


async def test_openai_route_stream_tool_compat_false_discards_tools() -> None:
    app = DummyApp()
    engine = DummyEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Ignore tools"}],
                "stream": False,
                "rwkv_tool_compat_mode": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    payload = json.loads(response.description)
    prompt = engine.graph_stream_calls[0]["inputs"][0]
    if "Available tools:" in prompt:
        raise AssertionError("false compat mode should discard tools without injection")
    _assert_equal(response.status_code, 200, "false compat request should succeed")
    _assert_equal(
        payload["choices"][0]["message"]["content"],
        "stream-graph",
        "false compat response content mismatch",
    )
    print("[PASS] test_openai_route_stream_tool_compat_false_discards_tools")


async def test_openai_route_stream_tool_compat_emits_tool_call_sse_on_fallback() -> (
    None
):
    app = DummyApp()

    class FallbackStreamingToolCallEngine(DummyEngine):
        def __init__(self):
            super().__init__()
            self.rocm_flag = True

        async def dynamic_batch_infer_stream(self, **kwargs):
            self.dynamic_stream_calls.append(kwargs)
            for piece in [
                "<tool_",
                'call>{"name":"lookup_weather",',
                '"arguments":{"city":"Beijing"}}',
                "</tool_call>",
            ]:
                yield {"type": "delta", "text": piece}
            yield {"type": "done", "finish_reason": "stop"}

    engine = FallbackStreamingToolCallEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Use a fallback tool stream"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if '"tool_calls"' not in combined or "lookup_weather" not in combined:
        raise AssertionError("fallback tool compat should emit tool_calls chunks")
    if '"finish_reason": "tool_calls"' not in combined:
        raise AssertionError("fallback tool compat should finish with tool_calls")
    if "<tool_call>" in combined:
        raise AssertionError("fallback stream should not leak raw tool_call tags")
    print("[PASS] test_openai_route_stream_tool_compat_emits_tool_call_sse_on_fallback")


async def test_openai_route_stream_tool_compat_preserves_trailing_text_fallback() -> (
    None
):
    app = DummyApp()

    class FallbackStreamingToolCallTrailingEngine(DummyEngine):
        def __init__(self):
            super().__init__()
            self.rocm_flag = True

        async def dynamic_batch_infer_stream(self, **kwargs):
            self.dynamic_stream_calls.append(kwargs)
            for piece in [
                "<tool_call>",
                '{"name":"lookup_weather","arguments":{"city":"Beijing"}}',
                "</tool",
                "_call>",
                " trailing note",
            ]:
                yield {"type": "delta", "text": piece}
            yield {"type": "done", "finish_reason": "stop"}

    engine = FallbackStreamingToolCallTrailingEngine()
    openai_routes.register_openai_routes(
        app, engine, password=None, chat_request_model=ChatRequest
    )
    route = app.routes["/openai/v1/chat/completions"]

    response = await route(
        DummyRequest(
            {
                "messages": [{"role": "user", "content": "Use a tool"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    chunks = []
    async for chunk in response.iterator:
        chunks.append(chunk)

    combined = "".join(chunks)
    if "trailing note" not in combined:
        raise AssertionError("fallback trailing text should be preserved")
    if '"tool_calls"' not in combined or "lookup_weather" not in combined:
        raise AssertionError("fallback tool compat should emit tool_calls chunks")
    if '"finish_reason": "tool_calls"' not in combined:
        raise AssertionError("fallback tool compat should finish with tool_calls")
    if "<tool_call>" in combined:
        raise AssertionError("fallback stream should not leak raw tool_call tags")
    print(
        "[PASS] test_openai_route_stream_tool_compat_preserves_trailing_text_fallback"
    )


async def main() -> None:
    await test_openai_route_stateless_non_stream_uses_graph_aggregation()
    await test_openai_route_stateless_non_stream_falls_back_without_graph()
    await test_openai_route_stateful_non_stream_uses_state_cache()
    await test_openai_route_stateful_stream_uses_state_stream()
    await test_openai_route_stateless_stream_uses_graph_stream()
    await test_openai_route_stateless_stream_falls_back_when_graph_not_supported()
    await test_openai_route_stateless_stream_propagates_graph_length_finish_reason()
    await test_openai_route_stateless_stream_skips_malformed_graph_json_payload()
    await test_openai_route_stateless_stream_respects_chunk_size_override()
    await (
        test_openai_route_non_stream_tool_compat_injects_prompt_and_parses_tool_calls()
    )
    await test_openai_route_stream_tool_compat_emits_tool_call_sse()
    await test_openai_route_stream_tool_compat_preserves_trailing_text_graph()
    await test_openai_route_stream_tool_compat_false_discards_tools()
    await test_openai_route_stream_tool_compat_emits_tool_call_sse_on_fallback()
    await test_openai_route_stream_tool_compat_preserves_trailing_text_fallback()
    print("All openai_routes tests passed.")


if __name__ == "__main__":
    asyncio.run(main())

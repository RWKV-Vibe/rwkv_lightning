#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request


def request_json(method: str, url: str, payload=None, *, timeout: int = 60, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(url, data=data, headers=merged_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, raw, dict(response.headers)


def request_stream(url: str, payload, *, timeout: int = 60, headers=None):
    merged_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=merged_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            chunks.append(line)
            if line == "data: [DONE]":
                break
        return response.status, chunks, dict(response.headers)


def print_result(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def parse_error_body(exc: urllib.error.HTTPError):
    raw = exc.read().decode("utf-8", errors="ignore")
    return exc.code, raw, json.loads(raw)


def test_invalid_auth(base_url: str):
    payload = {"model": "rwkv7", "messages": [{"role": "user", "content": "hi"}]}
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": "Bearer wrong"},
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = (
            status == 401
            and data.get("error", {}).get("type") == "authentication_error"
            and data.get("error", {}).get("code") == "invalid_api_key"
        )
        print_result("invalid auth", ok, raw[:160])
        return ok
    print_result("invalid auth", False, "request unexpectedly succeeded")
    return False


def test_invalid_model(base_url: str, password: str):
    payload = {
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 404 and data.get("error", {}).get("code") == "model_not_found"
        print_result("invalid model", ok, raw[:160])
        return ok
    print_result("invalid model", False, "request unexpectedly succeeded")
    return False


def test_invalid_runtime(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "runtime": "int4",
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and data.get("error", {}).get("param") == "runtime"
        print_result("invalid runtime", ok, raw[:160])
        return ok
    print_result("invalid runtime", False, "request unexpectedly succeeded")
    return False


def test_models_list(base_url: str, password: str):
    status, raw, _ = request_json(
        "GET",
        f"{base_url}/openai/v1/models",
        headers={"Authorization": f"Bearer {password}"},
        timeout=60,
    )
    data = json.loads(raw)
    models = data.get("data") or []
    first = models[0] if models else {}
    ok = (
        status == 200
        and data.get("object") == "list"
        and bool(models)
        and first.get("object") == "model"
        and isinstance(first.get("id"), str)
        and isinstance(first.get("created"), int)
        and isinstance(first.get("owned_by"), str)
    )
    print_result("models list", ok, raw[:160])
    return ok, first.get("id")


def test_model_retrieve(base_url: str, password: str, model_id: str):
    status, raw, _ = request_json(
        "GET",
        f"{base_url}/openai/v1/models/{model_id}",
        headers={"Authorization": f"Bearer {password}"},
        timeout=60,
    )
    data = json.loads(raw)
    ok = (
        status == 200
        and data.get("id") == model_id
        and data.get("object") == "model"
        and isinstance(data.get("created"), int)
        and isinstance(data.get("owned_by"), str)
    )
    print_result("model retrieve", ok, raw[:160])
    return ok


def test_models_endpoints(base_url: str, password: str):
    ok, model_id = test_models_list(base_url, password)
    if not ok or not model_id:
        return False
    return test_model_retrieve(base_url, password, model_id)


def test_multimodal_rejection(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
        "max_tokens": 8,
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "text content parts" in data.get("error", {}).get("message", "")
        print_result("multimodal rejection", ok, raw[:160])
        return ok
    print_result("multimodal rejection", False, "request unexpectedly succeeded")
    return False


def test_reasoning_effort(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "reasoning_effort": "medium",
        "max_tokens": 8,
        "temperature": 0.1,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=120,
    )
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("choices"))
    print_result("reasoning_effort", ok, raw[:160])
    return ok


def test_invalid_reasoning_effort(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "turbo",
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "reasoning_effort" in data.get("error", {}).get("message", "")
        print_result("invalid reasoning_effort", ok, raw[:160])
        return ok
    print_result("invalid reasoning_effort", False, "request unexpectedly succeeded")
    return False


def test_service_tier(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "service_tier": "priority",
        "max_tokens": 8,
        "temperature": 0.1,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=120,
    )
    data = json.loads(raw)
    ok = status == 200 and data.get("service_tier") == "default"
    print_result("service_tier", ok, raw[:160])
    return ok


def test_invalid_service_tier(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "instant",
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "service_tier" in data.get("error", {}).get("message", "")
        print_result("invalid service_tier", ok, raw[:160])
        return ok
    print_result("invalid service_tier", False, "request unexpectedly succeeded")
    return False


def test_metadata_store_user(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "metadata": {"suite": "contract", "case": "metadata"},
        "store": False,
        "user": "contract-user",
        "max_tokens": 8,
        "temperature": 0.1,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=120,
    )
    ok = status == 200
    print_result("metadata/store/user", ok, raw[:160])
    return ok


def test_invalid_metadata(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"suite": 123},
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "metadata" in data.get("error", {}).get("message", "")
        print_result("invalid metadata", ok, raw[:160])
        return ok
    print_result("invalid metadata", False, "request unexpectedly succeeded")
    return False


def test_logit_bias_validation(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "logit_bias": {"abc": 1},
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "logit_bias" in data.get("error", {}).get("message", "")
        print_result("invalid logit_bias", ok, raw[:160])
        return ok
    print_result("invalid logit_bias", False, "request unexpectedly succeeded")
    return False


def test_logit_bias(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with a single token if possible."}],
        "logit_bias": {"0": -100},
        "max_tokens": 4,
        "temperature": 0.2,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=120,
    )
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("choices"))
    print_result("logit_bias", ok, raw[:160])
    return ok


def test_parallel_tool_calls_validation(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "parallel_tool_calls": "yes",
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "parallel_tool_calls" in data.get("error", {}).get("message", "")
        print_result("invalid parallel_tool_calls", ok, raw[:160])
        return ok
    print_result("invalid parallel_tool_calls", False, "request unexpectedly succeeded")
    return False


def test_missing_tool_call_id(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "{}"},
            {"role": "user", "content": "Continue."},
        ],
        "max_tokens": 8,
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "tool_call_id" in data.get("error", {}).get("message", "")
        print_result("missing tool_call_id", ok, raw[:160])
        return ok
    print_result("missing tool_call_id", False, "request unexpectedly succeeded")
    return False


def test_unknown_tool_call_id(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_known",
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_unknown", "content": "{}"},
            {"role": "user", "content": "Continue."},
        ],
        "max_tokens": 8,
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        status, raw, data = parse_error_body(exc)
        ok = status == 400 and "Unknown tool_call_id" in data.get("error", {}).get("message", "")
        print_result("unknown tool_call_id", ok, raw[:160])
        return ok
    print_result("unknown tool_call_id", False, "request unexpectedly succeeded")
    return False


def test_parallel_tool_calls_non_stream(base_url: str, password: str, model_name: str):
    if importlib.util.find_spec("xgrammar") is None:
        print_result("parallel tool_calls non-stream", True, "skipped: xgrammar not installed")
        return True
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Call add_numbers with a=2 and b=3, and create_task with title='Buy milk'."}],
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_task",
                    "description": "Create a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "max_tokens": 96,
        "temperature": 0.2,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=240,
    )
    data = json.loads(raw)
    tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    names = {item.get("function", {}).get("name") for item in tool_calls}
    ok = status == 200 and len(tool_calls) >= 2 and {"add_numbers", "create_task"}.issubset(names)
    print_result("parallel tool_calls non-stream", ok, raw[:160])
    return ok


def test_parallel_tool_calls_stream(base_url: str, password: str, model_name: str):
    if importlib.util.find_spec("xgrammar") is None:
        print_result("parallel tool_calls stream", True, "skipped: xgrammar not installed")
        return True
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Call add_numbers with a=2 and b=3, and create_task with title='Buy milk'."}],
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_task",
                    "description": "Create a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "max_tokens": 96,
        "temperature": 0.2,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=240,
        headers={"Authorization": f"Bearer {password}"},
    )
    seen_tool_indices = set()
    seen_ids = set()
    saw_finish = False
    for chunk in chunks:
        if not chunk.startswith("data: {"):
            continue
        data = json.loads(chunk[6:])
        for choice in data.get("choices", []):
            for tool_delta in choice.get("delta", {}).get("tool_calls", []):
                seen_tool_indices.add(tool_delta.get("index"))
                if tool_delta.get("id"):
                    seen_ids.add(tool_delta.get("id"))
            if choice.get("finish_reason") == "tool_calls":
                saw_finish = True
    ok = status == 200 and len(seen_tool_indices) >= 2 and len(seen_ids) >= 2 and saw_finish
    print_result("parallel tool_calls stream", ok, ",".join(map(str, sorted(i for i in seen_tool_indices if i is not None))))
    return ok


def test_parallel_tool_calls_false(base_url: str, password: str, model_name: str):
    if importlib.util.find_spec("xgrammar") is None:
        print_result("parallel tool_calls false", True, "skipped: xgrammar not installed")
        return True
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Call add_numbers with a=2 and b=3, and create_task with title='Buy milk'."}],
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_task",
                    "description": "Create a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "max_tokens": 96,
        "temperature": 0.2,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=240,
    )
    data = json.loads(raw)
    tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    ok = status == 200 and len(tool_calls) <= 1
    print_result("parallel tool_calls false", ok, raw[:160])
    return ok


def test_n_validation(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "n": 2,
        "messages": [{"role": "user", "content": "say hi"}],
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=180,
    )
    data = json.loads(raw)
    choices = data.get("choices") or []
    indices = sorted(choice.get("index") for choice in choices)
    ok = status == 200 and len(choices) == 2 and indices == [0, 1]
    print_result("n > 1", ok, raw[:160])
    return ok


def test_tools_validation(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Call the function add_numbers with a=2 and b=3."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False
                }
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "add_numbers"}},
        "max_tokens": 48,
        "temperature": 0.2,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=180,
    )
    data = json.loads(raw)
    message = data.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    arguments = {}
    if tool_calls:
        try:
            arguments = json.loads(tool_calls[0].get("function", {}).get("arguments", "{}"))
        except Exception:
            arguments = {}
    ok = (
        status == 200
        and data.get("choices", [{}])[0].get("finish_reason") == "tool_calls"
        and len(tool_calls) == 1
        and tool_calls[0].get("function", {}).get("name") == "add_numbers"
        and arguments.get("a") == 2
        and arguments.get("b") == 3
    )
    print_result("tool calling", ok, raw[:160])
    return ok


def test_logprobs(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with a single short greeting."}],
        "logprobs": True,
        "top_logprobs": 3,
        "max_tokens": 8,
        "temperature": 0.4,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {password}"},
        timeout=180,
    )
    data = json.loads(raw)
    logprobs = data.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
    ok = status == 200 and bool(logprobs) and all(len(item.get("top_logprobs", [])) <= 3 for item in logprobs)
    print_result("logprobs", ok, raw[:160])
    return ok


def test_stream_n(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Give two very short greetings."}],
        "n": 2,
        "stream": True,
        "max_tokens": 8,
        "temperature": 0.3,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    seen_indices = set()
    for chunk in chunks:
        if chunk.startswith("data: {"):
            data = json.loads(chunk[6:])
            for choice in data.get("choices", []):
                if "index" in choice:
                    seen_indices.add(choice["index"])
    ok = status == 200 and seen_indices == {0, 1}
    print_result("stream n > 1", ok, ",".join(map(str, sorted(seen_indices))))
    return ok


def test_stream_logprobs(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "stream": True,
        "logprobs": True,
        "top_logprobs": 2,
        "max_tokens": 8,
        "temperature": 0.3,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    saw_logprobs = False
    for chunk in chunks:
        if not chunk.startswith("data: {"):
            continue
        data = json.loads(chunk[6:])
        for choice in data.get("choices", []):
            if choice.get("logprobs", {}).get("content"):
                saw_logprobs = True
    ok = status == 200 and saw_logprobs
    print_result("stream logprobs", ok, chunks[1][:160] if len(chunks) > 1 else "")
    return ok


def test_stream_tools(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Call the function create_task with title='Buy milk'."}],
        "stream": True,
        "tools": [{
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a task.",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "create_task"}},
        "max_tokens": 48,
        "temperature": 0.2,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=240,
        headers={"Authorization": f"Bearer {password}"},
    )
    saw_tool_name = False
    saw_argument_fragments = []
    saw_finish = False
    for chunk in chunks:
        if not chunk.startswith("data: {"):
            continue
        data = json.loads(chunk[6:])
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            for tool_delta in delta.get("tool_calls", []):
                function = tool_delta.get("function", {})
                if function.get("name") == "create_task":
                    saw_tool_name = True
                if "arguments" in function:
                    saw_argument_fragments.append(function["arguments"])
            if choice.get("finish_reason") == "tool_calls":
                saw_finish = True
    joined_arguments = "".join(saw_argument_fragments)
    ok = (
        status == 200
        and saw_tool_name
        and len(saw_argument_fragments) >= 2
        and '"title"' in joined_arguments
        and 'Buy milk' in joined_arguments
        and saw_finish
    )
    print_result("stream tools", ok, chunks[-2][:160] if len(chunks) >= 2 else "")
    return ok


def test_stream_logprobs_n(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Give two short greetings."}],
        "stream": True,
        "n": 2,
        "logprobs": True,
        "top_logprobs": 2,
        "max_tokens": 8,
        "temperature": 0.3,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=240,
        headers={"Authorization": f"Bearer {password}"},
    )
    seen_indices = set()
    logprob_indices = set()
    for chunk in chunks:
        if not chunk.startswith("data: {"):
            continue
        data = json.loads(chunk[6:])
        for choice in data.get("choices", []):
            if "index" in choice:
                seen_indices.add(choice["index"])
            if choice.get("logprobs", {}).get("content"):
                logprob_indices.add(choice.get("index"))
    ok = status == 200 and seen_indices == {0, 1} and logprob_indices == {0, 1}
    print_result("stream logprobs n>1", ok, ",".join(map(str, sorted(logprob_indices))))
    return ok


def test_tool_choice_auto_answer(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up weather.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 8,
        "temperature": 0.1,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    data = json.loads(raw)
    message = data.get("choices", [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    ok = status == 200 and not message.get("tool_calls") and "4" in content
    print_result("tool_choice auto answer", ok, raw[:160])
    return ok


def test_tool_choice_auto_tool(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Use the weather tool to get weather for Paris."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 48,
        "temperature": 0.2,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    data = json.loads(raw)
    message = data.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    arguments = {}
    if tool_calls:
        arguments = json.loads(tool_calls[0].get("function", {}).get("arguments", "{}"))
    ok = status == 200 and tool_calls and tool_calls[0].get("function", {}).get("name") == "lookup_weather" and arguments.get("city") == "Paris"
    print_result("tool_choice auto tool", ok, raw[:160])
    return ok


def test_stop_and_penalties(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply briefly."}],
        "stop": ["\nUser:"],
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
        "temperature": 0.4,
        "max_tokens": 12,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=120,
        headers={"Authorization": f"Bearer {password}"},
    )
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("choices"))
    print_result("stop + penalties", ok, raw[:160])
    return ok


def test_seed_determinism(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Give me three short words."}],
        "seed": 1234,
        "temperature": 0.8,
        "max_tokens": 8,
    }
    first_status, first_raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=120,
        headers={"Authorization": f"Bearer {password}"},
    )
    second_status, second_raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=120,
        headers={"Authorization": f"Bearer {password}"},
    )
    first = json.loads(first_raw)
    second = json.loads(second_raw)
    first_text = first.get("choices", [{}])[0].get("message", {}).get("content", "")
    second_text = second.get("choices", [{}])[0].get("message", {}).get("content", "")
    ok = first_status == 200 and second_status == 200 and first_text == second_text
    print_result("seed determinism", ok, first_text[:120])
    return ok


def test_stream_usage(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Count to two."}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 8,
        "temperature": 0.2,
    }
    status, chunks, _ = request_stream(
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=120,
        headers={"Authorization": f"Bearer {password}"},
    )
    usage_chunk = None
    for chunk in chunks:
        if chunk.startswith("data: {"):
            payload = json.loads(chunk[6:])
            if payload.get("choices") == [] and payload.get("usage"):
                usage_chunk = payload
                break
    ok = status == 200 and usage_chunk is not None
    print_result("stream usage", ok, json.dumps(usage_chunk or {})[:160])
    return ok


def test_json_response_format(base_url: str, password: str, model_name: str):
    if importlib.util.find_spec("xgrammar") is None:
        print_result("json_object response_format", True, "skipped: xgrammar not installed")
        return True

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Return a JSON object with an ok boolean."}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 32,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    data = json.loads(raw)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        ok = status == 200 and isinstance(parsed, dict)
    except Exception:
        parsed = None
        ok = False
    print_result("json_object response_format", ok, content[:160])
    return ok


def test_json_schema_response_format(base_url: str, password: str, model_name: str):
    if importlib.util.find_spec("xgrammar") is None:
        print_result("json_schema response_format", True, "skipped: xgrammar not installed")
        return True

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Return a JSON object with a required string field named name."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        "temperature": 0.2,
        "max_tokens": 48,
    }
    status, raw, _ = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    data = json.loads(raw)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        ok = status == 200 and isinstance(parsed, dict) and isinstance(parsed.get("name"), str)
    except Exception:
        ok = False
    print_result("json_schema response_format", ok, content[:160])
    return ok


def main():
    parser = argparse.ArgumentParser(description="Contract tests for OpenAI compatibility route")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--password", default="rwkv_test")
    parser.add_argument(
        "--model",
        default="rwkv7-g1f-7.2b-20260414-ctx8192:fp16",
    )
    args = parser.parse_args()

    tests = [
        ("invalid auth", lambda: test_invalid_auth(args.base_url)),
        ("models endpoints", lambda: test_models_endpoints(args.base_url, args.password)),
        ("invalid model", lambda: test_invalid_model(args.base_url, args.password)),
        ("invalid runtime", lambda: test_invalid_runtime(args.base_url, args.password, args.model)),
        ("multimodal rejection", lambda: test_multimodal_rejection(args.base_url, args.password, args.model)),
        ("reasoning_effort", lambda: test_reasoning_effort(args.base_url, args.password, args.model)),
        ("invalid reasoning_effort", lambda: test_invalid_reasoning_effort(args.base_url, args.password, args.model)),
        ("service_tier", lambda: test_service_tier(args.base_url, args.password, args.model)),
        ("invalid service_tier", lambda: test_invalid_service_tier(args.base_url, args.password, args.model)),
        ("metadata/store/user", lambda: test_metadata_store_user(args.base_url, args.password, args.model)),
        ("invalid metadata", lambda: test_invalid_metadata(args.base_url, args.password, args.model)),
        ("logit_bias", lambda: test_logit_bias(args.base_url, args.password, args.model)),
        ("invalid logit_bias", lambda: test_logit_bias_validation(args.base_url, args.password, args.model)),
        ("invalid parallel_tool_calls", lambda: test_parallel_tool_calls_validation(args.base_url, args.password, args.model)),
        ("missing tool_call_id", lambda: test_missing_tool_call_id(args.base_url, args.password, args.model)),
        ("unknown tool_call_id", lambda: test_unknown_tool_call_id(args.base_url, args.password, args.model)),
        ("n > 1", lambda: test_n_validation(args.base_url, args.password, args.model)),
        ("tool calling", lambda: test_tools_validation(args.base_url, args.password, args.model)),
        ("parallel tool_calls non-stream", lambda: test_parallel_tool_calls_non_stream(args.base_url, args.password, args.model)),
        ("parallel tool_calls false", lambda: test_parallel_tool_calls_false(args.base_url, args.password, args.model)),
        ("logprobs", lambda: test_logprobs(args.base_url, args.password, args.model)),
        ("stream n > 1", lambda: test_stream_n(args.base_url, args.password, args.model)),
        ("stream logprobs", lambda: test_stream_logprobs(args.base_url, args.password, args.model)),
        ("stream logprobs n>1", lambda: test_stream_logprobs_n(args.base_url, args.password, args.model)),
        ("stream tools", lambda: test_stream_tools(args.base_url, args.password, args.model)),
        ("parallel tool_calls stream", lambda: test_parallel_tool_calls_stream(args.base_url, args.password, args.model)),
        ("tool_choice auto answer", lambda: test_tool_choice_auto_answer(args.base_url, args.password, args.model)),
        ("tool_choice auto tool", lambda: test_tool_choice_auto_tool(args.base_url, args.password, args.model)),
        ("stop + penalties", lambda: test_stop_and_penalties(args.base_url, args.password, args.model)),
        ("seed determinism", lambda: test_seed_determinism(args.base_url, args.password, args.model)),
        ("stream usage", lambda: test_stream_usage(args.base_url, args.password, args.model)),
        ("json_object response_format", lambda: test_json_response_format(args.base_url, args.password, args.model)),
        ("json_schema response_format", lambda: test_json_schema_response_format(args.base_url, args.password, args.model)),
    ]

    failures = []
    for name, fn in tests:
        try:
            if not fn():
                failures.append(name)
        except Exception as exc:
            print_result(name, False, str(exc))
            failures.append(name)

    if failures:
        print(f"\nOpenAI contract tests failed: {', '.join(failures)}")
        return 1

    print("\nOpenAI contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

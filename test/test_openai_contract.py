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


def test_stream_logprobs_rejected(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "logprobs": True,
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
        )
    except urllib.error.HTTPError as exc:
        status, raw, _ = parse_error_body(exc)
        ok = status == 400
        print_result("stream logprobs rejected", ok, raw[:160])
        return ok
    print_result("stream logprobs rejected", False, "request unexpectedly succeeded")
    return False


def test_stream_tools_rejected(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "call foo"}],
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object"}}}],
    }
    try:
        request_json(
            "POST",
            f"{base_url}/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {password}"},
        )
    except urllib.error.HTTPError as exc:
        status, raw, _ = parse_error_body(exc)
        ok = status == 400
        print_result("stream tools rejected", ok, raw[:160])
        return ok
    print_result("stream tools rejected", False, "request unexpectedly succeeded")
    return False


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
        ("invalid model", lambda: test_invalid_model(args.base_url, args.password)),
        ("invalid runtime", lambda: test_invalid_runtime(args.base_url, args.password, args.model)),
        ("n > 1", lambda: test_n_validation(args.base_url, args.password, args.model)),
        ("tool calling", lambda: test_tools_validation(args.base_url, args.password, args.model)),
        ("logprobs", lambda: test_logprobs(args.base_url, args.password, args.model)),
        ("stream n > 1", lambda: test_stream_n(args.base_url, args.password, args.model)),
        ("stream logprobs rejected", lambda: test_stream_logprobs_rejected(args.base_url, args.password, args.model)),
        ("stream tools rejected", lambda: test_stream_tools_rejected(args.base_url, args.password, args.model)),
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

#!/usr/bin/env python3

import argparse
import json
import time
import urllib.request


def request_json(method: str, url: str, payload=None, *, timeout: int = 60, headers=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=merged_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw)


def print_result(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def semantic_math(base_url: str, password: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+3? Reply with only the number."}],
        "max_tokens": 6,
        "temperature": 0.1,
    }
    status, data = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=120,
        headers={"Authorization": f"Bearer {password}"},
    )
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    ok = status == 200 and "5" in text
    print_result("semantic math", ok, text[:120])
    return ok


def semantic_stateful_memory(base_url: str, password: str, model: str):
    session_id = f"semantic-{int(time.time())}"
    keyword = "neon-cactus-17"
    first_payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Remember this exact keyword: {keyword}. Reply only with OK."}],
        "max_tokens": 8,
        "temperature": 0.1,
        "password": password,
        "session_id": session_id,
    }
    second_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What exact keyword did I ask you to remember? Reply with only that keyword."}],
        "max_tokens": 16,
        "temperature": 0.1,
        "password": password,
        "session_id": session_id,
    }
    first_status, first_data = request_json(
        "POST", f"{base_url}/state/chat/completions", first_payload, timeout=120
    )
    second_status, second_data = request_json(
        "POST", f"{base_url}/state/chat/completions", second_payload, timeout=120
    )
    first_text = first_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    second_text = second_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    ok = first_status == 200 and second_status == 200 and keyword in second_text
    print_result("semantic stateful memory", ok, second_text[:120])
    return ok


def semantic_json_schema(base_url: str, password: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Classify the sentiment of: I absolutely love this movie."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["sentiment", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "max_tokens": 64,
        "temperature": 0.1,
    }
    status, data = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = json.loads(content)
    ok = status == 200 and parsed.get("sentiment") == "positive" and isinstance(parsed.get("reason"), str)
    print_result("semantic json schema", ok, content[:120])
    return ok


def semantic_tool_call(base_url: str, password: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Call the function create_reminder with title='Pay rent' and day=3."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_reminder",
                    "description": "Create a reminder.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "day": {"type": "integer"},
                        },
                        "required": ["title", "day"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "create_reminder"}},
        "max_tokens": 64,
        "temperature": 0.1,
    }
    status, data = request_json(
        "POST",
        f"{base_url}/openai/v1/chat/completions",
        payload,
        timeout=180,
        headers={"Authorization": f"Bearer {password}"},
    )
    message = data.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    arguments = {}
    if tool_calls:
        arguments = json.loads(tool_calls[0].get("function", {}).get("arguments", "{}"))
    ok = (
        status == 200
        and data.get("choices", [{}])[0].get("finish_reason") == "tool_calls"
        and len(tool_calls) == 1
        and tool_calls[0].get("function", {}).get("name") == "create_reminder"
        and arguments.get("title") == "Pay rent"
        and arguments.get("day") == 3
    )
    print_result("semantic tool call", ok, json.dumps(arguments, ensure_ascii=False)[:120])
    return ok


def main():
    parser = argparse.ArgumentParser(description="Semantic quality checks for RWKV Lightning")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--password", default="rwkv_test")
    parser.add_argument("--model", default="rwkv7-g1f-7.2b-20260414-ctx8192:fp16")
    args = parser.parse_args()

    tests = [
        semantic_math,
        semantic_stateful_memory,
        semantic_json_schema,
        semantic_tool_call,
    ]

    failures = []
    for test_fn in tests:
        try:
            if not test_fn(args.base_url, args.password, args.model):
                failures.append(test_fn.__name__)
        except Exception as exc:
            print_result(test_fn.__name__, False, str(exc))
            failures.append(test_fn.__name__)

    if failures:
        print(f"\nSemantic quality suite failed: {', '.join(failures)}")
        return 1

    print("\nSemantic quality suite passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request_json(method: str, url: str, payload=None, *, timeout: int = 60, headers=None):
    body = None
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=merged_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, raw, dict(response.headers)


def request_stream(url: str, payload, *, timeout: int = 60):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            chunks.append(line)
            if line == "data: [DONE]" or len(chunks) >= 6:
                break
        return response.status, chunks, dict(response.headers)


def print_result(name: str, ok: bool, details: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f" - {details}" if details else ""
    print(f"[{status}] {name}{suffix}")


def test_healthz(base_url: str):
    status, raw, _ = request_json("GET", f"{base_url}/healthz", timeout=15)
    data = json.loads(raw)
    ok = status == 200 and data.get("status") == "ok"
    print_result("healthz", ok, raw[:160])
    return ok


def test_models(base_url: str):
    status, raw, _ = request_json("GET", f"{base_url}/v1/models", timeout=15)
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("data"))
    first_model = data.get("data", [{}])[0].get("id", "")
    print_result("v1/models", ok, first_model)
    return ok, first_model


def test_v1_non_stream(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "contents": ["User: Reply with exactly the word ok.\n\nAssistant:"],
        "max_tokens": 8,
        "temperature": 0.1,
        "stream": False,
        "password": password,
    }
    status, raw, _ = request_json("POST", f"{base_url}/v1/chat/completions", payload, timeout=120)
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("choices"))
    detail = raw[:160]
    print_result("v1/chat non-stream", ok, detail)
    return ok


def test_v1_stream(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "contents": ["User: Count to 2.\n\nAssistant:"],
        "max_tokens": 8,
        "temperature": 0.1,
        "stream": True,
        "password": password,
    }
    status, chunks, _ = request_stream(f"{base_url}/v1/chat/completions", payload, timeout=120)
    ok = status == 200 and any(chunk.startswith("data: ") for chunk in chunks)
    print_result("v1/chat stream", ok, " | ".join(chunks[:3]))
    return ok


def test_v2_non_stream(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "scheduler": "continuous",
        "contents": ["User: Say hi.\n\nAssistant:"],
        "max_tokens": 8,
        "temperature": 0.1,
        "stream": False,
        "password": password,
    }
    status, raw, _ = request_json("POST", f"{base_url}/v2/chat/completions", payload, timeout=120)
    data = json.loads(raw)
    ok = status == 200 and bool(data.get("choices"))
    print_result("v2/chat non-stream", ok, raw[:160])
    return ok


def test_stateful(base_url: str, password: str, model_name: str):
    session_id = f"smoke-{int(time.time())}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Remember the number 7 and say ok."}],
        "max_tokens": 8,
        "temperature": 0.1,
        "password": password,
        "session_id": session_id,
    }
    status, raw, _ = request_json("POST", f"{base_url}/state/chat/completions", payload, timeout=120)
    ok = status == 200
    print_result("state/chat/completions", ok, raw[:160])
    return ok


def test_openai_non_stream(base_url: str, password: str, model_name: str):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "temperature": 0.1,
        "stream": False,
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
    print_result("openai/chat non-stream", ok, raw[:160])
    return ok


def main():
    parser = argparse.ArgumentParser(description="Smoke test RWKV Lightning API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--password", default="rwkv_test")
    args = parser.parse_args()

    failures = []

    try:
        ok = test_healthz(args.base_url)
        if not ok:
            failures.append("healthz")

        ok, model_name = test_models(args.base_url)
        if not ok:
            failures.append("v1/models")
            model_name = "rwkv7"

        for name, fn in [
            ("v1/chat non-stream", test_v1_non_stream),
            ("v1/chat stream", test_v1_stream),
            ("v2/chat non-stream", test_v2_non_stream),
            ("state/chat/completions", test_stateful),
            ("openai/chat non-stream", test_openai_non_stream),
        ]:
            if not fn(args.base_url, args.password, model_name):
                failures.append(name)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        print_result("unexpected http error", False, f"{exc.code}: {body[:200]}")
        failures.append("unexpected http error")
    except Exception as exc:
        print_result("unexpected error", False, str(exc))
        failures.append("unexpected error")

    if failures:
        print(f"\nAPI smoke test failed: {', '.join(failures)}")
        return 1

    print("\nAPI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

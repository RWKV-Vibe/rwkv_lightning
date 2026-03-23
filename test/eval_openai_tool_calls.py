#!/usr/bin/env python3
"""Evaluate real-model tool-call success rate against the OpenAI-compatible API.

Example:
  python test/eval_openai_tool_calls.py \
    --start-server \
    --server-model-path /path/to/model \
    --password rwkv7_7.2b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib import error, request
from urllib.parse import urlparse, urlunparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8000/openai/v1/chat/completions"


def make_tag_tool_prompt(tool_name: str, rule: str) -> str:
    return (
        "You are a tool-using assistant.\n\n"
        "When a tool is needed, you must output exactly one tool call using this exact format and nothing else:\n\n"
        "<tool_call>\n"
        '{"name":"TOOL_NAME","arguments":{"arg1":"value1"}}\n'
        "</tool_call>\n\n"
        "Rules:\n"
        "- Output only one tool call.\n"
        "- Output valid JSON.\n"
        '- "name" must be a string.\n'
        '- "arguments" must be a JSON object.\n'
        "- Do not add markdown fences.\n"
        "- Do not explain.\n"
        "- Do not add any text before or after the tool call.\n"
        f"- If the user request matches this task, always call the tool \"{tool_name}\".\n"
        f"- {rule}\n"
    )


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    system_prompt: str
    user_prompt: str
    expect_tool_call: bool
    expected_tool_name: Optional[str] = None
    expected_arguments_subset: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    index: int
    case_name: str
    expect_tool_call: bool
    ok: bool
    status_code: int
    latency_ms: float
    finish_reason: Optional[str]
    tool_call_present: bool
    tool_name: Optional[str]
    arguments: Optional[dict[str, Any]]
    content: Optional[str]
    failure_reason: str = ""
    raw_error: str = ""


DEFAULT_CASES = [
    CaseDefinition(
        name="weather_beijing",
        system_prompt=make_tag_tool_prompt(
            "get_weather",
            "If the user asks about weather, include location and date in arguments.",
        ),
        user_prompt="What is the weather in Beijing today?",
        expect_tool_call=True,
        expected_tool_name="get_weather",
        expected_arguments_subset={"location": "Beijing", "date": "today"},
    ),
    CaseDefinition(
        name="weather_shanghai",
        system_prompt=make_tag_tool_prompt(
            "get_weather",
            "If the user asks about weather, include location and date in arguments.",
        ),
        user_prompt="Please check the weather in Shanghai tomorrow.",
        expect_tool_call=True,
        expected_tool_name="get_weather",
        expected_arguments_subset={"location": "Shanghai", "date": "tomorrow"},
    ),
    CaseDefinition(
        name="web_search_rwkv",
        system_prompt=make_tag_tool_prompt(
            "web_search",
            "If the user asks to search, include query in arguments.",
        ),
        user_prompt="Search for the latest news about RWKV.",
        expect_tool_call=True,
        expected_tool_name="web_search",
        expected_arguments_subset={"query": "latest news about RWKV"},
    ),
    CaseDefinition(
        name="calculator_basic",
        system_prompt=make_tag_tool_prompt(
            "calculator",
            "If the user asks for a calculation, include expression in arguments.",
        ),
        user_prompt="Calculate 123 * 456.",
        expect_tool_call=True,
        expected_tool_name="calculator",
        expected_arguments_subset={"expression": "123 * 456"},
    ),
    CaseDefinition(
        name="repeat_plain_text",
        system_prompt=(
            "You are a tool-using assistant. If a tool is needed, output exactly one tool call with the <tool_call> JSON format. "
            "If no tool is needed, answer normally in plain text with no tool call."
        ),
        user_prompt="Say exactly: hello world",
        expect_tool_call=False,
    ),
    CaseDefinition(
        name="simple_translation",
        system_prompt=(
            "You are a tool-using assistant. If a tool is needed, output exactly one tool call with the <tool_call> JSON format. "
            "If no tool is needed, answer normally in plain text with no tool call."
        ),
        user_prompt="Translate 'good morning' to Chinese.",
        expect_tool_call=False,
    ),
]


class ManagedServer:
    def __init__(self, process: subprocess.Popen[str], log_path: Path):
        self.process = process
        self.log_path = log_path

    def shutdown(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@dataclass
class SimpleResponse:
    status_code: int
    text: str

    def json(self) -> dict[str, Any]:
        return json.loads(self.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate real-model tool-call success rate against RWKV Lightning OpenAI API."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Target /openai/v1/chat/completions URL")
    parser.add_argument("--model", default="rwkv7", help="Model field to send in requests")
    parser.add_argument("--password", default=None, help="API password; sent in body by default")
    parser.add_argument("--use-bearer", action="store_true", help="Send password via Authorization bearer header")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--repeats", type=int, default=5, help="How many times to run each case")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of loaded cases (0 = all)")
    parser.add_argument("--cases-file", default="", help="Optional JSON file with case definitions")
    parser.add_argument("--json-output", default="", help="Optional path to save structured JSON results")
    parser.add_argument("--tool-call-parser", choices=["auto", "tag", "prefix"], default="tag")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--alpha-presence", type=float, default=0.0)
    parser.add_argument("--alpha-frequency", type=float, default=0.0)
    parser.add_argument("--alpha-decay", type=float, default=0.996)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--enable-think", action="store_true", help="Enable think mode in requests")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case")
    parser.add_argument("--verbose", action="store_true", help="Print each response body")
    parser.add_argument("--start-server", action="store_true", help="Auto-start local app.py before evaluation")
    parser.add_argument("--server-model-path", default="", help="Model path passed to app.py --model-path")
    parser.add_argument("--server-port", type=int, default=8000, help="Port for auto-started local server")
    parser.add_argument("--server-password", default=None, help="Password for auto-started server; defaults to --password")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER, help="Optional custom startup command appended after --server-command")
    parser.add_argument("--startup-timeout", type=float, default=600.0, help="How long to wait for server readiness")
    parser.add_argument("--readiness-url", default="", help="Optional explicit readiness URL")
    parser.add_argument("--readiness-interval", type=float, default=2.0, help="Readiness poll interval in seconds")
    parser.add_argument("--server-log-file", default="", help="Optional file to capture auto-started server logs")
    parser.add_argument("--keep-server", action="store_true", help="Do not stop the auto-started server on exit")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be >= 1")
    if args.max_cases < 0:
        raise SystemExit("--max-cases must be >= 0")
    if args.start_server and not args.server_command and not args.server_model_path:
        raise SystemExit("--start-server requires --server-model-path or --server-command")
    return args


def default_server_command(args: argparse.Namespace) -> list[str]:
    password = args.server_password if args.server_password is not None else args.password
    command = [
        sys.executable,
        str(REPO_ROOT / "app.py"),
        "--model-path",
        args.server_model_path,
        "--port",
        str(args.server_port),
    ]
    if password:
        command.extend(["--password", password])
    return command


def derive_readiness_url(chat_url: str) -> str:
    parsed = urlparse(chat_url)
    return urlunparse((parsed.scheme, parsed.netloc, "/v1/models", "", "", ""))


def ensure_success_status(resp: SimpleResponse) -> None:
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")


def send_json_request(
    url: str,
    payload: Optional[dict[str, Any]],
    headers: dict[str, str],
    timeout_s: float,
    method: str = "POST",
) -> SimpleResponse:
    data: Optional[bytes] = None
    request_headers = dict(headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    req = request.Request(url=url, data=data, headers=request_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return SimpleResponse(status_code=getattr(resp, "status", 200), text=body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return SimpleResponse(status_code=exc.code, text=body)


def start_server(args: argparse.Namespace) -> ManagedServer:
    command = args.server_command if args.server_command else default_server_command(args)
    if not command:
        raise RuntimeError("Server command is empty")

    if args.server_log_file:
        log_path = Path(args.server_log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")
    else:
        temp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="toolcall-eval-server-", suffix=".log", delete=False
        )
        log_path = Path(temp.name)
        log_handle = temp

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    log_handle.flush()
    log_handle.close()
    managed = ManagedServer(process=process, log_path=log_path)
    wait_for_server_ready(
        process=process,
        readiness_url=args.readiness_url or derive_readiness_url(args.url),
        timeout_s=args.startup_timeout,
        interval_s=args.readiness_interval,
        verbose=args.verbose,
        log_path=log_path,
    )
    return managed


def wait_for_server_ready(
    process: subprocess.Popen[str],
    readiness_url: str,
    timeout_s: float,
    interval_s: float,
    verbose: bool,
    log_path: Path,
) -> None:
    start = time.perf_counter()
    last_error = ""
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {process.returncode}. See log: {log_path}"
            )
        try:
            response = send_json_request(
                url=readiness_url,
                payload=None,
                headers={},
                timeout_s=min(interval_s, 5.0) or 5.0,
                method="GET",
            )
            if response.status_code == 200:
                if verbose:
                    print(f"[server] ready: {readiness_url}")
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)

        elapsed = time.perf_counter() - start
        if elapsed >= timeout_s:
            raise RuntimeError(
                f"Timed out waiting for server readiness at {readiness_url}. Last error: {last_error}. See log: {log_path}"
            )
        time.sleep(interval_s)


def load_cases(args: argparse.Namespace) -> list[CaseDefinition]:
    if not args.cases_file:
        cases = list(DEFAULT_CASES)
    else:
        with open(args.cases_file, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
        if not isinstance(raw_cases, list):
            raise SystemExit("cases file must be a JSON list")
        cases = []
        for item in raw_cases:
            if not isinstance(item, dict):
                raise SystemExit("each case must be a JSON object")
            cases.append(
                CaseDefinition(
                    name=str(item["name"]),
                    system_prompt=str(item["system_prompt"]),
                    user_prompt=str(item["user_prompt"]),
                    expect_tool_call=bool(item["expect_tool_call"]),
                    expected_tool_name=(
                        str(item["expected_tool_name"])
                        if item.get("expected_tool_name") is not None
                        else None
                    ),
                    expected_arguments_subset=dict(item.get("expected_arguments_subset") or {}),
                )
            )
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No cases loaded")
    return cases


def build_payload(case: CaseDefinition, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": case.system_prompt},
            {"role": "user", "content": case.user_prompt},
        ],
        "stream": False,
        "enable_tool_calls": True,
        "tool_call_parser": args.tool_call_parser,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha_presence": args.alpha_presence,
        "alpha_frequency": args.alpha_frequency,
        "alpha_decay": args.alpha_decay,
        "chunk_size": args.chunk_size,
        "enable_think": args.enable_think,
    }
    if args.password and not args.use_bearer:
        payload["password"] = args.password
    return payload


def build_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if args.password and args.use_bearer:
        headers["Authorization"] = f"Bearer {args.password}"
    return headers


def parse_tool_call_from_message(message: dict[str, Any]) -> tuple[bool, Optional[str], Optional[dict[str, Any]], Optional[str], str]:
    tool_calls = message.get("tool_calls")
    content = message.get("content")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False, None, None, content, "no_tool_call"

    first = tool_calls[0]
    if not isinstance(first, dict):
        return False, None, None, content, "invalid_tool_call_shape"
    function = first.get("function")
    if not isinstance(function, dict):
        return False, None, None, content, "missing_function_object"
    name = function.get("name")
    arguments_raw = function.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return False, None, None, content, "invalid_tool_name"
    if not isinstance(arguments_raw, str):
        return False, name, None, content, "arguments_not_string"
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return False, name, None, content, "arguments_not_json"
    if not isinstance(arguments, dict):
        return False, name, None, content, "arguments_not_object"
    return True, name.strip(), arguments, content, ""


def arguments_match_subset(actual: dict[str, Any], expected_subset: dict[str, Any]) -> bool:
    for key, expected_value in expected_subset.items():
        if key not in actual:
            return False
        if actual[key] != expected_value:
            return False
    return True


def classify_response(
    case: CaseDefinition,
    status_code: int,
    latency_ms: float,
    data: dict[str, Any],
) -> EvalResult:
    choice = ((data.get("choices") or [{}])[0]) if isinstance(data, dict) else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    tool_ok, tool_name, arguments, content, tool_failure = parse_tool_call_from_message(message)

    result = EvalResult(
        index=-1,
        case_name=case.name,
        expect_tool_call=case.expect_tool_call,
        ok=False,
        status_code=status_code,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        tool_call_present=tool_ok,
        tool_name=tool_name,
        arguments=arguments,
        content=content,
    )

    if case.expect_tool_call:
        if not tool_ok:
            result.failure_reason = tool_failure
            return result
        if case.expected_tool_name and tool_name != case.expected_tool_name:
            result.failure_reason = f"unexpected_tool_name:{tool_name}"
            return result
        expected_subset = case.expected_arguments_subset or {}
        if expected_subset and not arguments_match_subset(arguments or {}, expected_subset):
            result.failure_reason = "arguments_mismatch"
            return result
        result.ok = True
        return result

    if tool_ok:
        result.failure_reason = f"unexpected_tool_call:{tool_name}"
        return result
    text = content if isinstance(content, str) else ""
    if not text.strip():
        result.failure_reason = "missing_plain_text_answer"
        return result
    result.ok = True
    return result


def run_case(case: CaseDefinition, args: argparse.Namespace, index: int) -> EvalResult:
    headers = build_headers(args)
    payload = build_payload(case, args)
    start = time.perf_counter()
    try:
        response = send_json_request(
            url=args.url,
            payload=payload,
            headers=headers,
            timeout_s=args.timeout,
            method="POST",
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        ensure_success_status(response)
        data = response.json()
        result = classify_response(case, response.status_code, latency_ms, data)
        result.index = index
        if args.verbose:
            print(f"\n===== case #{index} {case.name} =====")
            print(json.dumps(data, ensure_ascii=False, indent=2))
        return result
    except Exception as exc:
        return EvalResult(
            index=index,
            case_name=case.name,
            expect_tool_call=case.expect_tool_call,
            ok=False,
            status_code=0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            finish_reason=None,
            tool_call_present=False,
            tool_name=None,
            arguments=None,
            content=None,
            failure_reason="request_failed",
            raw_error=str(exc),
        )


def expand_cases(cases: list[CaseDefinition], repeats: int) -> list[CaseDefinition]:
    expanded = []
    for _ in range(repeats):
        expanded.extend(cases)
    return expanded


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.ok)
    positive = [item for item in results if item.expect_tool_call]
    negative = [item for item in results if not item.expect_tool_call]
    positive_ok = sum(1 for item in positive if item.ok)
    negative_ok = sum(1 for item in negative if item.ok)
    tool_present = sum(1 for item in positive if item.tool_call_present)

    failures: dict[str, int] = {}
    for item in results:
        if item.ok:
            continue
        failures[item.failure_reason or "unknown"] = failures.get(item.failure_reason or "unknown", 0) + 1

    def pct(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else 100.0 * numerator / denominator

    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "overall_success_rate": pct(passed, total),
        "positive_cases": len(positive),
        "positive_successes": positive_ok,
        "positive_success_rate": pct(positive_ok, len(positive)),
        "positive_parse_hits": tool_present,
        "positive_parse_hit_rate": pct(tool_present, len(positive)),
        "negative_cases": len(negative),
        "negative_successes": negative_ok,
        "negative_success_rate": pct(negative_ok, len(negative)),
        "failure_breakdown": failures,
        "mean_latency_ms": (
            sum(item.latency_ms for item in results) / total if total else 0.0
        ),
    }
    return summary


def print_summary(summary: dict[str, Any], results: list[EvalResult]) -> None:
    print("=========== Tool Call Eval Result ===========")
    print(f"Total cases:                {summary['total_cases']}")
    print(f"Passed cases:               {summary['passed_cases']}")
    print(f"Overall success rate:       {summary['overall_success_rate']:.2f}%")
    print(f"Positive parse hit rate:    {summary['positive_parse_hit_rate']:.2f}%")
    print(f"Positive exact success:     {summary['positive_success_rate']:.2f}%")
    print(f"Negative suppression rate:  {summary['negative_success_rate']:.2f}%")
    print(f"Mean latency (ms):          {summary['mean_latency_ms']:.2f}")
    print("--------------- Failures -------------------")
    if not summary["failure_breakdown"]:
        print("None")
    else:
        for reason, count in sorted(summary["failure_breakdown"].items()):
            print(f"{reason}: {count}")
    print("--------------- Per Case -------------------")
    for item in results:
        status = "OK" if item.ok else "FAIL"
        print(
            f"[{status}] #{item.index:03d} {item.case_name} "
            f"latency_ms={item.latency_ms:.2f} status={item.status_code} "
            f"finish={item.finish_reason or '-'} tool={item.tool_name or '-'}"
        )
        if not item.ok:
            detail = item.failure_reason
            if item.raw_error:
                detail = f"{detail} ({item.raw_error})"
            print(f"      {detail}")
    print("============================================")


def maybe_write_json(path_value: str, args: argparse.Namespace, summary: dict[str, Any], results: list[EvalResult]) -> None:
    if not path_value:
        return
    path = Path(path_value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "url": args.url,
            "model": args.model,
            "repeats": args.repeats,
            "tool_call_parser": args.tool_call_parser,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "enable_think": args.enable_think,
        },
        "summary": summary,
        "results": [item.__dict__ for item in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON report to: {path}")


def main() -> None:
    args = parse_args()
    server: Optional[ManagedServer] = None
    try:
        if args.start_server:
            server = start_server(args)
            if args.password is None:
                args.password = (
                    args.server_password if args.server_password is not None else args.password
                )

        cases = load_cases(args)
        expanded_cases = expand_cases(cases, args.repeats)
        results: list[EvalResult] = []
        for index, case in enumerate(expanded_cases):
            result = run_case(case, args, index)
            results.append(result)
            if args.fail_fast and not result.ok:
                break

        summary = summarize(results)
        print_summary(summary, results)
        maybe_write_json(args.json_output, args, summary, results)
        if summary["passed_cases"] != summary["total_cases"]:
            raise SystemExit(1)
    finally:
        if server is not None and not args.keep_server:
            server.shutdown()
            print(f"Stopped managed server. Log: {server.log_path}")
        elif server is not None:
            print(f"Leaving managed server running. Log: {server.log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx


DEFAULT_PROMPTS = [
    "请简要介绍一下中国古代四大发明。",
    "解释一下什么是动态批处理，以及它为什么能提升推理吞吐。",
    "写一段关于春天的短文，不超过120字。",
    "什么是矩阵乘法？给一个简单例子。",
    "介绍一下 Transformer 的基本结构。",
    "用中文解释 TCP 和 UDP 的区别。",
    "帮我写一个 Python 异步 HTTP 请求的示例。",
    "为什么大模型流式输出时首 token 延迟很重要？",
]


@dataclass
class RequestResult:
    index: int
    ok: bool
    status_code: int
    elapsed_ms: float
    ttft_ms: Optional[float]
    output_chars: int
    finish_reason: Optional[str]
    error: str = ""


def build_messages(prompt: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


async def run_one_non_stream(
    client: httpx.AsyncClient,
    url: str,
    index: int,
    prompt: str,
    args: argparse.Namespace,
    started_evt: asyncio.Event,
) -> RequestResult:
    await started_evt.wait()
    payload = {
        "model": args.model,
        "messages": build_messages(prompt, args.system),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha_presence": args.alpha_presence,
        "alpha_frequency": args.alpha_frequency,
        "alpha_decay": args.alpha_decay,
        "stream": False,
        "enable_think": args.enable_think,
        "chunk_size": args.chunk_size,
    }
    if args.password:
        payload["password"] = args.password

    headers = {"Content-Type": "application/json"}
    if args.password and args.use_bearer:
        headers["Authorization"] = f"Bearer {args.password}"

    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        finish_reason = choice.get("finish_reason")
        return RequestResult(
            index=index,
            ok=True,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            ttft_ms=None,
            output_chars=len(text),
            finish_reason=finish_reason,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=getattr(getattr(exc, "response", None), "status_code", 0) or 0,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ttft_ms=None,
            output_chars=0,
            finish_reason=None,
            error=str(exc),
        )


async def run_one_stream(
    client: httpx.AsyncClient,
    url: str,
    index: int,
    prompt: str,
    args: argparse.Namespace,
    started_evt: asyncio.Event,
) -> RequestResult:
    await started_evt.wait()
    payload = {
        "model": args.model,
        "messages": build_messages(prompt, args.system),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha_presence": args.alpha_presence,
        "alpha_frequency": args.alpha_frequency,
        "alpha_decay": args.alpha_decay,
        "stream": True,
        "enable_think": args.enable_think,
        "chunk_size": args.chunk_size,
    }
    if args.password:
        payload["password"] = args.password

    headers = {"Content-Type": "application/json"}
    if args.password and args.use_bearer:
        headers["Authorization"] = f"Bearer {args.password}"

    start = time.perf_counter()
    first_chunk_time = None
    output_parts = []
    finish_reason = None

    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                now = time.perf_counter()
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_chunk_time is None:
                            first_chunk_time = now
                        output_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ttft_ms = None if first_chunk_time is None else (first_chunk_time - start) * 1000.0
        text = "".join(output_parts)
        return RequestResult(
            index=index,
            ok=True,
            status_code=200,
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            output_chars=len(text),
            finish_reason=finish_reason,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=getattr(getattr(exc, "response", None), "status_code", 0) or 0,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ttft_ms=None,
            output_chars=0,
            finish_reason=None,
            error=str(exc),
        )


def summarize(results: list[RequestResult], total_elapsed_s: float) -> None:
    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]

    def mean_or_na(values: list[float]) -> str:
        return "N/A" if not values else f"{statistics.mean(values):.2f}"

    elapsed_values = [r.elapsed_ms for r in ok_results]
    ttft_values = [r.ttft_ms for r in ok_results if r.ttft_ms is not None]
    total_chars = sum(r.output_chars for r in ok_results)
    rps = len(ok_results) / total_elapsed_s if total_elapsed_s > 0 else 0.0

    print("========== Dynamic Batching Test ==========")
    print(f"Total requests:         {len(results)}")
    print(f"Successful requests:    {len(ok_results)}")
    print(f"Failed requests:        {len(fail_results)}")
    print(f"Wall time (s):          {total_elapsed_s:.2f}")
    print(f"Request throughput:     {rps:.2f} req/s")
    print(f"Total output chars:     {total_chars}")
    print(f"Mean latency (ms):      {mean_or_na(elapsed_values)}")
    print(f"Mean TTFT (ms):         {mean_or_na(ttft_values)}")
    print("-------------------------------------------")

    for result in results:
        status = "OK" if result.ok else "ERR"
        ttft = "N/A" if result.ttft_ms is None else f"{result.ttft_ms:.2f}"
        print(
            f"[{status}] #{result.index:03d} "
            f"status={result.status_code} "
            f"latency_ms={result.elapsed_ms:.2f} "
            f"ttft_ms={ttft} "
            f"chars={result.output_chars} "
            f"finish={result.finish_reason or '-'}"
        )
        if result.error:
            print(f"      error={result.error}")


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise SystemExit("prompts file must be a JSON list")
        return [str(x) for x in prompts]

    prompts = list(DEFAULT_PROMPTS)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(prompts)

    if args.requests <= len(prompts):
        return prompts[: args.requests]

    expanded = []
    while len(expanded) < args.requests:
        expanded.extend(prompts)
    return expanded[: args.requests]


async def main_async(args: argparse.Namespace) -> None:
    prompts = load_prompts(args)
    started_evt = asyncio.Event()
    limits = httpx.Limits(max_connections=args.requests * 2, max_keepalive_connections=args.requests * 2)
    timeout = httpx.Timeout(timeout=args.timeout)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = []
        for i, prompt in enumerate(prompts):
            if args.stream:
                tasks.append(asyncio.create_task(run_one_stream(client, args.url, i, prompt, args, started_evt)))
            else:
                tasks.append(asyncio.create_task(run_one_non_stream(client, args.url, i, prompt, args, started_evt)))

        await asyncio.sleep(args.stagger_ms / 1000.0)
        start = time.perf_counter()
        started_evt.set()
        results = await asyncio.gather(*tasks)
        total_elapsed_s = time.perf_counter() - start

    summarize(results, total_elapsed_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async concurrent test for OpenAI-compatible dynamic batching.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/openai/chat/completions")
    parser.add_argument("--model", default="rwkv7")
    parser.add_argument("--password", default="")
    parser.add_argument("--use-bearer", action="store_true", help="Send password via Authorization Bearer header.")
    parser.add_argument("--requests", type=int, default=16, help="Number of concurrent single requests.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode.")
    parser.add_argument("--enable-think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--alpha-presence", type=float, default=0.5)
    parser.add_argument("--alpha-frequency", type=float, default=0.5)
    parser.add_argument("--alpha-decay", type=float, default=0.996)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stagger-ms", type=float, default=50.0, help="Delay before releasing all tasks together.")
    parser.add_argument("--prompts-file", default="", help="Optional JSON file containing a list of prompts.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system",
        default="你是一个简洁、准确的中文助手。请直接回答问题，不要输出多余前言。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
    ## codex resume 019cdcbe-908e-7900-aed7-687dc94fd1e1
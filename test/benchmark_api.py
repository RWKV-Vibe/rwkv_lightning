#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple streaming benchmark client for RWKV Lightning API.

Example:
  python test/benchmark_api.py --url http://localhost:8000/v1/chat/completions
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from infer.rwkv_batch.utils import TRIE_TOKENIZER
except Exception as exc:
    raise SystemExit(f"Failed to import tokenizer: {exc}")


DEFAULT_PROMPTS = [
    "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
    "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:",
]


@dataclass
class ReqStats:
    index: int
    ttft_ms: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    output_tokens: int = 0
    itl_ms: List[float] = field(default_factory=list)
    tpot_ms: List[float] = field(default_factory=list)
    text: List[str] = field(default_factory=list)

    def add_tokens(self, now: float, token_count: int):
        if token_count <= 0:
            return
        if self.first_token_time is None:
            self.first_token_time = now
        if self.last_token_time is not None:
            delta_ms = (now - self.last_token_time) * 1000.0
            # Distribute chunk latency across tokens for a stable per-token estimate.
            per_token_ms = delta_ms / token_count
            self.itl_ms.extend([per_token_ms] * token_count)
            self.tpot_ms.extend([per_token_ms] * token_count)
        self.last_token_time = now
        self.output_tokens += token_count


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    k = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[k]


def apply_template(text: str, template: str) -> str:
    return template.format(prompt=text)


def iter_sharegpt_prompts(path: str, pick: str, template: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        conv = item.get("conversations", [])
        if not conv:
            continue
        humans = [m.get("value", "") for m in conv if m.get("from") == "human"]
        if not humans:
            continue
        if pick == "first":
            yield apply_template(humans[0], template)
        elif pick == "last":
            yield apply_template(humans[-1], template)
        else:
            # random
            yield apply_template(random.choice(humans), template)


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError("prompts file must be a JSON list of strings")
            return prompts

    if args.dataset:
        random.seed(args.seed)
        prompts = list(iter_sharegpt_prompts(args.dataset, args.pick_human, args.template))
        if args.shuffle:
            random.shuffle(prompts)
        if args.num_prompts > 0:
            prompts = prompts[: args.num_prompts]
        return prompts

    return DEFAULT_PROMPTS


def run_benchmark(args: argparse.Namespace) -> None:
    tokenizer = TRIE_TOKENIZER(args.vocab)

    body = {
        "contents": args.prompts,
        "max_tokens": args.max_tokens,
        "stop_tokens": args.stop_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "alpha_presence": args.alpha_presence,
        "alpha_frequency": args.alpha_frequency,
        "alpha_decay": args.alpha_decay,
        "stream": True,
        "password": args.password,
    }

    total_input_tokens = sum(len(tokenizer.encode(p)) for p in args.prompts)

    stats: Dict[int, ReqStats] = {i: ReqStats(i) for i in range(len(args.prompts))}

    start = time.perf_counter()
    first_byte_time: Optional[float] = None

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", args.url, json=body, headers={"Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if first_byte_time is None:
                    first_byte_time = time.perf_counter()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter()
                    for choice in chunk.get("choices", []):
                        idx = choice.get("index")
                        if idx is None or idx not in stats:
                            continue
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue
                        if stats[idx].ttft_ms is None:
                            stats[idx].ttft_ms = (now - start) * 1000.0
                        token_count = len(tokenizer.encode(content))
                        stats[idx].add_tokens(now, token_count)
                        stats[idx].text.append(content)

    end = time.perf_counter()
    duration = end - start

    output_tokens = sum(s.output_tokens for s in stats.values())
    successful = sum(1 for s in stats.values() if s.output_tokens > 0)

    ttft_vals = [s.ttft_ms for s in stats.values() if s.ttft_ms is not None]
    itl_vals = [v for s in stats.values() for v in s.itl_ms]
    tpot_vals = [v for s in stats.values() for v in s.tpot_ms]

    req_throughput = successful / duration if duration > 0 else 0.0
    out_tok_throughput = output_tokens / duration if duration > 0 else 0.0
    total_tok_throughput = (total_input_tokens + output_tokens) / duration if duration > 0 else 0.0

    def fmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.2f}"

    print("============ Serving Benchmark Result ============")
    print(f"Successful requests:                     {successful}")
    print(f"Benchmark duration (s):                  {duration:.2f}")
    print(f"Total input tokens:                      {total_input_tokens}")
    print(f"Total generated tokens:                  {output_tokens}")
    print(f"Request throughput (req/s):              {req_throughput:.2f}")
    print(f"Output token throughput (tok/s):         {out_tok_throughput:.2f}")
    print(f"Total token throughput (tok/s):          {total_tok_throughput:.2f}")
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {fmt(sum(ttft_vals)/len(ttft_vals) if ttft_vals else None)}")
    print(f"Median TTFT (ms):                        {fmt(percentile(ttft_vals, 50) if ttft_vals else None)}")
    print(f"P99 TTFT (ms):                           {fmt(percentile(ttft_vals, 99) if ttft_vals else None)}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {fmt(sum(tpot_vals)/len(tpot_vals) if tpot_vals else None)}")
    print(f"Median TPOT (ms):                        {fmt(percentile(tpot_vals, 50) if tpot_vals else None)}")
    print(f"P99 TPOT (ms):                           {fmt(percentile(tpot_vals, 99) if tpot_vals else None)}")
    print("---------------Inter-token Latency----------------")
    print(f"Mean ITL (ms):                           {fmt(sum(itl_vals)/len(itl_vals) if itl_vals else None)}")
    print(f"Median ITL (ms):                         {fmt(percentile(itl_vals, 50) if itl_vals else None)}")
    print(f"P99 ITL (ms):                            {fmt(percentile(itl_vals, 99) if itl_vals else None)}")
    print("==================================================")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark RWKV Lightning chat completions (streaming).")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions", help="API endpoint")
    p.add_argument("--password", default="rwkv7_7.2b", help="API password")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.6)
    p.add_argument("--alpha-presence", type=float, default=1.0)
    p.add_argument("--alpha-frequency", type=float, default=0.1)
    p.add_argument("--alpha-decay", type=float, default=0.99)
    p.add_argument("--stop-tokens", type=int, nargs="+", default=[0, 261, 24281])
    p.add_argument("--vocab", default="infer/rwkv_batch/rwkv_vocab_v20230424.txt")
    p.add_argument("--prompts-file", default="", help="Optional JSON file with a list of prompts")
    p.add_argument("--dataset", default="", help="ShareGPT JSON dataset file")
    p.add_argument("--num-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle dataset prompts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pick-human", choices=["first", "last", "random"], default="first")
    p.add_argument(
        "--template",
        default="User: {prompt}\\n\\nAssistant: <think>\\n</think>\\n",
        help="Prompt template; use {prompt} placeholder",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.prompts = load_prompts(args)
    if not args.prompts:
        raise SystemExit("No prompts loaded. Check dataset or prompts file.")
    run_benchmark(args)


if __name__ == "__main__":
    main()

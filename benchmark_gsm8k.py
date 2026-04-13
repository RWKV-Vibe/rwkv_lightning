import argparse
import gc
import json
import random
import re
import time
import types
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEFAULT_MODEL = "/mnt/pc411_data/rwkv_translate/rwkv7-g1e-2.9b-20260312-ctx8192"
DEFAULT_DATA = "bak/test_batch_scripts/GSM8k_100sample.jsonl"
DEFAULT_OUT = "gsm8k_bsz32_results.jsonl"


def load_examples(path: Path):
    if path.is_dir():
        from datasets import load_from_disk

        return list(load_from_disk(str(path)))

    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(data.values())

    raise ValueError(f"Unsupported data path: {path}")


def get_problem(example):
    for key in ("question", "problem", "input", "prompt"):
        if key in example:
            return str(example[key])
    raise KeyError(f"Cannot find question/problem field in example keys: {list(example.keys())}")


def get_answer(example):
    for key in ("answer", "expected_answer", "target", "output"):
        if key in example:
            return str(example[key])
    raise KeyError(f"Cannot find answer field in example keys: {list(example.keys())}")


def build_prompt(problem: str) -> str:
    return (
        "User: "
        f"{problem.strip()}\n\n"
        "Assistant: <think"
    )


def extract_answer_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes:
        return hashes[-1].strip()

    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?", text)
    return numbers[-1].strip() if numbers else text.strip()


def normalize_number(text: str):
    text = extract_answer_text(text)
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip()

    frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", text)
    try:
        if frac:
            denom = Decimal(frac.group(2))
            if denom == 0:
                return None
            return Decimal(frac.group(1)) / denom

        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match:
            return Decimal(match.group(0))
    except InvalidOperation:
        return None
    return None


def is_correct(prediction: str, reference: str) -> bool:
    try:
        from math_verify import verify

        if verify(extract_answer_text(prediction), extract_answer_text(reference)):
            return True
    except Exception:
        pass

    pred_num = normalize_number(prediction)
    ref_num = normalize_number(reference)
    if pred_num is None or ref_num is None:
        return extract_answer_text(prediction).strip() == extract_answer_text(reference).strip()
    return abs(pred_num - ref_num) <= Decimal("1e-6")


def xprint(title: str):
    c0, c1 = 3, 80 - len(title) - 3
    print(f"\n{'#' * c0} {title} {'#' * c1}\n")


@torch.no_grad()
def generate_batch(model, tokenizer, sampler, prompts, args, seed_offset: int):
    bsz = len(prompts)
    state = model.generate_zero_state(bsz)
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    out = model.forward_batch(prompt_tokens, state, full_output=False)

    rand_states = sampler.setup_rand(args.seed + seed_offset, bsz)
    generated = [[] for _ in range(bsz)]
    finished = [False for _ in range(bsz)]
    stop_token = 0

    for _ in range(args.max_new_tokens):
        logits = out.float().contiguous()
        next_tokens = sampler.batch_sampling_temperature_topk_topp(
            logits,
            rand_states,
            args.temperature,
            args.top_k,
            args.top_p,
        ).tolist()

        step_tokens = []
        all_finished = True
        for i, token in enumerate(next_tokens):
            if finished[i]:
                step_tokens.append([stop_token])
                continue

            token = int(token)
            generated[i].append(token)
            text = tokenizer.decode(generated[i], utf8_errors="ignore")
            if token == stop_token or "\nUser:" in text or "\n\nUser:" in text:
                finished[i] = True
            else:
                all_finished = False
            step_tokens.append([token])

        if all_finished:
            break

        out = model.forward_batch(step_tokens, state, full_output=False)

    outputs = [tokenizer.decode(tokens, utf8_errors="ignore").strip() for tokens in generated]
    del state, out
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Batch GSM8K evaluation for RWKV7 W8A8.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path without .pth suffix")
    parser.add_argument("--data-path", default=DEFAULT_DATA, help="JSONL/JSON file or datasets.load_from_disk directory")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--bsz", type=int, default=32, help="Batch size")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum generated tokens per problem")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.3, help="Top-p; use 0 for greedy/top-1 in this sampler")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k; <=0 means no top-k limit")
    parser.add_argument("--limit", type=int, default=100, help="Evaluate only the first N examples")
    parser.add_argument("--seed", type=int, default=42, help="Sampler RNG seed")
    args = parser.parse_args()

    examples = load_examples(Path(args.data_path))
    if args.limit > 0:
        examples = examples[: args.limit]

    model_args = types.SimpleNamespace()
    model_args.vocab_size = 65536
    model_args.head_size = 64
    model_args.MODEL_NAME = args.model

    print(f"\nUsing CUDA fp16. Loading {model_args.MODEL_NAME} ...\n")
    from infer.rwkv_batch.rwkv7.modeling_rwkv7 import RWKV_x070
    from infer.rwkv_batch.rwkv7.ops.sampler import sample
    from infer.rwkv_batch.utils import TRIE_TOKENIZER

    model = RWKV_x070(model_args)
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

    xprint("GSM8K")
    print(f"examples={len(examples)} bsz={args.bsz} temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"output={args.output}")

    correct = 0
    total = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(range(0, len(examples), args.bsz), total=(len(examples) + args.bsz - 1) // args.bsz)
        for start in pbar:
            batch = examples[start : start + args.bsz]
            problems = [get_problem(example) for example in batch]
            references = [get_answer(example) for example in batch]
            prompts = [build_prompt(problem) for problem in problems]

            responses = generate_batch(model, tokenizer, sample, prompts, args, start)
            for offset, (example, problem, reference, response) in enumerate(zip(batch, problems, references, responses)):
                prediction = extract_answer_text(response)
                ok = is_correct(prediction, reference)
                correct += int(ok)
                total += 1

                fout.write(json.dumps({
                    "id": start + offset,
                    "problem": problem,
                    "ground_truth": reference,
                    "prediction": prediction,
                    "response": response,
                    "correct": ok,
                }, ensure_ascii=False) + "\n")
                fout.flush()

            pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / max(total, 1):.5f}")

            del responses
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.perf_counter() - t_start
    print(f"\nFinal Accuracy: {correct}/{total} = {correct / max(total, 1):.5f}")
    print(f"Elapsed: {elapsed:.2f}s || examples/s: {total / max(elapsed, 1e-9):.2f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

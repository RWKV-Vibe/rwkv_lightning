import asyncio
import json
import random
import time
from contextlib import suppress
from dataclasses import dataclass, field


DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_PREFILL_AREA = 4096
DEFAULT_PREFILL_TARGET_BATCH_SIZE = 16
PREFILL_TAIL_SPLIT_RATIO = 1.25
PREFILL_TAIL_TILE_RATIO = 0.5
PREFILL_SPEED_TABLE = {
    1: {1: 129.17, 2: 239.94, 4: 406.75, 8: 638.69, 16: 1102.78, 32: 2085.79, 64: 3951.58, 128: 6755.10, 256: 7917.47, 512: 9025.30},
    2: {1: 241.07, 2: 407.43, 4: 636.70, 8: 1106.40, 16: 2087.18, 32: 3971.44, 64: 6886.01, 128: 8084.23, 256: 9429.32, 512: 10347.83},
    4: {1: 406.47, 2: 632.08, 4: 1104.11, 8: 2087.36, 16: 3976.45, 32: 6892.96, 64: 8126.68, 128: 9582.31, 256: 10630.94, 512: 11208.20},
    8: {1: 647.80, 2: 1102.28, 4: 1831.00, 8: 3665.70, 16: 6765.12, 32: 8025.73, 64: 9515.72, 128: 10664.96, 256: 11397.15, 512: 11741.77},
    16: {1: 1059.56, 2: 2102.50, 4: 3941.70, 8: 6635.34, 16: 7954.59, 32: 9531.16, 64: 10698.72, 128: 11364.22, 256: 11743.86, 512: 11563.38},
    32: {1: 2005.99, 2: 3636.61, 4: 6173.57, 8: 7644.24, 16: 9257.79, 32: 10504.88, 64: 11270.07, 128: 11703.44, 256: 11533.98, 512: 10992.51},
    64: {1: 3185.94, 2: 5478.85, 4: 7076.99, 8: 8882.84, 16: 10197.11, 32: 11097.12, 64: 11638.75, 128: 11507.13, 256: 10971.67, 512: 10906.86},
    128: {1: 4414.56, 2: 6109.71, 4: 8096.33, 8: 9699.29, 16: 10743.78, 32: 11436.23, 64: 11414.44, 128: 10946.36, 256: 10905.61, 512: 10955.42},
    256: {1: 4841.88, 2: 6861.50, 4: 8755.31, 8: 10165.14, 16: 11080.96, 32: 11232.95, 64: 10864.33, 128: 10875.29, 256: 10922.62, 512: 10972.19},
    512: {1: 5323.32, 2: 7345.75, 4: 9170.39, 8: 10490.49, 16: 10897.48, 32: 10699.75, 64: 10774.75, 128: 10851.10, 256: 10904.98},
}


@dataclass
class HighThroughputConfig:
    enabled: bool = False
    decode_max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    prefill_area: int = DEFAULT_PREFILL_AREA
    prefill_target_batch_size: int = DEFAULT_PREFILL_TARGET_BATCH_SIZE

    def normalize(self) -> "HighThroughputConfig":
        self.decode_max_batch_size = max(1, int(self.decode_max_batch_size))
        self.prefill_area = max(1, int(self.prefill_area))
        self.prefill_target_batch_size = max(1, int(self.prefill_target_batch_size))
        return self


@dataclass(frozen=True)
class HighThroughputPlan:
    max_active_states: int
    decode_batch_size: int
    prefill_batch_size: int
    prefill_chunk_size: int
    prefill_area: int
    requested_prefill_area: int
    requested_prefill_target_batch_size: int
    tail_split_area: int

    def as_dict(self):
        return {
            "max_active_states": self.max_active_states,
            "decode_batch_size": self.decode_batch_size,
            "prefill_batch_size": self.prefill_batch_size,
            "prefill_chunk_size": self.prefill_chunk_size,
            "prefill_area": self.prefill_area,
            "requested_prefill_area": self.requested_prefill_area,
            "requested_prefill_target_batch_size": self.requested_prefill_target_batch_size,
            "tail_split_area": self.tail_split_area,
        }


@dataclass
class HighThroughputTask:
    index: int
    item_id: int | None
    prompt_tokens: list[int]
    stop_state: dict
    stop_token_ids: set[int]
    text: str = ""
    finish_reason: str = "length"
    completion_tokens: int = 0
    stream_buffer: str = ""
    stream_token_count: int = 0
    started_at: float = field(default_factory=time.time)


class HighThroughputResidentRuntime:
    def __init__(self, engine, config: HighThroughputConfig | None = None):
        self.engine = engine
        self.config = (config or HighThroughputConfig()).normalize()
        self.max_batch_size = self.config.decode_max_batch_size
        self.state = None
        self.logits_pool = None
        self.penalty_pool = None
        self.sample_rand_states = None
        self.rand_state_stride = 0
        self._allocate_resident_pool_with_fallback(self.max_batch_size)
        startup_plan = self.make_plan(total_items=self.max_batch_size)
        print(
            "[HighThroughput] resident pool ready "
            f"max_batch_size={self.max_batch_size} "
            f"prefill_batch_size={startup_plan.prefill_batch_size} "
            f"prefill_chunk_size={startup_plan.prefill_chunk_size} "
            f"prefill_area={startup_plan.prefill_area}"
        )

    def _allocate_resident_pool_with_fallback(self, requested_max_batch_size: int):
        from infer import inference_deps

        torch = inference_deps.get_torch()
        candidate = requested_max_batch_size
        last_error = None
        while candidate >= 1:
            try:
                self._allocate_resident_pool(candidate)
                self.max_batch_size = candidate
                return
            except RuntimeError as exc:
                last_error = exc
                if "out of memory" not in str(exc).lower():
                    raise
                print(
                    "[HighThroughput] resident pool OOM, retrying "
                    f"max_batch_size={max(1, candidate // 2)}"
                )
                self.state = None
                self.logits_pool = None
                self.penalty_pool = None
                self.sample_rand_states = None
                torch.cuda.empty_cache()
                candidate //= 2

        raise last_error

    def _allocate_resident_pool(self, max_batch_size: int):
        from infer import inference_deps

        torch = inference_deps.get_torch()
        vocab_size = int(
            getattr(
                getattr(self.engine.model, "args", None),
                "vocab_size",
                getattr(self.engine.args, "vocab_size", 0),
            )
        )
        if vocab_size <= 0:
            vocab_size = int(self.engine.model.z["head.weight"].shape[0])

        self.state = self.engine.model.generate_zero_state(max_batch_size)
        self.logits_pool = torch.empty(
            (max_batch_size, vocab_size),
            dtype=torch.float32,
            device="cuda",
        )
        self.penalty_pool = torch.zeros(
            (max_batch_size, vocab_size),
            dtype=torch.float32,
            device="cuda",
        )
        self.sample_rand_states = inference_deps.get_sample().setup_rand(
            random.randint(0, 2**63 - 1),
            max_batch_size,
        )
        self.rand_state_stride = self.sample_rand_states.numel() // max_batch_size

    def make_plan(
        self,
        *,
        total_items: int,
        decode_max_batch_size: int | None = None,
        prefill_area: int | None = None,
        prefill_target_batch_size: int | None = None,
    ) -> HighThroughputPlan:
        requested_decode_max = (
            self.config.decode_max_batch_size
            if decode_max_batch_size is None
            else int(decode_max_batch_size)
        )
        decode_batch_size = max(
            1,
            min(total_items, self.max_batch_size, max(1, requested_decode_max)),
        )

        target_area = (
            self.config.prefill_area
            if prefill_area is None
            else max(1, int(prefill_area))
        )
        target_batch_size = (
            self.config.prefill_target_batch_size
            if prefill_target_batch_size is None
            else max(1, int(prefill_target_batch_size))
        )

        max_prefill_batch_size = max(
            1,
            min(total_items, decode_batch_size),
        )
        prefill_batch_size = choose_power_of_two_prefill_batch_size(
            target_area=target_area,
            target_batch_size=target_batch_size,
            max_batch_size=max_prefill_batch_size,
        )
        prefill_chunk_size = max(1, round(target_area / prefill_batch_size))

        return HighThroughputPlan(
            max_active_states=self.max_batch_size,
            decode_batch_size=decode_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_chunk_size=prefill_chunk_size,
            prefill_area=prefill_batch_size * prefill_chunk_size,
            requested_prefill_area=target_area,
            requested_prefill_target_batch_size=target_batch_size,
            tail_split_area=max(1, int(target_area * PREFILL_TAIL_TILE_RATIO)),
        )

    def state_head_view(self, batch_size: int):
        return [
            self.state[0][:, :, :batch_size, :],
            self.state[1][:, :batch_size],
            self.state[2][:batch_size],
        ]

    def state_range_view(self, start: int, end: int):
        return [
            self.state[0][:, :, start:end, :],
            self.state[1][:, start:end],
            self.state[2][start:end],
        ]

    def reset_active(self, batch_size: int):
        self.state[0][:, :, :batch_size, :].zero_()
        self.state[1][:, :batch_size].zero_()
        self.state[2][:batch_size].zero_()
        self.penalty_pool[:batch_size].zero_()

    def prefill_into_logits_pool(
        self,
        tokens: list[list[int]],
        plan: HighThroughputPlan,
        cancel_token=None,
    ):
        engine = self.engine
        lengths = [len(item) for item in tokens]
        positions = [0] * len(tokens)

        while True:
            engine._raise_if_cancelled(cancel_token)
            tile_indices, step = choose_prefill_tile(lengths, positions, plan)
            if not tile_indices:
                break

            batch_tokens = [
                tokens[i][positions[i] : positions[i] + step]
                for i in tile_indices
            ]
            batch_state = [
                self.state[0][:, :, tile_indices],
                self.state[1][:, tile_indices],
                self.state[2][tile_indices],
            ]

            new_logits = engine.model.forward_batch_same_length(batch_tokens, batch_state)
            if new_logits.dim() == 3:
                new_logits = new_logits[:, -1]

            for active_index, original_index in enumerate(tile_indices):
                self.logits_pool[original_index].copy_(new_logits[active_index])
                positions[original_index] += step
                self.state[0][:, :, original_index].copy_(
                    batch_state[0][:, :, active_index]
                )
                self.state[1][:, original_index].copy_(
                    batch_state[1][:, active_index]
                )
                self.state[2][original_index].copy_(batch_state[2][active_index])


def powers_of_two_up_to(max_value: int) -> list[int]:
    values = []
    value = 1
    while value <= max_value:
        values.append(value)
        value *= 2
    return values or [1]


def prefill_speed(batch_size: int, chunk_size: int) -> float:
    return PREFILL_SPEED_TABLE.get(batch_size, {}).get(chunk_size, 0.0)


def batch_distance(batch_size: int, target_batch_size: int):
    target_batch_size = max(1, target_batch_size)
    return (
        abs(batch_size - target_batch_size),
        -batch_size,
    )


def choose_power_of_two_prefill_batch_size(
    *,
    target_area: int,
    target_batch_size: int,
    max_batch_size: int,
) -> int:
    candidates = powers_of_two_up_to(max(1, max_batch_size))
    exact_area_candidates = [
        batch_size
        for batch_size in candidates
        if target_area % batch_size == 0 and target_area // batch_size >= 1
    ]
    if exact_area_candidates:
        return min(
            exact_area_candidates,
            key=lambda batch_size: batch_distance(batch_size, target_batch_size),
        )

    return min(
        candidates,
        key=lambda batch_size: batch_distance(batch_size, target_batch_size),
    )


def choose_prefill_tile(
    lengths: list[int],
    positions: list[int],
    plan: HighThroughputPlan,
) -> tuple[list[int], int]:
    remaining = [
        (i, lengths[i] - positions[i])
        for i in range(len(lengths))
        if positions[i] < lengths[i]
    ]
    if not remaining:
        return [], 0

    remaining.sort(key=lambda item: item[1], reverse=True)
    total_area = sum(rem for _, rem in remaining)
    target_area = plan.requested_prefill_area
    max_tile_area = target_area
    if target_area < total_area <= int(target_area * PREFILL_TAIL_SPLIT_RATIO):
        max_tile_area = plan.tail_split_area

    candidates = []
    for batch_size in powers_of_two_up_to(min(len(remaining), plan.decode_batch_size)):
        selected = remaining[:batch_size]
        max_chunk_size = min(rem for _, rem in selected)
        for chunk_size in powers_of_two_up_to(max_chunk_size):
            area = batch_size * chunk_size
            if area > max_tile_area:
                continue
            speed = prefill_speed(batch_size, chunk_size)
            if speed <= 0:
                continue
            candidates.append((speed, area, batch_size, chunk_size, selected))

    if not candidates:
        original_index, _ = remaining[0]
        return [original_index], 1

    _, _, batch_size, chunk_size, selected = max(
        candidates,
        key=lambda item: (
            item[1],
            -batch_distance(item[2], plan.requested_prefill_target_batch_size)[0],
            item[2],
            item[0],
            -abs(item[2] - plan.requested_prefill_target_batch_size),
        ),
    )
    return [original_index for original_index, _ in selected[:batch_size]], chunk_size


def encode_high_throughput_prompts(engine, prompts: list[str], cancel_token=None):
    encoded_prompts = []
    for prompt in prompts:
        engine._raise_if_cancelled(cancel_token)
        encoded_prompts.append(engine.tokenizer.encode(prompt) or [0])
    return encoded_prompts


def split_stop_tokens(stop_tokens):
    stop_strings = []
    stop_token_ids = set()
    for stop_token in stop_tokens or []:
        if isinstance(stop_token, int):
            stop_token_ids.add(stop_token)
        elif isinstance(stop_token, str) and stop_token:
            stop_strings.append(stop_token)
    return stop_strings, stop_token_ids


def swap_state_rows(state, left: int, right: int):
    tmp0 = state[0][:, :, left : left + 1, :].clone()
    state[0][:, :, left : left + 1, :].copy_(state[0][:, :, right : right + 1, :])
    state[0][:, :, right : right + 1, :].copy_(tmp0)

    tmp1 = state[1][:, left : left + 1].clone()
    state[1][:, left : left + 1].copy_(state[1][:, right : right + 1])
    state[1][:, right : right + 1].copy_(tmp1)

    tmp2 = state[2][left : left + 1].clone()
    state[2][left : left + 1].copy_(state[2][right : right + 1])
    state[2][right : right + 1].copy_(tmp2)


def swap_tensor_rows(tensor, left: int, right: int):
    tmp = tensor[left : left + 1].clone()
    tensor[left : left + 1].copy_(tensor[right : right + 1])
    tensor[right : right + 1].copy_(tmp)


def swap_rand_state_rows(sample_rand_states, row_stride: int, left: int, right: int):
    if row_stride <= 0:
        return
    left_start = left * row_stride
    right_start = right * row_stride
    tmp = sample_rand_states[left_start : left_start + row_stride].clone()
    sample_rand_states[left_start : left_start + row_stride].copy_(
        sample_rand_states[right_start : right_start + row_stride]
    )
    sample_rand_states[right_start : right_start + row_stride].copy_(tmp)


def finish_task(engine, task: HighThroughputTask, results: list, usages: list):
    flushed = engine._flush_stop_state(task.stop_state, final=True)
    if flushed:
        task.text += flushed
    choice = {
        "index": task.index,
        "message": {"role": "assistant", "content": task.text},
        "finish_reason": task.finish_reason,
    }
    if task.item_id is not None:
        choice["item_id"] = task.item_id
    results[task.index] = choice

    usages[task.index] = {
        "prompt_tokens": len(task.prompt_tokens),
        "completion_tokens": task.completion_tokens,
        "cost_time": round(time.time() - task.started_at, 3),
    }
    if task.item_id is not None:
        usages[task.index]["item_id"] = task.item_id
    return flushed


def emit_stream_contents(stream_callback, tasks: list[HighThroughputTask]):
    if stream_callback is None:
        return

    choices = []
    for task in sorted(tasks, key=lambda item: item.index):
        if not task.stream_buffer:
            continue
        choices.append(
            {
                "index": task.index,
                "delta": {"content": task.stream_buffer},
            }
        )
        task.stream_buffer = ""
        task.stream_token_count = 0

    if not choices:
        return

    chunk = {
        "object": "chat.completion.chunk",
        "choices": choices,
    }
    stream_callback(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")


def run_high_throughput_chunk(
    runtime: HighThroughputResidentRuntime,
    prompt_tokens: list[list[int]],
    *,
    plan: HighThroughputPlan,
    item_ids: list[int | None],
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    stop_tokens,
    result_offset: int,
    results: list,
    usages: list,
    cancel_token=None,
    stream_callback=None,
    stream_chunk_size: int = 1,
):
    engine = runtime.engine
    batch_size = len(prompt_tokens)
    stream_chunk_size = max(1, int(stream_chunk_size))
    stop_strings, stop_token_ids = split_stop_tokens(stop_tokens)
    tasks = [
        HighThroughputTask(
            index=result_offset + i,
            item_id=item_ids[i],
            prompt_tokens=tokens,
            stop_state=engine._create_stop_state(stop_strings),
            stop_token_ids=stop_token_ids,
        )
        for i, tokens in enumerate(prompt_tokens)
    ]

    if max_length <= 0:
        for task in tasks:
            flushed = finish_task(engine, task, results, usages)
            if stream_callback is not None and flushed:
                task.stream_buffer += flushed
        emit_stream_contents(stream_callback, tasks)
        return

    runtime.reset_active(batch_size)
    runtime.prefill_into_logits_pool(prompt_tokens, plan, cancel_token=cancel_token)

    active_size = batch_size
    temperature = max(0.001, float(temperature))

    while active_size > 0:
        from infer import inference_deps

        engine._raise_if_cancelled(cancel_token)
        next_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
            runtime.logits_pool[:active_size],
            runtime.penalty_pool[:active_size],
            runtime.sample_rand_states,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            temperature,
            top_k,
            top_p,
        ).tolist()

        i = 0
        stream_ready_tasks = []
        while i < active_size:
            task = tasks[i]
            token = next_tokens[i]
            if token in task.stop_token_ids:
                content = ""
                should_stop = True
            else:
                content, should_stop = engine._ingest_token_with_stop(task.stop_state, token)
                if content:
                    task.text += content
                    if stream_callback is not None:
                        task.stream_buffer += content

            if should_stop:
                task.finish_reason = "stop"
                stopped = True
            else:
                task.completion_tokens += 1
                if stream_callback is not None:
                    task.stream_token_count += 1
                stopped = task.completion_tokens >= max_length
                if stopped:
                    task.finish_reason = "length"

            if stopped:
                flushed = finish_task(engine, task, results, usages)
                if flushed:
                    task.stream_buffer += flushed
                stream_ready_tasks.append(task)
                last = active_size - 1
                if i != last:
                    tasks[i], tasks[last] = tasks[last], tasks[i]
                    next_tokens[i], next_tokens[last] = next_tokens[last], next_tokens[i]
                    swap_state_rows(runtime.state, i, last)
                    swap_tensor_rows(runtime.penalty_pool, i, last)
                    swap_rand_state_rows(
                        runtime.sample_rand_states,
                        runtime.rand_state_stride,
                        i,
                        last,
                    )
                active_size -= 1
                continue

            if task.stream_buffer and task.stream_token_count >= stream_chunk_size:
                stream_ready_tasks.append(task)

            i += 1

        emit_stream_contents(stream_callback, stream_ready_tasks)

        if active_size > 0:
            active_state = runtime.state_head_view(active_size)
            new_logits = engine.model.forward_batch_same_length(
                [[token] for token in next_tokens[:active_size]],
                active_state,
            )
            if new_logits.dim() == 3:
                new_logits = new_logits[:, -1]
            runtime.logits_pool[:active_size].copy_(new_logits)


def run_high_throughput_generate(
    runtime: HighThroughputResidentRuntime,
    *,
    plan: HighThroughputPlan,
    prompts: list[str],
    encoded_prompts: list[list[int]] | None = None,
    item_ids: list[int | None] | None = None,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    stop_tokens,
    cancel_token=None,
):
    engine = runtime.engine
    if encoded_prompts is None:
        encoded_prompts = encode_high_throughput_prompts(
            engine,
            prompts,
            cancel_token=cancel_token,
        )
    if item_ids is None:
        item_ids = [None] * len(encoded_prompts)
    results = [None] * len(encoded_prompts)
    usages = [None] * len(encoded_prompts)

    with engine.model_lock:
        for start in range(0, len(encoded_prompts), plan.decode_batch_size):
            engine._raise_if_cancelled(cancel_token)
            chunk_tokens = encoded_prompts[start : start + plan.decode_batch_size]
            chunk_item_ids = item_ids[start : start + plan.decode_batch_size]
            run_high_throughput_chunk(
                runtime,
                chunk_tokens,
                plan=plan,
                item_ids=chunk_item_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_presence=alpha_presence,
                alpha_frequency=alpha_frequency,
                alpha_decay=alpha_decay,
                stop_tokens=stop_tokens,
                result_offset=start,
                results=results,
                usages=usages,
                cancel_token=cancel_token,
            )

    return results, usages


async def run_high_throughput_stream(
    runtime: HighThroughputResidentRuntime,
    *,
    plan: HighThroughputPlan,
    prompts: list[str],
    encoded_prompts: list[list[int]] | None = None,
    item_ids: list[int | None] | None = None,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    stop_tokens,
    chunk_size: int = 1,
    cancel_token=None,
):
    engine = runtime.engine
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    done_marker = object()

    def enqueue(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker():
        try:
            worker_encoded_prompts = encoded_prompts
            if worker_encoded_prompts is None:
                worker_encoded_prompts = encode_high_throughput_prompts(
                    engine,
                    prompts,
                    cancel_token=cancel_token,
                )
            window_item_ids = item_ids
            if window_item_ids is None:
                window_item_ids = [None] * len(worker_encoded_prompts)
            results = [None] * len(worker_encoded_prompts)
            usages = [None] * len(worker_encoded_prompts)

            with engine.model_lock:
                for start in range(0, len(worker_encoded_prompts), plan.decode_batch_size):
                    engine._raise_if_cancelled(cancel_token)
                    chunk_tokens = worker_encoded_prompts[
                        start : start + plan.decode_batch_size
                    ]
                    chunk_item_ids = window_item_ids[
                        start : start + plan.decode_batch_size
                    ]
                    run_high_throughput_chunk(
                        runtime,
                        chunk_tokens,
                        plan=plan,
                        item_ids=chunk_item_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        alpha_presence=alpha_presence,
                        alpha_frequency=alpha_frequency,
                        alpha_decay=alpha_decay,
                        stop_tokens=stop_tokens,
                        result_offset=start,
                        results=results,
                        usages=usages,
                        cancel_token=cancel_token,
                        stream_callback=enqueue,
                        stream_chunk_size=chunk_size,
                    )
            enqueue("data: [DONE]\n\n")
        except Exception as exc:
            enqueue(exc)
        finally:
            enqueue(done_marker)

    future = loop.run_in_executor(engine.executor, worker)
    try:
        while True:
            item = await queue.get()
            if item is done_marker:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if cancel_token is not None:
            cancel_token.cancel()
        if not future.done():
            with suppress(asyncio.CancelledError):
                await asyncio.wrap_future(future)

import asyncio
import gc
import json
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event, Lock, Thread

import torch

from infer.rwkv_batch.sampler import sample
from infer.rwkv_batch.utils import sampler_gumbel_batch


class InferenceEngine:
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="model_inference"
        )
        self.dynamic_batch_lock = Lock()
        self.dynamic_batch_stop_event = Event()
        self.dynamic_batch_schedulers = {}
        self.dynamic_batch_max_size = max(
            1, int(getattr(args, "dynamic_batch_max_size", 32))
        )
        self.dynamic_batch_wait_ms = max(
            0, int(getattr(args, "dynamic_batch_wait_ms", 10))
        )

    def shutdown(self):
        self.dynamic_batch_stop_event.set()
        with self.dynamic_batch_lock:
            schedulers = list(self.dynamic_batch_schedulers.values())
        for scheduler in schedulers:
            thread = scheduler.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=0.2)
        self.executor.shutdown(wait=False)

    def _dynamic_batch_key(
        self, temperature, top_k, top_p, alpha_presence, alpha_frequency, alpha_decay
    ):
        return (
            float(temperature),
            int(top_k),
            float(top_p),
            float(alpha_presence),
            float(alpha_frequency),
            float(alpha_decay),
        )

    def _init_cuda_graph_state(self, token, state, out):
        x_emb = self.model.z["emb.weight"][token]

        static_input = torch.empty_like(x_emb, device="cuda")
        static_state = [None, None, None]
        static_state[0] = torch.empty_like(state[0], device="cuda")
        static_state[1] = torch.empty_like(state[1], device="cuda")
        static_state[2] = torch.empty_like(state[2], device="cuda")
        static_output = torch.empty_like(out, device="cuda")

        static_output = self.model.forward(static_input, static_state)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.model.forward(static_input, static_state)

        static_input.copy_(x_emb)
        static_state[0].copy_(state[0])
        static_state[1].copy_(state[1])
        static_state[2].copy_(state[2])
        static_output.copy_(out)

        return static_input, static_state, static_output, g

    def _sample_next_token(
        self,
        static_output,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        temperature,
        top_k,
        top_p,
    ):
        logits_reshaped = static_output.unsqueeze(0).float()

        sample_rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), 1)
        penalties = torch.zeros(1, 65536).to(0)
        new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(
            logits_reshaped,
            penalties,
            sample_rand_states,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            temperature,
            top_k,
            top_p,
        ).tolist()
        return new_tokens[0]

    @staticmethod
    def _cleanup_cuda_state(state):
        del state
        gc.collect()
        torch.cuda.empty_cache()

    def _get_dynamic_scheduler(self, key):
        with self.dynamic_batch_lock:
            scheduler = self.dynamic_batch_schedulers.get(key)
            if scheduler is None:
                scheduler = {"queue": Queue(), "thread": None}
                self.dynamic_batch_schedulers[key] = scheduler

            thread = scheduler["thread"]
            if thread is None or not thread.is_alive():
                thread = Thread(
                    target=self._dynamic_batch_worker,
                    args=(key, scheduler["queue"]),
                    name=f"dynamic_batch_{len(self.dynamic_batch_schedulers)}",
                    daemon=True,
                )
                scheduler["thread"] = thread
                thread.start()

            return scheduler

    def _submit_dynamic_batch_request(
        self,
        prompt,
        max_generate_tokens,
        stop_tokens,
        pad_zero,
        temperature,
        top_k,
        top_p,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        chunk_size,
        stream,
    ):
        key = self._dynamic_batch_key(
            temperature, top_k, top_p, alpha_presence, alpha_frequency, alpha_decay
        )
        scheduler = self._get_dynamic_scheduler(key)
        request_queue = Queue()
        request_data = {
            "prompt": prompt,
            "max_generate_tokens": max(0, int(max_generate_tokens)),
            "stop_tokens": set(stop_tokens),
            "pad_zero": bool(pad_zero),
            "chunk_size": max(1, int(chunk_size)),
            "stream": bool(stream),
            "queue": request_queue,
        }
        scheduler["queue"].put(request_data)
        return request_queue

    def _append_state_slot(self, states, device):
        extra_state = self.model.generate_zero_state(1)
        if states is None:
            new_states = [extra_state[0], extra_state[1], extra_state[2]]
        else:
            new_states = [
                torch.cat([states[0], extra_state[0]], dim=2),
                torch.cat([states[1], extra_state[1]], dim=1),
                torch.cat([states[2], extra_state[2]], dim=0),
            ]
        del extra_state
        return new_states

    def _drain_dynamic_requests(self, request_queue, pending_requests, limit):
        while len(pending_requests) < limit:
            try:
                pending_requests.append(request_queue.get_nowait())
            except Empty:
                break

    def _add_dynamic_tasks(
        self,
        task_pool,
        pending_requests,
        states,
        occurrence,
        alpha_presence_vector,
        device,
    ):
        while pending_requests and len(task_pool) < self.dynamic_batch_max_size:
            request_data = pending_requests.popleft()
            states = self._append_state_slot(states, device)

            zeros = torch.zeros(
                (1, self.args.vocab_size), dtype=torch.float32, device=device
            )
            if occurrence is None:
                occurrence = zeros.clone()
                alpha_presence_vector = zeros
            else:
                occurrence = torch.cat([occurrence, zeros], dim=0)
                alpha_presence_vector = torch.cat(
                    [
                        alpha_presence_vector,
                        torch.zeros(
                            (1, self.args.vocab_size),
                            dtype=torch.float32,
                            device=device,
                        ),
                    ],
                    dim=0,
                )

            input_tokens = self.tokenizer.encode(request_data["prompt"])
            if request_data["pad_zero"]:
                input_tokens = [0] + input_tokens
            if not input_tokens:
                input_tokens = [0]

            task_pool.append(
                {
                    "request": request_data,
                    "input_token": input_tokens,
                    "state_pos": len(task_pool),
                    "generated_tokens": [],
                    "token_buffer": [],
                    "new_token": None,
                }
            )

        return states, occurrence, alpha_presence_vector

    @staticmethod
    def _queue_dynamic_delta(task, text):
        if text and task["request"]["stream"]:
            task["request"]["queue"].put({"type": "delta", "text": text})

    def _finish_dynamic_task(self, task, finish_reason):
        if task["token_buffer"]:
            text_chunk = self.tokenizer.decode(
                task["token_buffer"], utf8_errors="ignore"
            )
            task["token_buffer"].clear()
            self._queue_dynamic_delta(task, text_chunk)

        full_text = self.tokenizer.decode(
            task["generated_tokens"], utf8_errors="ignore"
        )
        task["request"]["queue"].put(
            {"type": "done", "text": full_text, "finish_reason": finish_reason}
        )

    def _dynamic_batch_worker(self, key, request_queue):
        temperature, top_k, top_p, alpha_presence, alpha_frequency, alpha_decay = key
        device = self.model.z["head.weight"].device
        alpha_presence_val = torch.tensor(
            alpha_presence, dtype=torch.float32, device=device
        )
        no_penalty_token_ids = {33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58}

        if temperature == 0:
            temperature = 1.0
            top_k = 1

        while not self.dynamic_batch_stop_event.is_set():
            try:
                first_request = request_queue.get(timeout=0.1)
            except Empty:
                continue

            pending_requests = deque([first_request])
            collect_deadline = time.monotonic() + self.dynamic_batch_wait_ms / 1000.0
            while len(pending_requests) < self.dynamic_batch_max_size:
                remaining = collect_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pending_requests.append(request_queue.get(timeout=remaining))
                except Empty:
                    break

            task_pool = []
            states = None
            occurrence = None
            alpha_presence_vector = None

            try:
                with self.model_lock:
                    states, occurrence, alpha_presence_vector = self._add_dynamic_tasks(
                        task_pool,
                        pending_requests,
                        states,
                        occurrence,
                        alpha_presence_vector,
                        device,
                    )

                    while task_pool or pending_requests:
                        self._drain_dynamic_requests(
                            request_queue,
                            pending_requests,
                            self.dynamic_batch_max_size - len(task_pool),
                        )
                        if len(task_pool) < self.dynamic_batch_max_size:
                            states, occurrence, alpha_presence_vector = (
                                self._add_dynamic_tasks(
                                    task_pool,
                                    pending_requests,
                                    states,
                                    occurrence,
                                    alpha_presence_vector,
                                    device,
                                )
                            )

                        accomplished_task_indices = []
                        state_slots_to_remove = set()

                        for task_idx, task in enumerate(task_pool):
                            if len(task["input_token"]) != 0:
                                continue
                            if task["new_token"] is None:
                                continue

                            new_token = task["new_token"]
                            request_data = task["request"]
                            token_in_stop = new_token in request_data["stop_tokens"]
                            length_exceed = (
                                len(task["generated_tokens"])
                                >= request_data["max_generate_tokens"]
                            )

                            if not token_in_stop and not length_exceed:
                                task["generated_tokens"].append(new_token)
                                task["token_buffer"].append(new_token)

                                if (
                                    len(task["token_buffer"])
                                    >= request_data["chunk_size"]
                                ):
                                    text_chunk = self.tokenizer.decode(
                                        task["token_buffer"], utf8_errors="ignore"
                                    )
                                    task["token_buffer"].clear()
                                    self._queue_dynamic_delta(task, text_chunk)

                            if token_in_stop or length_exceed:
                                finish_reason = "stop" if token_in_stop else "length"
                                self._finish_dynamic_task(task, finish_reason)

                                if pending_requests:
                                    request_data = pending_requests.popleft()
                                    input_tokens = self.tokenizer.encode(
                                        request_data["prompt"]
                                    )
                                    if request_data["pad_zero"]:
                                        input_tokens = [0] + input_tokens
                                    if not input_tokens:
                                        input_tokens = [0]

                                    state_pos = task["state_pos"]
                                    task_pool[task_idx] = {
                                        "request": request_data,
                                        "input_token": input_tokens,
                                        "state_pos": state_pos,
                                        "generated_tokens": [],
                                        "token_buffer": [],
                                        "new_token": None,
                                    }
                                    states[0][:, :, state_pos, :] = 0
                                    states[1][:, state_pos, :, :] = 0
                                    states[2][state_pos] = 0
                                    occurrence[state_pos, :] = 0
                                    alpha_presence_vector[state_pos, :] = 0
                                else:
                                    accomplished_task_indices.append(task_idx)
                                    state_slots_to_remove.add(task["state_pos"])
                            else:
                                task["input_token"].append(new_token)
                                penalty_scale = (
                                    0.0 if new_token in no_penalty_token_ids else 1.0
                                )
                                occurrence[task["state_pos"], new_token] += (
                                    penalty_scale
                                )
                                alpha_presence_vector[task["state_pos"], new_token] = (
                                    alpha_presence_val
                                )

                        if accomplished_task_indices:
                            sorted_slots = sorted(state_slots_to_remove, reverse=True)
                            for slot in sorted_slots:
                                states[0] = torch.cat(
                                    [
                                        states[0][:, :, :slot, :],
                                        states[0][:, :, slot + 1 :, :],
                                    ],
                                    dim=2,
                                )
                                states[1] = torch.cat(
                                    [
                                        states[1][:, :slot, :, :],
                                        states[1][:, slot + 1 :, :, :],
                                    ],
                                    dim=1,
                                )
                                states[2] = torch.cat(
                                    [states[2][:slot], states[2][slot + 1 :]], dim=0
                                )
                                occurrence = torch.cat(
                                    [occurrence[:slot, :], occurrence[slot + 1 :, :]],
                                    dim=0,
                                )
                                alpha_presence_vector = torch.cat(
                                    [
                                        alpha_presence_vector[:slot, :],
                                        alpha_presence_vector[slot + 1 :, :],
                                    ],
                                    dim=0,
                                )

                            for task_idx in sorted(
                                accomplished_task_indices, reverse=True
                            ):
                                del task_pool[task_idx]

                            remaining_slots = sorted(
                                task["state_pos"] for task in task_pool
                            )
                            pos_map = {
                                old_pos: new_pos
                                for new_pos, old_pos in enumerate(remaining_slots)
                            }
                            for task in task_pool:
                                task["state_pos"] = pos_map[task["state_pos"]]

                        if not task_pool:
                            if pending_requests:
                                states, occurrence, alpha_presence_vector = (
                                    self._add_dynamic_tasks(
                                        task_pool,
                                        pending_requests,
                                        states,
                                        occurrence,
                                        alpha_presence_vector,
                                        device,
                                    )
                                )
                                continue
                            break

                        next_tokens = [None] * len(task_pool)
                        for task in task_pool:
                            next_tokens[task["state_pos"]] = [
                                task["input_token"].pop(0)
                            ]

                        out = self.model.forward_batch(next_tokens, states)

                        if alpha_presence != 0 or alpha_frequency != 0:
                            mask = (occurrence > 0).float()
                            out -= mask * alpha_presence + occurrence * alpha_frequency

                        occurrence *= alpha_decay
                        out -= alpha_presence_vector + occurrence * alpha_frequency

                        if temperature != 1.0:
                            out /= temperature

                        if self.rocm_flag:
                            new_tokens = self._torch_top_k_top_p(out, top_k, top_p)
                        else:
                            try:
                                import flashinfer  # type: ignore

                                new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                    out, top_k, top_p
                                )
                            except Exception:
                                new_tokens = self._torch_top_k_top_p(out, top_k, top_p)

                        new_tokens = new_tokens.tolist()
                        for task in task_pool:
                            task["new_token"] = new_tokens[task["state_pos"]]
            except Exception as exc:
                error_message = str(exc)
                for task in task_pool:
                    task["request"]["queue"].put(
                        {"type": "error", "error": error_message}
                    )
                while pending_requests:
                    request_data = pending_requests.popleft()
                    request_data["queue"].put({"type": "error", "error": error_message})
            finally:
                if states is not None:
                    del states
                if occurrence is not None:
                    del occurrence
                if alpha_presence_vector is not None:
                    del alpha_presence_vector
                gc.collect()
                torch.cuda.empty_cache()

    async def dynamic_batch_generate(
        self,
        prompt,
        max_generate_tokens,
        stop_tokens,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
    ):
        request_queue = self._submit_dynamic_batch_request(
            prompt=prompt,
            max_generate_tokens=max_generate_tokens,
            stop_tokens=stop_tokens,
            pad_zero=pad_zero,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            alpha_presence=alpha_presence,
            alpha_frequency=alpha_frequency,
            alpha_decay=alpha_decay,
            chunk_size=chunk_size,
            stream=False,
        )

        while True:
            await asyncio.sleep(0.01)
            try:
                item = request_queue.get_nowait()
            except Empty:
                continue

            if item["type"] == "done":
                return item["text"], item["finish_reason"]
            if item["type"] == "error":
                raise RuntimeError(item["error"])

    async def dynamic_batch_infer_stream(
        self,
        prompt,
        max_generate_tokens,
        stop_tokens,
        pad_zero=True,
        temperature=1.0,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
    ):
        request_queue = self._submit_dynamic_batch_request(
            prompt=prompt,
            max_generate_tokens=max_generate_tokens,
            stop_tokens=stop_tokens,
            pad_zero=pad_zero,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            alpha_presence=alpha_presence,
            alpha_frequency=alpha_frequency,
            alpha_decay=alpha_decay,
            chunk_size=chunk_size,
            stream=True,
        )

        while True:
            await asyncio.sleep(0.01)
            while True:
                try:
                    item = request_queue.get_nowait()
                except Empty:
                    break

                if item["type"] == "delta":
                    yield item
                    continue
                if item["type"] == "done":
                    yield item
                    return
                if item["type"] == "error":
                    raise RuntimeError(item["error"])

    @staticmethod
    def _torch_top_k_top_p(logits, top_k, top_p):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = (
                logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., :1] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        probabilities = torch.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probabilities, 1).squeeze(-1)

        return sampled_tokens

    def batch_generate(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=(0, 261, 24281),
    ):
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()

        finished = [False] * batch_size
        generated_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            sample_rand_states = sample.setup_rand(
                random.randint(0, 2**63 - 1), batch_size
            )
            penalties = torch.zeros(batch_size, 65536).to(0)
            new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(
                out,
                penalties,
                sample_rand_states,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                temperature,
                top_k,
                top_p,
            ).tolist()
            new_tokens = [[token] for token in new_tokens]
            out = self.model.forward_batch(new_tokens, state).float()

            for i in range(batch_size):
                tok = (
                    new_tokens[i][0]
                    if isinstance(new_tokens[i], list)
                    else new_tokens[i]
                )
                if finished[i]:
                    continue
                if tok in stop_tokens:
                    finished[i] = True
                    continue
                generated_tokens[i].append(tok)

            if all(finished):
                break

        del state
        gc.collect()

        decoded = []
        for i in range(batch_size):
            text = self.tokenizer.decode(generated_tokens[i], utf8_errors="ignore")
            decoded.append(text)
        torch.cuda.empty_cache()
        return decoded

    async def batch_infer_stream(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=(0, 261, 24281),
        chunk_size=32,
    ):
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()

        finished = [False] * batch_size
        generated_tokens = [[] for _ in range(batch_size)]
        token_buffers = [[] for _ in range(batch_size)]

        try:
            while not all(finished) and max_length > 0:
                sample_rand_states = sample.setup_rand(
                    random.randint(0, 2**63 - 1), batch_size
                )
                penalties = torch.zeros(batch_size, 65536).to(0)
                new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()
                new_tokens = [[token] for token in new_tokens]
                out = self.model.forward_batch(new_tokens, state).float()
                max_length -= 1

                contents_to_send = [""] * batch_size

                for i in range(batch_size):
                    if finished[i]:
                        continue

                    tok = (
                        new_tokens[i][0]
                        if isinstance(new_tokens[i], list)
                        else new_tokens[i]
                    )

                    if tok in stop_tokens:
                        finished[i] = True
                        if token_buffers[i]:
                            contents_to_send[i] = self.tokenizer.decode(
                                token_buffers[i], utf8_errors="ignore"
                            )
                            token_buffers[i].clear()
                        continue

                    token_buffers[i].append(tok)
                    generated_tokens[i].append(tok)

                    if len(token_buffers[i]) >= chunk_size:
                        contents_to_send[i] = self.tokenizer.decode(
                            token_buffers[i], utf8_errors="ignore"
                        )
                        token_buffers[i].clear()

                if any(contents_to_send):
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": i, "delta": {"content": contents_to_send[i]}}
                            for i in range(batch_size)
                            if contents_to_send[i]
                        ],
                    }
                    if chunk["choices"]:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                await asyncio.sleep(0)

            remaining_contents = [""] * batch_size
            for i in range(batch_size):
                if token_buffers[i]:
                    remaining_contents[i] = self.tokenizer.decode(
                        token_buffers[i], utf8_errors="ignore"
                    )
                    token_buffers[i].clear()

            if any(remaining_contents):
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": i, "delta": {"content": remaining_contents[i]}}
                        for i in range(batch_size)
                        if remaining_contents[i]
                    ],
                }
                if chunk["choices"]:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            del state
            torch.cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

    def batch_generate_state(
        self,
        prompts,
        state,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=(0, 261, 24281),
    ):
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]

        tokens = encoded_prompts[0]
        out = self.model.forward(tokens, state).float()

        generated_tokens = []
        for _ in range(max_length):
            if out.dim() == 1:
                out = out.unsqueeze(0)

            sample_rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = torch.zeros(1, 65536).to(out.device)
            new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(
                out,
                penalties,
                sample_rand_states,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                temperature,
                top_k,
                top_p,
            ).tolist()

            tok = new_tokens[0]

            if tok in stop_tokens:
                break

            generated_tokens.append(tok)
            out = self.model.forward(tok, state).float()
        decoded = [self.tokenizer.decode(generated_tokens, utf8_errors="ignore")]

        gc.collect()
        torch.cuda.empty_cache()
        return decoded

    async def batch_infer_stream_state(
        self,
        prompts,
        state,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=(0, 261, 24281),
        chunk_size=32,
        session_id=None,
        state_manager=None,
    ):
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]

        try:
            tokens = encoded_prompts[0]
            out = self.model.forward(tokens, state).float()

            token_buffer = []

            while max_length > 0:
                max_length -= 1
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                sample_rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), 1)
                penalties = torch.zeros(1, 65536).to(out.device)
                new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]

                if tok in stop_tokens:
                    if token_buffer:
                        content = self.tokenizer.decode(
                            token_buffer, utf8_errors="ignore"
                        )
                        token_buffer.clear()
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": content}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    break

                if not token_buffer:
                    content = self.tokenizer.decode([tok], utf8_errors="ignore")
                    if content:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": content}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        token_buffer.append(tok)
                else:
                    token_buffer.append(tok)
                    if len(token_buffer) >= chunk_size:
                        content = self.tokenizer.decode(
                            token_buffer, utf8_errors="ignore"
                        )
                        token_buffer.clear()
                        if content:
                            chunk = {
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {"index": 0, "delta": {"content": content}}
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                out = self.model.forward(tok, state).float()

                await asyncio.sleep(0)

            if token_buffer:
                content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                token_buffer.clear()
                if content:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            if state_manager and session_id:
                state_manager.put_state(session_id, state)
                print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")

            del state
            torch.cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

    async def graph_generate(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
    ):
        prompt = inputs[0]
        encoded_prompt = self.tokenizer.encode(prompt)
        state = self.model.generate_zero_state(0)

        try:
            if max_generate_tokens <= 0:
                return [""]

            out = self.model.forward(encoded_prompt, state)

            token = sampler_gumbel_batch(logits=out, temp=temperature).item()
            if token in stop_tokens:
                return [""]

            static_input, _static_state, static_output, g = self._init_cuda_graph_state(
                token, state, out
            )

            generated_tokens = [token]

            for _ in range(max_generate_tokens - 1):
                x_emb = self.model.z["emb.weight"][token]
                static_input.copy_(x_emb)

                g.replay()
                token = self._sample_next_token(
                    static_output,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                )
                if token in stop_tokens:
                    break
                generated_tokens.append(token)

            decoded = self.tokenizer.decode(generated_tokens, utf8_errors="ignore")
            return [decoded]
        finally:
            self._cleanup_cuda_state(state)

    async def graph_infer_stream(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
    ):
        prompt = inputs[0]
        if self.rocm_flag or not torch.cuda.is_available():
            async for item in self.dynamic_batch_infer_stream(
                prompt=prompt,
                max_generate_tokens=max_generate_tokens,
                stop_tokens=stop_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_presence=alpha_presence,
                alpha_frequency=alpha_frequency,
                alpha_decay=alpha_decay,
                chunk_size=chunk_size,
            ):
                if item["type"] == "delta" and item["text"]:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": item["text"]}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    continue
                if item["type"] == "done":
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": item["finish_reason"],
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    break
            yield "data: [DONE]\n\n"
            return

        encoded_prompt = self.tokenizer.encode(prompt)
        finish_reason = "length"
        token_buffer = []
        state = self.model.generate_zero_state(0)

        try:
            if max_generate_tokens <= 0:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            out = self.model.forward(encoded_prompt, state)
            token = sampler_gumbel_batch(logits=out, temp=temperature).item()

            if token in stop_tokens:
                finish_reason = "stop"
            else:
                content = self.tokenizer.decode([token], utf8_errors="ignore")
                if content:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    token_buffer.append(token)

                static_input, _static_state, static_output, g = (
                    self._init_cuda_graph_state(token, state, out)
                )

                for _ in range(max_generate_tokens - 1):
                    x_emb = self.model.z["emb.weight"][token]
                    static_input.copy_(x_emb)

                    g.replay()
                    token = self._sample_next_token(
                        static_output,
                        alpha_presence,
                        alpha_frequency,
                        alpha_decay,
                        temperature,
                        top_k,
                        top_p,
                    )
                    if token in stop_tokens:
                        finish_reason = "stop"
                        break

                    token_buffer.append(token)

                    if len(token_buffer) >= chunk_size:
                        content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                        token_buffer.clear()
                        if content:
                            chunk = {
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": {"content": content}}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    await asyncio.sleep(0)

            if token_buffer:
                content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                if content:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            chunk = {
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            self._cleanup_cuda_state(state)

        yield "data: [DONE]\n\n"

    def _continuous_batching_stream_sync(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        batch_size,
        output_queue,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
    ):
        stop_tokens_set = stop_tokens
        max_generate_tokens = max_generate_tokens
        batch_size = batch_size
        pad_zero = pad_zero
        chunk_size = chunk_size

        device = self.model.z["head.weight"].device
        alpha_presence_val = torch.tensor(
            alpha_presence, dtype=torch.float32, device=device
        )

        if temperature == 0:
            temperature = 1.0
            top_k = 1

        encoded_inputs = []
        for prompt in inputs:
            input_token = self.tokenizer.encode(prompt)
            if pad_zero:
                input_token = [0] + input_token
            encoded_inputs.append((prompt, input_token))
        input_queue = deque(encoded_inputs)

        states = self.model.generate_zero_state(batch_size)
        task_pool = []
        token_buffers = {}

        prompt_idx = 0
        for i in range(batch_size):
            prompt, input_token = input_queue.popleft()
            task_pool.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "input_token": input_token,
                    "state_pos": i,
                    "generated_tokens": [],
                    "new_token": None,
                }
            )
            token_buffers[prompt_idx] = []
            prompt_idx += 1

        occurrence = torch.zeros(
            (batch_size, self.args.vocab_size), dtype=torch.float32, device=device
        )
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        alpha_presence_vector = torch.zeros(
            (batch_size, self.args.vocab_size), dtype=torch.float32, device=device
        )

        try:
            while True:
                contents_to_send = {}
                accomplished_task_indices = []
                state_slots_to_remove = set()

                for task_idx, task in enumerate(task_pool):
                    if len(task["input_token"]) == 0:
                        if task["new_token"] is None:
                            continue

                        new_token = task["new_token"]
                        prompt_id = task["prompt_idx"]

                        is_finished = (
                            new_token in stop_tokens_set
                            or len(task["generated_tokens"]) >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_tokens"].append(new_token)
                            token_buffers[prompt_id].append(new_token)

                            if len(token_buffers[prompt_id]) >= chunk_size:
                                text_chunk = self.tokenizer.decode(
                                    token_buffers[prompt_id], utf8_errors="ignore"
                                )
                                contents_to_send[prompt_id] = text_chunk
                                token_buffers[prompt_id].clear()

                        if is_finished:
                            if token_buffers[prompt_id]:
                                text_chunk = self.tokenizer.decode(
                                    token_buffers[prompt_id], utf8_errors="ignore"
                                )
                                contents_to_send[prompt_id] = (
                                    contents_to_send.get(prompt_id, "") + text_chunk
                                )
                                token_buffers[prompt_id].clear()

                            del token_buffers[prompt_id]

                            if len(input_queue) > 0:
                                prompt, input_token = input_queue.popleft()
                                new_prompt_idx = prompt_idx
                                task_pool[task_idx] = {
                                    "prompt_idx": new_prompt_idx,
                                    "prompt": prompt,
                                    "input_token": input_token,
                                    "state_pos": task["state_pos"],
                                    "generated_tokens": [],
                                    "new_token": None,
                                }
                                token_buffers[new_prompt_idx] = []
                                prompt_idx += 1

                                state_pos = task["state_pos"]
                                states[0][:, :, state_pos, :] = 0
                                states[1][:, state_pos, :, :] = 0
                                occurrence[state_pos, :] = 0
                                alpha_presence_vector[state_pos, :] = 0
                            else:
                                accomplished_task_indices.append(task_idx)
                                state_slots_to_remove.add(task["state_pos"])
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in no_penalty_token_ids else 1.0
                            occurrence[task["state_pos"], new_token] += www
                            alpha_presence_vector[task["state_pos"], new_token] = (
                                alpha_presence_val
                            )

                if contents_to_send:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": pid, "delta": {"content": content}}
                            for pid, content in contents_to_send.items()
                            if content
                        ],
                    }
                    if chunk["choices"]:
                        output_queue.put(
                            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        )

                if accomplished_task_indices:
                    sorted_slots = sorted(list(state_slots_to_remove), reverse=True)

                    for slot in sorted_slots:
                        states[0] = torch.cat(
                            [states[0][:, :, :slot, :], states[0][:, :, slot + 1 :, :]],
                            dim=2,
                        )
                        states[1] = torch.cat(
                            [states[1][:, :slot, :, :], states[1][:, slot + 1 :, :, :]],
                            dim=1,
                        )
                        occurrence = torch.cat(
                            [occurrence[:slot, :], occurrence[slot + 1 :, :]], dim=0
                        )
                        alpha_presence_vector = torch.cat(
                            [
                                alpha_presence_vector[:slot, :],
                                alpha_presence_vector[slot + 1 :, :],
                            ],
                            dim=0,
                        )

                    for task_idx in sorted(accomplished_task_indices, reverse=True):
                        del task_pool[task_idx]

                    remaining_slots = sorted([t["state_pos"] for t in task_pool])
                    pos_map = {
                        old_pos: new_pos
                        for new_pos, old_pos in enumerate(remaining_slots)
                    }
                    for task in task_pool:
                        task["state_pos"] = pos_map[task["state_pos"]]

                if len(task_pool) == 0:
                    break

                current_batch_size = len(task_pool)
                next_tokens = [None] * current_batch_size
                for task in task_pool:
                    next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]

                out = self.model.forward_batch(next_tokens, states)

                if alpha_presence != 0 or alpha_frequency != 0:
                    mask = (occurrence > 0).float()
                    out -= mask * alpha_presence + occurrence * alpha_frequency

                occurrence *= alpha_decay
                out -= alpha_presence_vector + occurrence * alpha_frequency

                if temperature != 1.0:
                    out /= temperature

                if self.rocm_flag:
                    new_tokens = self._torch_top_k_top_p(out, top_k, top_p)
                else:
                    try:
                        import flashinfer  # type: ignore

                        new_tokens = (
                            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                out, top_k, top_p
                            )
                        )
                    except Exception:
                        new_tokens = self._torch_top_k_top_p(out, top_k, top_p)

                new_tokens = new_tokens.tolist()

                for task in task_pool:
                    state_pos = task["state_pos"]
                    task["new_token"] = new_tokens[state_pos]

        finally:
            del states
            del occurrence
            del alpha_presence_vector
            gc.collect()
            torch.cuda.empty_cache()
            output_queue.put("EOF")

    def _continuous_batching_sync(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        batch_size,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
    ):
        stop_tokens_set = stop_tokens
        max_generate_tokens = max_generate_tokens
        batch_size = batch_size
        pad_zero = pad_zero

        device = self.model.z["head.weight"].device
        alpha_presence_val = torch.tensor(
            alpha_presence, dtype=torch.float32, device=device
        )

        if temperature == 0:
            temperature = 1.0
            top_k = 1

        encoded_inputs = []
        for prompt in inputs:
            input_token = self.tokenizer.encode(prompt)
            if pad_zero:
                input_token = [0] + input_token
            encoded_inputs.append((prompt, input_token))
        input_queue = deque(encoded_inputs)

        states = self.model.generate_zero_state(batch_size)
        task_pool = []
        results = {}

        prompt_idx = 0
        for i in range(batch_size):
            prompt, input_token = input_queue.popleft()
            task_pool.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "input_token": input_token,
                    "state_pos": i,
                    "generated_tokens": [],
                    "new_token": None,
                }
            )
            prompt_idx += 1

        occurrence = torch.zeros(
            (batch_size, self.args.vocab_size), dtype=torch.float32, device=device
        )
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        alpha_presence_vector = torch.zeros(
            (batch_size, self.args.vocab_size), dtype=torch.float32, device=device
        )

        try:
            while True:
                accomplished_task_indices = []
                state_slots_to_remove = set()

                for task_idx, task in enumerate(task_pool):
                    if len(task["input_token"]) == 0:
                        if task["new_token"] is None:
                            continue

                        new_token = task["new_token"]
                        prompt_id = task["prompt_idx"]

                        is_finished = (
                            new_token in stop_tokens_set
                            or len(task["generated_tokens"]) >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_tokens"].append(new_token)

                        if is_finished:
                            if task["generated_tokens"]:
                                text = self.tokenizer.decode(
                                    task["generated_tokens"], utf8_errors="ignore"
                                )
                                results[prompt_id] = text
                            else:
                                results[prompt_id] = ""

                            if len(input_queue) > 0:
                                prompt, input_token = input_queue.popleft()
                                new_prompt_idx = prompt_idx
                                task_pool[task_idx] = {
                                    "prompt_idx": new_prompt_idx,
                                    "prompt": prompt,
                                    "input_token": input_token,
                                    "state_pos": task["state_pos"],
                                    "generated_tokens": [],
                                    "new_token": None,
                                }
                                prompt_idx += 1

                                state_pos = task["state_pos"]
                                states[0][:, :, state_pos, :] = 0
                                states[1][:, state_pos, :, :] = 0
                                occurrence[state_pos, :] = 0
                                alpha_presence_vector[state_pos, :] = 0
                            else:
                                accomplished_task_indices.append(task_idx)
                                state_slots_to_remove.add(task["state_pos"])
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in no_penalty_token_ids else 1.0
                            occurrence[task["state_pos"], new_token] += www
                            alpha_presence_vector[task["state_pos"], new_token] = (
                                alpha_presence_val
                            )

                if accomplished_task_indices:
                    sorted_slots = sorted(list(state_slots_to_remove), reverse=True)

                    for slot in sorted_slots:
                        states[0] = torch.cat(
                            [states[0][:, :, :slot, :], states[0][:, :, slot + 1 :, :]],
                            dim=2,
                        )
                        states[1] = torch.cat(
                            [states[1][:, :slot, :, :], states[1][:, slot + 1 :, :, :]],
                            dim=1,
                        )
                        occurrence = torch.cat(
                            [occurrence[:slot, :], occurrence[slot + 1 :, :]], dim=0
                        )
                        alpha_presence_vector = torch.cat(
                            [
                                alpha_presence_vector[:slot, :],
                                alpha_presence_vector[slot + 1 :, :],
                            ],
                            dim=0,
                        )

                    for task_idx in sorted(accomplished_task_indices, reverse=True):
                        del task_pool[task_idx]

                    remaining_slots = sorted([t["state_pos"] for t in task_pool])
                    pos_map = {
                        old_pos: new_pos
                        for new_pos, old_pos in enumerate(remaining_slots)
                    }
                    for task in task_pool:
                        task["state_pos"] = pos_map[task["state_pos"]]

                if len(task_pool) == 0:
                    break

                current_batch_size = len(task_pool)
                next_tokens = [None] * current_batch_size
                for task in task_pool:
                    next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]

                out = self.model.forward_batch(next_tokens, states)

                if alpha_presence != 0 or alpha_frequency != 0:
                    mask = (occurrence > 0).float()
                    out -= mask * alpha_presence + occurrence * alpha_frequency

                occurrence *= alpha_decay
                out -= alpha_presence_vector + occurrence * alpha_frequency

                if temperature != 1.0:
                    out /= temperature

                if self.rocm_flag:
                    new_tokens = self._torch_top_k_top_p(out, top_k, top_p)
                else:
                    try:
                        import flashinfer  # type: ignore

                        new_tokens = (
                            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                out, top_k, top_p
                            )
                        )
                    except Exception:
                        new_tokens = self._torch_top_k_top_p(out, top_k, top_p)

                new_tokens = new_tokens.tolist()

                for task in task_pool:
                    state_pos = task["state_pos"]
                    task["new_token"] = new_tokens[state_pos]

        finally:
            del states
            del occurrence
            del alpha_presence_vector
            gc.collect()
            torch.cuda.empty_cache()

        return [results.get(i, "") for i in range(len(inputs))]

    async def continuous_batching_stream(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        batch_size,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
        chunk_size=32,
    ):
        from queue import Queue

        output_queue = Queue()

        loop = asyncio.get_event_loop()

        with self.model_lock:
            future = loop.run_in_executor(
                self.executor,
                self._continuous_batching_stream_sync,
                inputs,
                stop_tokens,
                max_generate_tokens,
                batch_size,
                output_queue,
                pad_zero,
                temperature,
                top_k,
                top_p,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                chunk_size,
            )

        while True:
            try:
                await asyncio.sleep(0.01)

                while not output_queue.empty():
                    data = output_queue.get_nowait()
                    if data == "EOF":
                        yield "data: [DONE]\n\n"
                        await future
                        return
                    yield data

                if future.done():
                    while not output_queue.empty():
                        data = output_queue.get_nowait()
                        if data == "EOF":
                            yield "data: [DONE]\n\n"
                            return
                        yield data
                    break
            except Exception as exc:
                print(f"Error in stream: {exc}")
                break

        yield "data: [DONE]\n\n"

    def continuous_batching(
        self,
        inputs,
        stop_tokens,
        max_generate_tokens,
        batch_size,
        pad_zero=True,
        temperature=1,
        top_k=50,
        top_p=0.3,
        alpha_presence=0.5,
        alpha_frequency=0.5,
        alpha_decay=0.996,
    ):
        return self._continuous_batching_sync(
            inputs=inputs,
            stop_tokens=stop_tokens,
            max_generate_tokens=max_generate_tokens,
            batch_size=batch_size,
            pad_zero=pad_zero,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            alpha_presence=alpha_presence,
            alpha_frequency=alpha_frequency,
            alpha_decay=alpha_decay,
        )

    async def big_batch_stream(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        stop_tokens=(0, 261, 24281),
        chunk_size=32,
    ):
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state)

        finished = [False] * batch_size
        generated_tokens = [[] for _ in range(batch_size)]
        token_buffers = [[] for _ in range(batch_size)]

        try:
            step_count = 0
            cleanup_interval = 100

            while not all(finished) and max_length > 0:
                new_tokens = sampler_gumbel_batch(logits=out, temp=temperature).tolist()
                out = self.model.forward_batch(new_tokens, state)
                max_length -= 1
                step_count += 1

                contents_to_send = [""] * batch_size

                for i in range(batch_size):
                    if finished[i]:
                        continue

                    tok = (
                        new_tokens[i][0]
                        if isinstance(new_tokens[i], list)
                        else new_tokens[i]
                    )

                    if tok in stop_tokens:
                        finished[i] = True
                        if token_buffers[i]:
                            contents_to_send[i] = self.tokenizer.decode(
                                token_buffers[i], utf8_errors="ignore"
                            )
                            token_buffers[i].clear()
                        continue

                    token_buffers[i].append(tok)
                    generated_tokens[i].append(tok)

                    if len(token_buffers[i]) >= chunk_size:
                        contents_to_send[i] = self.tokenizer.decode(
                            token_buffers[i], utf8_errors="ignore"
                        )
                        token_buffers[i].clear()

                if any(contents_to_send):
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": i, "delta": {"content": contents_to_send[i]}}
                            for i in range(batch_size)
                            if contents_to_send[i]
                        ],
                    }
                    if chunk["choices"]:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                await asyncio.sleep(0)

                if step_count % cleanup_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            remaining_contents = [""] * batch_size
            for i in range(batch_size):
                if token_buffers[i]:
                    remaining_contents[i] = self.tokenizer.decode(
                        token_buffers[i], utf8_errors="ignore"
                    )
                    token_buffers[i].clear()

            if any(remaining_contents):
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": i, "delta": {"content": remaining_contents[i]}}
                        for i in range(batch_size)
                        if remaining_contents[i]
                    ],
                }
                if chunk["choices"]:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            del state
            del encoded_prompts
            del out
            del finished
            del generated_tokens
            del token_buffers
            torch.cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

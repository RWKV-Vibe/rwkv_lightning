import asyncio
import gc
import json
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import torch

from infer.rwkv_batch.sampler import sample
from infer.rwkv_batch.utils import sampler_simple


class InferenceEngine:
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=128, thread_name_prefix="model_inference")

    def shutdown(self):
        self.executor.shutdown(wait=False)

    @staticmethod
    def _torch_top_k_top_p(logits, top_k, top_p):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., :1] = False

            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
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
            sample_rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), batch_size)
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
                tok = new_tokens[i][0] if isinstance(new_tokens[i], list) else new_tokens[i]
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
                sample_rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), batch_size)
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

                    tok = new_tokens[i][0] if isinstance(new_tokens[i], list) else new_tokens[i]

                    if tok in stop_tokens:
                        finished[i] = True
                        if token_buffers[i]:
                            contents_to_send[i] = self.tokenizer.decode(token_buffers[i], utf8_errors="ignore")
                            token_buffers[i].clear()
                        continue

                    token_buffers[i].append(tok)
                    generated_tokens[i].append(tok)

                    if len(token_buffers[i]) >= chunk_size:
                        contents_to_send[i] = self.tokenizer.decode(token_buffers[i], utf8_errors="ignore")
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
                    remaining_contents[i] = self.tokenizer.decode(token_buffers[i], utf8_errors="ignore")
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
                        content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                        token_buffer.clear()
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": content}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    break

                token_buffer.append(tok)
                if len(token_buffer) >= chunk_size:
                    content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                    token_buffer.clear()
                    if content:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": content}}],
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

        out = self.model.forward(encoded_prompt, state)

        token = sampler_simple(out, noise=0, temp=temperature).item()

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

        generated_tokens = [token]

        for _ in range(max_generate_tokens):
            x_emb = self.model.z["emb.weight"][token]
            static_input.copy_(x_emb)

            g.replay()

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
            token = new_tokens[0]
            generated_tokens.append(token)
            if token in stop_tokens:
                break

        del state
        del static_state
        del static_input
        del static_output
        del g
        gc.collect()
        torch.cuda.empty_cache()

        decoded = self.tokenizer.decode(generated_tokens, utf8_errors="ignore")
        return [decoded]

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
        encoded_prompt = self.tokenizer.encode(prompt)

        state = self.model.generate_zero_state(0)

        out = self.model.forward(encoded_prompt, state)

        token = sampler_simple(out, noise=0, temp=temperature).item()

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

        token_buffer = [token]

        try:
            if token not in stop_tokens:
                if len(token_buffer) >= chunk_size:
                    content = self.tokenizer.decode(token_buffer, utf8_errors="ignore")
                    token_buffer.clear()
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            for _ in range(max_generate_tokens):
                x_emb = self.model.z["emb.weight"][token]
                static_input.copy_(x_emb)

                g.replay()

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
                token = new_tokens[0]
                if token in stop_tokens:
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

        finally:
            del state
            del static_state
            del static_input
            del static_output
            del g
            gc.collect()
            torch.cuda.empty_cache()

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
        alpha_presence_val = torch.tensor(alpha_presence, dtype=torch.float32, device=device)

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

        occurrence = torch.zeros((batch_size, self.args.vocab_size), dtype=torch.float32, device=device)
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        alpha_presence_vector = torch.zeros((batch_size, self.args.vocab_size), dtype=torch.float32, device=device)

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

                        is_finished = new_token in stop_tokens_set or len(task["generated_tokens"]) >= max_generate_tokens

                        if not is_finished:
                            task["generated_tokens"].append(new_token)
                            token_buffers[prompt_id].append(new_token)

                            if len(token_buffers[prompt_id]) >= chunk_size:
                                text_chunk = self.tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                                contents_to_send[prompt_id] = text_chunk
                                token_buffers[prompt_id].clear()

                        if is_finished:
                            if token_buffers[prompt_id]:
                                text_chunk = self.tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                                contents_to_send[prompt_id] = contents_to_send.get(prompt_id, "") + text_chunk
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
                            alpha_presence_vector[task["state_pos"], new_token] = alpha_presence_val

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
                        output_queue.put(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")

                if accomplished_task_indices:
                    sorted_slots = sorted(list(state_slots_to_remove), reverse=True)

                    for slot in sorted_slots:
                        states[0] = torch.cat([states[0][:, :, :slot, :], states[0][:, :, slot + 1 :, :]], dim=2)
                        states[1] = torch.cat([states[1][:, :slot, :, :], states[1][:, slot + 1 :, :, :]], dim=1)
                        occurrence = torch.cat([occurrence[:slot, :], occurrence[slot + 1 :, :]], dim=0)
                        alpha_presence_vector = torch.cat(
                            [alpha_presence_vector[:slot, :], alpha_presence_vector[slot + 1 :, :]], dim=0
                        )

                    for task_idx in sorted(accomplished_task_indices, reverse=True):
                        del task_pool[task_idx]

                    remaining_slots = sorted([t["state_pos"] for t in task_pool])
                    pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
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

                        new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
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
        alpha_presence_val = torch.tensor(alpha_presence, dtype=torch.float32, device=device)

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

        occurrence = torch.zeros((batch_size, self.args.vocab_size), dtype=torch.float32, device=device)
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        alpha_presence_vector = torch.zeros((batch_size, self.args.vocab_size), dtype=torch.float32, device=device)

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

                        is_finished = new_token in stop_tokens_set or len(task["generated_tokens"]) >= max_generate_tokens

                        if not is_finished:
                            task["generated_tokens"].append(new_token)

                        if is_finished:
                            if task["generated_tokens"]:
                                text = self.tokenizer.decode(task["generated_tokens"], utf8_errors="ignore")
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
                            alpha_presence_vector[task["state_pos"], new_token] = alpha_presence_val

                if accomplished_task_indices:
                    sorted_slots = sorted(list(state_slots_to_remove), reverse=True)

                    for slot in sorted_slots:
                        states[0] = torch.cat([states[0][:, :, :slot, :], states[0][:, :, slot + 1 :, :]], dim=2)
                        states[1] = torch.cat([states[1][:, :slot, :, :], states[1][:, slot + 1 :, :, :]], dim=1)
                        occurrence = torch.cat([occurrence[:slot, :], occurrence[slot + 1 :, :]], dim=0)
                        alpha_presence_vector = torch.cat(
                            [alpha_presence_vector[:slot, :], alpha_presence_vector[slot + 1 :, :]], dim=0
                        )

                    for task_idx in sorted(accomplished_task_indices, reverse=True):
                        del task_pool[task_idx]

                    remaining_slots = sorted([t["state_pos"] for t in task_pool])
                    pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
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

                        new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
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

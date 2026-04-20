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


class PagedStateManager:
    def __init__(self, model, vocab_size, device, decode_capacity, prefill_capacity):
        self.model = model
        self.device = device
        self.vocab_size = vocab_size
        self.decode_capacity = decode_capacity
        self.prefill_capacity = prefill_capacity
        self.total_pages = decode_capacity + prefill_capacity
        self.states = model.generate_zero_state(self.total_pages)
        self.occurrence = torch.zeros(
            (self.total_pages, vocab_size), dtype=torch.float32, device=device
        )
        self.alpha_presence_vector = torch.zeros(
            (self.total_pages, vocab_size), dtype=torch.float32, device=device
        )
        self.free_pages = deque(range(self.total_pages))

    def allocate_page(self):
        if not self.free_pages:
            return None
        return self.free_pages.popleft()

    def release_page(self, page_idx):
        self.reset_page(page_idx)
        self.free_pages.append(page_idx)

    def reset_page(self, page_idx):
        self.states[0][:, :, page_idx, :].zero_()
        self.states[1][:, page_idx, :, :, :].zero_()
        self.states[2][page_idx] = 0
        self.occurrence[page_idx, :].zero_()
        self.alpha_presence_vector[page_idx, :].zero_()

    def page_table(self, tasks):
        return torch.tensor(
            [task["state_pos"] for task in tasks],
            device=self.device,
            dtype=torch.int32,
        )


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
        self._cuda_graph_sessions = {}
        self._batch_cuda_graphs = {}
        self._flashinfer = None
        self._flashinfer_checked = False
        self._continuous_prefill_candidates = (256, 128, 64, 32, 16, 8, 4, 2)

    def shutdown(self):
        self.executor.shutdown(wait=False)
        for session_id in list(self._cuda_graph_sessions.keys()):
            self.cleanup_cuda_graph_session(session_id)
        self.cleanup_batch_cuda_graphs()

    @staticmethod
    def _normalize_stop_strings(stop_tokens):
        if not stop_tokens:
            return ()
        return tuple(token for token in stop_tokens if isinstance(token, str) and token)

    @staticmethod
    def _match_stop_suffix(decoded_text, stop_strings):
        for stop_string in stop_strings:
            if decoded_text.endswith(stop_string):
                return stop_string
        return None

    def _create_stop_state(self, stop_tokens):
        return {
            "stop_strings": self._normalize_stop_strings(stop_tokens),
            "pending_tokens": deque(),
            "window_size": 6,
        }

    def _ingest_token_with_stop(self, stop_state, token):
        if token == 0:
            return "", True

        pending_tokens = stop_state["pending_tokens"]
        pending_tokens.append(token)

        decoded_window = self.tokenizer.decode(
            list(pending_tokens), utf8_errors="ignore"
        )
        matched_stop = self._match_stop_suffix(
            decoded_window, stop_state["stop_strings"]
        )
        if matched_stop is not None:
            pending_tokens.clear()
            return decoded_window[: -len(matched_stop)], True

        if len(pending_tokens) > stop_state["window_size"]:
            pending_tokens.popleft()
            trailing_text = self.tokenizer.decode(
                list(pending_tokens), utf8_errors="ignore"
            )
            if not trailing_text:
                return decoded_window, False
            if decoded_window.endswith(trailing_text):
                return decoded_window[: -len(trailing_text)], False

            return self.tokenizer.decode([token], utf8_errors="ignore"), False

        return "", False

    def _flush_stop_state(self, stop_state, final=False):
        pending_tokens = stop_state["pending_tokens"]
        if not pending_tokens:
            return ""

        if not final and len(pending_tokens) <= stop_state["window_size"]:
            return ""

        decoded = self.tokenizer.decode(list(pending_tokens), utf8_errors="ignore")
        pending_tokens.clear()
        return decoded

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

    def _create_sampling_state(self, batch_size, device, vocab_size=None):
        return {
            "penalties": torch.zeros(
                (batch_size, vocab_size or self.args.vocab_size),
                dtype=torch.float32,
                device=device,
            ),
            "rand_states": sample.setup_rand(random.randint(0, 2**63 - 1), batch_size),
        }

    def _sample_with_repetition(
        self,
        logits,
        sampling_state,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        temperature,
        top_k,
        top_p,
    ):
        logits_reshaped = logits.unsqueeze(0).float() if logits.dim() == 1 else logits.float()
        return sample.batch_sampling_repetition_temperature_topk_topp(
            logits_reshaped,
            sampling_state["penalties"],
            sampling_state["rand_states"],
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            temperature,
            top_k,
            top_p,
        )

    def _sample_top_k_top_p(self, logits, top_k, top_p, temperature=1.0):
        logits_reshaped = logits.unsqueeze(0).float() if logits.dim() == 1 else logits.float()
        if temperature != 1.0:
            logits_reshaped = logits_reshaped / temperature

        if self.rocm_flag:
            return self._torch_top_k_top_p(logits_reshaped, top_k, top_p)

        flashinfer = self._get_flashinfer()
        if flashinfer is not None:
            return flashinfer.sampling.top_k_top_p_sampling_from_logits(
                logits_reshaped, top_k, top_p
            )
        return self._torch_top_k_top_p(logits_reshaped, top_k, top_p)

    def _get_flashinfer(self):
        if self.rocm_flag:
            return None
        if not self._flashinfer_checked:
            try:
                import flashinfer  # type: ignore

                self._flashinfer = flashinfer
            except Exception:
                self._flashinfer = None
            self._flashinfer_checked = True
        return self._flashinfer

    @staticmethod
    def _sample_throughput_tokens(logits, temperature):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        return sampler_gumbel_batch(logits=logits, temp=temperature)

    def _select_continuous_prefill_group(self, task_pool):
        prefill_tasks = [
            task
            for task in task_pool
            if task["pending_out"] is None
            and task["new_token"] is None
            and not task["generated_tokens"]
            and len(task["input_token"]) > 0
        ]
        if not prefill_tasks:
            return [], 0

        best_tasks = []
        best_step = 0
        best_score = 0
        max_remaining = max(len(task["input_token"]) for task in prefill_tasks)
        candidates = [
            step for step in self._continuous_prefill_candidates if step <= max_remaining
        ]
        if max_remaining not in candidates:
            candidates = [min(max_remaining, 512)] + candidates

        for step in candidates:
            eligible = [task for task in prefill_tasks if len(task["input_token"]) >= step]
            if not eligible:
                continue
            score = step * len(eligible)
            if score > best_score or (score == best_score and step > best_step):
                best_score = score
                best_step = step
                best_tasks = eligible

        return best_tasks, best_step

    def _apply_continuous_prefill(self, tasks, states, step_len):
        if not tasks or step_len <= 0:
            return

        slot_indices = [task["state_pos"] for task in tasks]
        device = self.model.z["head.weight"].device
        batch_tokens = torch.tensor(
            [task["input_token"][:step_len] for task in tasks],
            device=device,
            dtype=torch.long,
        )
        page_table = torch.tensor(slot_indices, device=device, dtype=torch.int32)
        out = self.model.forward_batch_paged(batch_tokens, states, page_table)

        for idx, task in enumerate(tasks):
            del task["input_token"][:step_len]
            if not task["input_token"]:
                task["pending_out"] = out[idx]

    @staticmethod
    def _ready_for_decode(task):
        if len(task["input_token"]) == 0:
            return False
        if len(task["input_token"]) == 1:
            return True
        return bool(task["generated_tokens"] or task["new_token"] is not None)

    @staticmethod
    def _state_token_count(state):
        token_count = state[2]
        if token_count.numel() == 0:
            return 0
        return int(token_count.reshape(-1)[0].item())

    @staticmethod
    def _cleanup_cuda_state(state):
        del state
        gc.collect()

    @staticmethod
    def _cleanup_cuda_memory():
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    @staticmethod
    def _copy_state_slot(states, occurrence, alpha_presence_vector, dst_slot, src_slot):
        if dst_slot == src_slot:
            return
        states[0][:, :, dst_slot, :] = states[0][:, :, src_slot, :]
        states[1][:, dst_slot, :, :, :] = states[1][:, src_slot, :, :, :]
        states[2][dst_slot] = states[2][src_slot]
        occurrence[dst_slot, :] = occurrence[src_slot, :]
        alpha_presence_vector[dst_slot, :] = alpha_presence_vector[src_slot, :]

    @staticmethod
    def _reset_state_slot(states, occurrence, alpha_presence_vector, slot):
        states[0][:, :, slot, :].zero_()
        states[1][:, slot, :, :, :].zero_()
        states[2][slot] = 0
        occurrence[slot, :].zero_()
        alpha_presence_vector[slot, :].zero_()

    def _release_finished_tasks(
        self,
        task_pool,
        finished_task_indices,
    ):
        for task_idx in sorted(finished_task_indices, reverse=True):
            task_pool.pop(task_idx)

    def _continuous_prefill_capacity(self, decode_capacity):
        return max(1, min(4, decode_capacity))

    def _launch_prefill_tasks(
        self,
        input_queue,
        state_manager,
        prefill_tasks,
        prompt_idx,
        stop_tokens,
        prompt_state_store,
        pad_zero=True,
        prefix_cache_manager=None,
    ):
        while (
            input_queue
            and len(prefill_tasks) < state_manager.prefill_capacity
            and state_manager.free_pages
        ):
            prompt = input_queue.popleft()
            page_idx = state_manager.allocate_page()
            prefill_tasks.append(
                self._initialize_continuous_task(
                    prompt,
                    prompt_idx,
                    page_idx,
                    state_manager.states,
                    state_manager.occurrence,
                    state_manager.alpha_presence_vector,
                    pad_zero=pad_zero,
                    prefix_cache_manager=prefix_cache_manager,
                )
            )
            prompt_state_store(prompt_idx)
            prompt_idx += 1
        return prompt_idx

    def _promote_ready_prefill_tasks(
        self,
        prefill_tasks,
        decode_tasks,
        decode_capacity,
    ):
        if len(decode_tasks) >= decode_capacity:
            return

        promoted_indices = []
        for idx, task in enumerate(prefill_tasks):
            if len(decode_tasks) >= decode_capacity:
                break
            if len(task["input_token"]) != 0:
                continue
            decode_tasks.append(task)
            promoted_indices.append(idx)

        for idx in reversed(promoted_indices):
            prefill_tasks.pop(idx)

    def _prefill_prompt_from_state_or_prefix(
        self, prompt, state=None, prefix_cache_manager=None
    ):
        if prefix_cache_manager is not None and state is not None and self._state_token_count(state) == 0:
            encoded_prompt, cached_state, out, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            if cached_state is not state:
                state[0].copy_(cached_state[0])
                state[1].copy_(cached_state[1])
                state[2].copy_(cached_state[2])
            return encoded_prompt, state, out

        encoded_prompt = self.tokenizer.encode(prompt)
        if not encoded_prompt:
            raise ValueError("Empty prompt")
        if state is None:
            state = self.model.generate_zero_state(0)
        out = self.model.forward(encoded_prompt, state).float()
        return encoded_prompt, state, out

    def _initialize_continuous_task(
        self,
        prompt,
        prompt_idx,
        state_pos,
        states,
        occurrence,
        alpha_presence_vector,
        pad_zero=True,
        prefix_cache_manager=None,
    ):
        encoded_prompt = self.tokenizer.encode(prompt)
        if not encoded_prompt:
            raise ValueError("Empty prompt")

        prompt_tokens = ([0] + encoded_prompt) if pad_zero else encoded_prompt
        pending_out = None

        if prefix_cache_manager is not None:
            cache_match = prefix_cache_manager.match_prefix_state(prompt_tokens, device="cuda")
            if cache_match is not None:
                matched_state = cache_match["state"]
                states[0][:, :, state_pos, :] = matched_state[0]
                states[1][:, state_pos, :, :, :] = matched_state[1]
                states[2][state_pos] = matched_state[2].reshape(-1)[0]
                occurrence[state_pos, :].zero_()
                alpha_presence_vector[state_pos, :].zero_()
                pending_out = cache_match["logits"]
                prompt_tokens = prompt_tokens[int(cache_match["matched_tokens"]) :]
                if pending_out is None and not prompt_tokens:
                    pending_out = None
                    prompt_tokens = ([0] + encoded_prompt) if pad_zero else encoded_prompt
                    self._reset_state_slot(
                        states, occurrence, alpha_presence_vector, state_pos
                    )
            else:
                self._reset_state_slot(states, occurrence, alpha_presence_vector, state_pos)
        else:
            self._reset_state_slot(states, occurrence, alpha_presence_vector, state_pos)

        return {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_token": prompt_tokens,
            "state_pos": state_pos,
            "generated_tokens": [],
            "new_token": None,
            "pending_out": pending_out,
        }

    def _get_or_create_cuda_graph(self, session_id, token, state, out):
        if session_id in self._cuda_graph_sessions:
            sess = self._cuda_graph_sessions[session_id]
            sess["static_input"].copy_(self.model.z["emb.weight"][token])
            sess["static_state"][0].copy_(state[0])
            sess["static_state"][1].copy_(state[1])
            sess["static_state"][2].copy_(state[2])
            sess["static_output"].copy_(out)
            return sess

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

        sess = {
            "static_input": static_input,
            "static_state": static_state,
            "static_output": static_output,
            "graph": g,
        }
        self._cuda_graph_sessions[session_id] = sess
        return sess

    def cleanup_cuda_graph_session(self, session_id):
        sess = self._cuda_graph_sessions.pop(session_id, None)
        if sess is not None:
            del sess["graph"]
            del sess["static_output"]
            del sess["static_state"]
            del sess["static_input"]
            gc.collect()
            torch.cuda.empty_cache()

    def _get_or_create_batch_cuda_graph(self, batch_size, token_tensor, state, out):
        if batch_size in self._batch_cuda_graphs:
            sess = self._batch_cuda_graphs[batch_size]
            sess["static_tokens"].copy_(token_tensor)
            sess["static_state"][0].copy_(state[0])
            sess["static_state"][1].copy_(state[1])
            sess["static_state"][2].copy_(state[2])
            sess["static_output"].copy_(out)
            return sess

        static_tokens = torch.empty_like(token_tensor, device=token_tensor.device)
        static_state = [None, None, None]
        static_state[0] = torch.empty_like(state[0], device="cuda")
        static_state[1] = torch.empty_like(state[1], device="cuda")
        static_state[2] = torch.empty_like(state[2], device="cuda")
        static_output = torch.empty_like(out, device=out.device)

        static_output = self.model.forward_batch(static_tokens, static_state)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.model.forward_batch(static_tokens, static_state)

        static_tokens.copy_(token_tensor)
        static_state[0].copy_(state[0])
        static_state[1].copy_(state[1])
        static_state[2].copy_(state[2])
        static_output.copy_(out)

        sess = {
            "static_tokens": static_tokens,
            "static_state": static_state,
            "static_output": static_output,
            "graph": g,
        }
        self._batch_cuda_graphs[batch_size] = sess
        return sess

    def cleanup_batch_cuda_graphs(self):
        for batch_size in list(self._batch_cuda_graphs.keys()):
            sess = self._batch_cuda_graphs.pop(batch_size, None)
            if sess is None:
                continue
            del sess["graph"]
            del sess["static_output"]
            del sess["static_state"]
            del sess["static_tokens"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _sync_cuda_graph_state(static_state, state):
        state[0].copy_(static_state[0])
        state[1].copy_(static_state[1])
        state[2].copy_(static_state[2])

    def graph_generate_state(
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
        stop_tokens=("\nUser:",),
        session_id=None,
        prefix_cache_manager=None,
    ):
        _, state, out = self._prefill_prompt_from_state_or_prefix(
            prompts[0], state, prefix_cache_manager=prefix_cache_manager
        )
        sampling_state = self._create_sampling_state(1, out.device)
        first_tokens = self._sample_with_repetition(
            out,
            sampling_state,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            temperature,
            top_k,
            top_p,
        ).tolist()
        tok = first_tokens[0]

        sess = self._get_or_create_cuda_graph(session_id, tok, state, out)
        static_input = sess["static_input"]
        static_state = sess["static_state"]
        static_output = sess["static_output"]
        g = sess["graph"]

        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        content, should_stop = self._ingest_token_with_stop(stop_state, tok)
        if content:
            generated_text += content

        for _ in range(max_length - 1):
            if should_stop:
                break
            static_input.copy_(self.model.z["emb.weight"][tok])
            g.replay()
            out = static_output.float()
            new_tokens = self._sample_with_repetition(
                out,
                sampling_state,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                temperature,
                top_k,
                top_p,
            ).tolist()
            tok = new_tokens[0]

            content, should_stop = self._ingest_token_with_stop(stop_state, tok)
            if content:
                generated_text += content

        self._sync_cuda_graph_state(static_state, state)
        generated_text += self._flush_stop_state(stop_state, final=True)
        return [generated_text]

    async def graph_infer_stream_state(
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
        stop_tokens=("\nUser:",),
        chunk_size=32,
        session_id=None,
        state_manager=None,
        prefix_cache_manager=None,
    ):
        chunk_size = max(1, int(chunk_size))
        static_state = None
        _, state, out = self._prefill_prompt_from_state_or_prefix(
            prompts[0], state, prefix_cache_manager=prefix_cache_manager
        )
        sampling_state = self._create_sampling_state(1, out.device)
        first_tokens = self._sample_with_repetition(
            out,
            sampling_state,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            temperature,
            top_k,
            top_p,
        ).tolist()
        tok = first_tokens[0]

        sess = self._get_or_create_cuda_graph(session_id, tok, state, out)
        static_input = sess["static_input"]
        static_state = sess["static_state"]
        static_output = sess["static_output"]
        g = sess["graph"]

        stop_state = self._create_stop_state(stop_tokens)
        buffered_tokens = 0
        text_buffer = ""

        content, should_stop = self._ingest_token_with_stop(stop_state, tok)
        if content:
            text_buffer += content

        try:
            while max_length > 1:
                max_length -= 1
                if should_stop:
                    flushed = self._flush_stop_state(stop_state, final=True)
                    if flushed:
                        text_buffer += flushed
                    if text_buffer:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    break

                static_input.copy_(self.model.z["emb.weight"][tok])
                g.replay()
                out = static_output.float()
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()
                tok = new_tokens[0]

                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    text_buffer += content

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and text_buffer:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    text_buffer = ""
                    buffered_tokens = 0

                await asyncio.sleep(0)

            flushed = self._flush_stop_state(stop_state, final=True)
            if flushed:
                text_buffer += flushed
            if text_buffer:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            print(f"[ERROR] graph_infer_stream_state: {exc}")
        finally:
            if static_state is not None:
                self._sync_cuda_graph_state(static_state, state)
            if state_manager and session_id:
                state_manager.put_state(session_id, state)
                print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")

            yield "data: [DONE]\n\n"

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

    def _prefill_prompt_with_prefix_cache(self, prompt, prefix_cache_manager=None):
        encoded_prompt = self.tokenizer.encode(prompt)
        if not encoded_prompt:
            raise ValueError("Empty prompt")

        state = None
        out = None
        matched_tokens = 0
        cache_source = None

        if prefix_cache_manager is not None:
            cache_match = prefix_cache_manager.match_prefix_state(encoded_prompt, device="cuda")
            if cache_match is not None:
                state = cache_match["state"]
                out = cache_match["logits"]
                matched_tokens = int(cache_match["matched_tokens"])
                cache_source = cache_match["cache_source"]

        if state is None:
            state = self.model.generate_zero_state(0)

        if prefix_cache_manager is not None:
            bucket_checkpoints = [
                bucket for bucket in getattr(prefix_cache_manager, "prefix_l2_cache", {}).keys()
                if matched_tokens < bucket <= len(encoded_prompt)
            ]
            bucket_checkpoints.sort()
        else:
            bucket_checkpoints = []

        cursor = matched_tokens
        for checkpoint in bucket_checkpoints:
            segment = encoded_prompt[cursor:checkpoint]
            if segment:
                out = self.model.forward(segment, state).float()
                prefix_cache_manager.put_prefix_state(encoded_prompt[:checkpoint], state, out)
                cursor = checkpoint

        remaining_tokens = encoded_prompt[cursor:]
        if remaining_tokens:
            out = self.model.forward(remaining_tokens, state).float()
        elif out is None:
            # Older cache rows may exist without logits. Fall back to recomputing once.
            del state
            state = self.model.generate_zero_state(0)
            out = self.model.forward(encoded_prompt, state).float()
            matched_tokens = 0
            cache_source = None

        return encoded_prompt, state, out, matched_tokens, cache_source

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
        stop_tokens=("\nUser:",),
    ):
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()
        sampling_state = self._create_sampling_state(batch_size, out.device)

        finished = [False] * batch_size
        stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
        generated_text = [""] * batch_size
        batch_graph = None

        for _ in range(max_length):
            new_tokens_tensor = self._sample_with_repetition(
                out,
                sampling_state,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                temperature,
                top_k,
                top_p,
            ).reshape(-1, 1)
            new_tokens = new_tokens_tensor.view(-1).tolist()

            if batch_graph is None and torch.cuda.is_available():
                batch_graph = self._get_or_create_batch_cuda_graph(
                    batch_size, new_tokens_tensor, state, out
                )
                state = batch_graph["static_state"]
            if batch_graph is not None:
                batch_graph["static_tokens"].copy_(new_tokens_tensor)
                batch_graph["graph"].replay()
                out = batch_graph["static_output"].float()
            else:
                out = self.model.forward_batch(new_tokens_tensor, state).float()

            for i in range(batch_size):
                tok = new_tokens[i]
                if finished[i]:
                    continue

                content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                if content:
                    generated_text[i] += content

                if should_stop:
                    finished[i] = True
                    continue

            if all(finished):
                break

        del state
        gc.collect()

        decoded = []
        for i in range(batch_size):
            generated_text[i] += self._flush_stop_state(stop_states[i], final=True)
            decoded.append(generated_text[i])
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
        stop_tokens=("\nUser:",),
        chunk_size=32,
    ):
        batch_size = len(prompts)
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()

        finished = [False] * batch_size
        stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
        chunk_token_counts = [0] * batch_size
        text_buffers = [""] * batch_size
        sampling_state = self._create_sampling_state(batch_size, out.device)
        batch_graph = None

        try:
            while not all(finished) and max_length > 0:
                new_tokens_tensor = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).reshape(-1, 1)
                new_tokens = new_tokens_tensor.view(-1).tolist()

                if batch_graph is None and torch.cuda.is_available():
                    batch_graph = self._get_or_create_batch_cuda_graph(
                        batch_size, new_tokens_tensor, state, out
                    )
                    state = batch_graph["static_state"]
                if batch_graph is not None:
                    batch_graph["static_tokens"].copy_(new_tokens_tensor)
                    batch_graph["graph"].replay()
                    out = batch_graph["static_output"].float()
                else:
                    out = self.model.forward_batch(new_tokens_tensor, state).float()
                max_length -= 1

                contents_to_send = [""] * batch_size

                for i in range(batch_size):
                    if finished[i]:
                        continue

                    tok = new_tokens[i]

                    content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                    if content:
                        text_buffers[i] += content

                    if should_stop:
                        finished[i] = True
                        flushed = self._flush_stop_state(stop_states[i], final=True)
                        if flushed:
                            text_buffers[i] += flushed
                        if text_buffers[i]:
                            contents_to_send[i] += text_buffers[i]
                            text_buffers[i] = ""
                        continue

                    chunk_token_counts[i] += 1
                    if chunk_token_counts[i] >= chunk_size and text_buffers[i]:
                        contents_to_send[i] += text_buffers[i]
                        text_buffers[i] = ""
                        chunk_token_counts[i] = 0

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
                flushed = self._flush_stop_state(stop_states[i], final=True)
                if flushed:
                    text_buffers[i] += flushed
                remaining_contents[i] = text_buffers[i]

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
        stop_tokens=("\nUser:",),
        prefix_cache_manager=None,
        session_id=None,
    ):
        _, state, out = self._prefill_prompt_from_state_or_prefix(
            prompts[0], state, prefix_cache_manager=prefix_cache_manager
        )
        sampling_state = self._create_sampling_state(1, out.device)

        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        for _ in range(max_length):
            new_tokens = self._sample_with_repetition(
                out,
                sampling_state,
                alpha_presence,
                alpha_frequency,
                alpha_decay,
                temperature,
                top_k,
                top_p,
            ).tolist()

            tok = new_tokens[0]

            content, should_stop = self._ingest_token_with_stop(stop_state, tok)
            if content:
                generated_text += content

            if should_stop:
                break

            out = self.model.forward(tok, state).float()
        generated_text += self._flush_stop_state(stop_state, final=True)
        return [generated_text]

    async def singe_infer(
        self,
        prompt,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        prefix_cache_manager=None,
    ):
        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        finish_reason = "length"

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            sampling_state = self._create_sampling_state(1, out.device)

            while max_length > 0:
                max_length -= 1
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    finish_reason = "stop"
                    break

                out = self.model.forward(tok, state).float()
                await asyncio.sleep(0)

            generated_text += self._flush_stop_state(stop_state, final=True)
            return generated_text, finish_reason
        finally:
            del state
            gc.collect()

    async def singe_infer_stream(
        self,
        prompt,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        prefix_cache_manager=None,
    ):
        finish_reason = "length"

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""
            sampling_state = self._create_sampling_state(1, out.device)

            while max_length > 0:
                max_length -= 1
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    text_buffer += content

                if should_stop:
                    finish_reason = "stop"
                    flushed = self._flush_stop_state(stop_state, final=True)
                    if flushed:
                        text_buffer += flushed
                    if text_buffer:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        text_buffer = ""
                    break

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and text_buffer:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    text_buffer = ""
                    buffered_tokens = 0

                out = self.model.forward(tok, state).float()
                await asyncio.sleep(0)

            flushed = self._flush_stop_state(stop_state, final=True)
            if flushed:
                text_buffer += flushed
            if text_buffer:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            del state
            gc.collect()

        yield "data: [DONE]\n\n"

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
        stop_tokens=("\nUser:",),
        chunk_size=32,
        session_id=None,
        state_manager=None,
        prefix_cache_manager=None,
    ):
        chunk_size = max(1, int(chunk_size))

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompts[0], state, prefix_cache_manager=prefix_cache_manager
            )
            sampling_state = self._create_sampling_state(1, out.device)

            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""

            while max_length > 0:
                max_length -= 1
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]

                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    text_buffer += content

                if should_stop:
                    flushed = self._flush_stop_state(stop_state, final=True)
                    if flushed:
                        text_buffer += flushed
                    if text_buffer:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        text_buffer = ""
                    break

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and text_buffer:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    text_buffer = ""
                    buffered_tokens = 0

                out = self.model.forward(tok, state).float()

                await asyncio.sleep(0)

            flushed = self._flush_stop_state(stop_state, final=True)
            if flushed:
                text_buffer += flushed
            if text_buffer:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            if state_manager and session_id:
                state_manager.put_state(session_id, state)
                print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")

            del state
            gc.collect()

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
        prefix_cache_manager=None,
    ):
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

        decode_capacity = batch_size
        prefill_capacity = self._continuous_prefill_capacity(decode_capacity)
        state_manager = PagedStateManager(
            self.model,
            self.args.vocab_size,
            device,
            decode_capacity,
            prefill_capacity,
        )
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        input_queue = deque(inputs)
        prefill_tasks = []
        decode_tasks = []
        stop_states = {}
        chunk_token_counts = {}
        text_buffers = {}

        def _register_prompt(prompt_id):
            stop_states[prompt_id] = self._create_stop_state(stop_tokens)
            chunk_token_counts[prompt_id] = 0
            text_buffers[prompt_id] = ""

        prompt_idx = 0
        prompt_idx = self._launch_prefill_tasks(
            input_queue,
            state_manager,
            prefill_tasks,
            prompt_idx,
            stop_tokens,
            _register_prompt,
            pad_zero=pad_zero,
            prefix_cache_manager=prefix_cache_manager,
        )

        try:
            while True:
                contents_to_send = {}
                accomplished_task_indices = []

                prompt_idx = self._launch_prefill_tasks(
                    input_queue,
                    state_manager,
                    prefill_tasks,
                    prompt_idx,
                    stop_tokens,
                    _register_prompt,
                    pad_zero=pad_zero,
                    prefix_cache_manager=prefix_cache_manager,
                )

                active_prefill_tasks, prefill_step = self._select_continuous_prefill_group(prefill_tasks)
                if active_prefill_tasks:
                    self._apply_continuous_prefill(active_prefill_tasks, state_manager.states, prefill_step)

                self._promote_ready_prefill_tasks(
                    prefill_tasks,
                    decode_tasks,
                    decode_capacity,
                )

                for task_idx, task in enumerate(decode_tasks):
                    if len(task["input_token"]) == 0:
                        if task["new_token"] is None:
                            if task["pending_out"] is None:
                                continue
                            task["new_token"] = self._sample_top_k_top_p(
                                task["pending_out"], top_k, top_p, temperature
                            ).tolist()[0]
                            task["pending_out"] = None

                        new_token = task["new_token"]
                        prompt_id = task["prompt_idx"]

                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[prompt_id], new_token
                        )

                        if content:
                            text_buffers[prompt_id] += content

                        is_finished = should_stop or (
                            len(task["generated_tokens"]) >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_tokens"].append(new_token)
                            chunk_token_counts[prompt_id] += 1

                            if (
                                chunk_token_counts[prompt_id] >= chunk_size
                                and text_buffers[prompt_id]
                            ):
                                contents_to_send[prompt_id] = (
                                    contents_to_send.get(prompt_id, "")
                                    + text_buffers[prompt_id]
                                )
                                text_buffers[prompt_id] = ""
                                chunk_token_counts[prompt_id] = 0

                        if is_finished:
                            text_chunk = self._flush_stop_state(
                                stop_states[prompt_id], final=True
                            )
                            if text_chunk:
                                text_buffers[prompt_id] += text_chunk
                            if text_buffers[prompt_id]:
                                contents_to_send[prompt_id] = (
                                    contents_to_send.get(prompt_id, "")
                                    + text_buffers[prompt_id]
                                )
                                text_buffers[prompt_id] = ""

                            del stop_states[prompt_id]
                            del chunk_token_counts[prompt_id]
                            del text_buffers[prompt_id]

                            state_manager.release_page(task["state_pos"])
                            accomplished_task_indices.append(task_idx)
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in no_penalty_token_ids else 1.0
                            state_manager.occurrence[task["state_pos"], new_token] += www
                            state_manager.alpha_presence_vector[task["state_pos"], new_token] = (
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
                    self._release_finished_tasks(decode_tasks, accomplished_task_indices)

                if not decode_tasks and not prefill_tasks and not input_queue:
                    break

                decode_ready_tasks = [task for task in decode_tasks if self._ready_for_decode(task)]
                if not decode_ready_tasks:
                    continue

                slot_indices = [task["state_pos"] for task in decode_ready_tasks]
                token_tensor = torch.tensor(
                    [[task["input_token"].pop(0)] for task in decode_ready_tasks],
                    device=device,
                    dtype=torch.long,
                )
                page_table = state_manager.page_table(decode_ready_tasks)
                out = self.model.forward_batch_paged(token_tensor, state_manager.states, page_table)

                decode_occurrence = state_manager.occurrence[slot_indices]
                decode_presence = state_manager.alpha_presence_vector[slot_indices]

                if alpha_presence != 0 or alpha_frequency != 0:
                    mask = (decode_occurrence > 0).float()
                    out -= mask * alpha_presence + decode_occurrence * alpha_frequency

                decode_occurrence *= alpha_decay
                out -= decode_presence + decode_occurrence * alpha_frequency
                state_manager.occurrence[slot_indices] = decode_occurrence

                if temperature != 1.0:
                    out /= temperature

                new_tokens = self._sample_top_k_top_p(out, top_k, top_p).tolist()

                for idx, task in enumerate(decode_ready_tasks):
                    task["new_token"] = new_tokens[idx]

        finally:
            del state_manager
            gc.collect()
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
        prefix_cache_manager=None,
    ):
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

        decode_capacity = batch_size
        prefill_capacity = self._continuous_prefill_capacity(decode_capacity)
        state_manager = PagedStateManager(
            self.model,
            self.args.vocab_size,
            device,
            decode_capacity,
            prefill_capacity,
        )
        no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
        input_queue = deque(inputs)
        prefill_tasks = []
        decode_tasks = []
        results = {}
        stop_states = {}
        result_parts = {}

        def _register_prompt(prompt_id):
            stop_states[prompt_id] = self._create_stop_state(stop_tokens)
            result_parts[prompt_id] = []

        prompt_idx = 0
        prompt_idx = self._launch_prefill_tasks(
            input_queue,
            state_manager,
            prefill_tasks,
            prompt_idx,
            stop_tokens,
            _register_prompt,
            pad_zero=pad_zero,
            prefix_cache_manager=prefix_cache_manager,
        )

        try:
            while True:
                accomplished_task_indices = []

                prompt_idx = self._launch_prefill_tasks(
                    input_queue,
                    state_manager,
                    prefill_tasks,
                    prompt_idx,
                    stop_tokens,
                    _register_prompt,
                    pad_zero=pad_zero,
                    prefix_cache_manager=prefix_cache_manager,
                )

                active_prefill_tasks, prefill_step = self._select_continuous_prefill_group(prefill_tasks)
                if active_prefill_tasks:
                    self._apply_continuous_prefill(active_prefill_tasks, state_manager.states, prefill_step)

                self._promote_ready_prefill_tasks(
                    prefill_tasks,
                    decode_tasks,
                    decode_capacity,
                )

                for task_idx, task in enumerate(decode_tasks):
                    if len(task["input_token"]) == 0:
                        if task["new_token"] is None:
                            if task["pending_out"] is None:
                                continue
                            task["new_token"] = self._sample_top_k_top_p(
                                task["pending_out"], top_k, top_p, temperature
                            ).tolist()[0]
                            task["pending_out"] = None

                        new_token = task["new_token"]
                        prompt_id = task["prompt_idx"]

                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[prompt_id], new_token
                        )
                        if content:
                            result_parts[prompt_id].append(content)
                        is_finished = should_stop or (
                            len(task["generated_tokens"]) >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_tokens"].append(new_token)

                        if is_finished:
                            result_parts[prompt_id].append(
                                self._flush_stop_state(
                                    stop_states[prompt_id], final=True
                                )
                            )
                            results[prompt_id] = "".join(result_parts[prompt_id])
                            del stop_states[prompt_id]
                            del result_parts[prompt_id]

                            state_manager.release_page(task["state_pos"])
                            accomplished_task_indices.append(task_idx)
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in no_penalty_token_ids else 1.0
                            state_manager.occurrence[task["state_pos"], new_token] += www
                            state_manager.alpha_presence_vector[task["state_pos"], new_token] = (
                                alpha_presence_val
                            )

                if accomplished_task_indices:
                    self._release_finished_tasks(decode_tasks, accomplished_task_indices)

                if not decode_tasks and not prefill_tasks and not input_queue:
                    break

                decode_ready_tasks = [task for task in decode_tasks if self._ready_for_decode(task)]
                if not decode_ready_tasks:
                    continue

                slot_indices = [task["state_pos"] for task in decode_ready_tasks]
                token_tensor = torch.tensor(
                    [[task["input_token"].pop(0)] for task in decode_ready_tasks],
                    device=device,
                    dtype=torch.long,
                )
                page_table = state_manager.page_table(decode_ready_tasks)
                out = self.model.forward_batch_paged(token_tensor, state_manager.states, page_table)

                decode_occurrence = state_manager.occurrence[slot_indices]
                decode_presence = state_manager.alpha_presence_vector[slot_indices]

                if alpha_presence != 0 or alpha_frequency != 0:
                    mask = (decode_occurrence > 0).float()
                    out -= mask * alpha_presence + decode_occurrence * alpha_frequency

                decode_occurrence *= alpha_decay
                out -= decode_presence + decode_occurrence * alpha_frequency
                state_manager.occurrence[slot_indices] = decode_occurrence

                if temperature != 1.0:
                    out /= temperature

                new_tokens = self._sample_top_k_top_p(out, top_k, top_p).tolist()

                for idx, task in enumerate(decode_ready_tasks):
                    task["new_token"] = new_tokens[idx]

        finally:
            del state_manager
            gc.collect()

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
        prefix_cache_manager=None,
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
                prefix_cache_manager,
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
        prefix_cache_manager=None,
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
            prefix_cache_manager=prefix_cache_manager,
        )

    async def big_batch_stream(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        stop_tokens=("\nUser:",),
        chunk_size=32,
    ):
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        finished = None
        generated_tokens = None
        token_buffers = None
        new_tokens_tensor = None
        new_tokens = None
        batch_graph = None

        try:
            with torch.inference_mode():
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
                out = self.model.forward_batch(encoded_prompts, state)

                finished = [False] * batch_size
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
                chunk_token_counts = [0] * batch_size
                text_buffers = [""] * batch_size

                while not all(finished) and max_length > 0:
                    new_tokens_tensor = self._sample_throughput_tokens(out, temperature)
                    new_tokens = new_tokens_tensor.tolist()

                    if batch_graph is None and torch.cuda.is_available():
                        batch_graph = self._get_or_create_batch_cuda_graph(
                            batch_size, new_tokens_tensor, state, out
                        )
                        state = batch_graph["static_state"]
                    if batch_graph is not None:
                        batch_graph["static_tokens"].copy_(new_tokens_tensor)
                        batch_graph["graph"].replay()
                        out = batch_graph["static_output"]
                    else:
                        prev_out = out
                        out = self.model.forward_batch(new_tokens_tensor, state)
                        del prev_out

                    del new_tokens_tensor
                    new_tokens_tensor = None

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

                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[i], tok
                        )
                        if content:
                            text_buffers[i] += content

                        if should_stop:
                            finished[i] = True
                            flushed = self._flush_stop_state(
                                stop_states[i], final=True
                            )
                            if flushed:
                                text_buffers[i] += flushed
                            if text_buffers[i]:
                                contents_to_send[i] += text_buffers[i]
                                text_buffers[i] = ""
                            continue

                        chunk_token_counts[i] += 1
                        if chunk_token_counts[i] >= chunk_size and text_buffers[i]:
                            contents_to_send[i] += text_buffers[i]
                            text_buffers[i] = ""
                            chunk_token_counts[i] = 0

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
                            yield (
                                f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            )

                    new_tokens = None
                    await asyncio.sleep(0)

                remaining_contents = [""] * batch_size
                for i in range(batch_size):
                    flushed = self._flush_stop_state(
                        stop_states[i], final=True
                    )
                    if flushed:
                        text_buffers[i] += flushed
                    remaining_contents[i] = text_buffers[i]

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
            if new_tokens_tensor is not None:
                del new_tokens_tensor
            if out is not None:
                del out
            if state is not None:
                del state
            if encoded_prompts is not None:
                del encoded_prompts
            if finished is not None:
                del finished
            if generated_tokens is not None:
                del generated_tokens
            if token_buffers is not None:
                del token_buffers
            if new_tokens is not None:
                del new_tokens
            self._cleanup_cuda_memory()

        yield "data: [DONE]\n\n"

    def big_batch_generate(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        stop_tokens=("\nUser:",),
    ):
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        finished = None
        text_buffers = None
        stop_states = None
        new_tokens_tensor = None
        new_tokens = None
        batch_graph = None

        try:
            with torch.inference_mode():
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
                out = self.model.forward_batch(encoded_prompts, state)

                finished = [False] * batch_size
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
                text_buffers = [""] * batch_size

                while not all(finished) and max_length > 0:
                    new_tokens_tensor = self._sample_throughput_tokens(out, temperature)
                    new_tokens = new_tokens_tensor.tolist()

                    if batch_graph is None and torch.cuda.is_available():
                        batch_graph = self._get_or_create_batch_cuda_graph(
                            batch_size, new_tokens_tensor, state, out
                        )
                        state = batch_graph["static_state"]
                    if batch_graph is not None:
                        batch_graph["static_tokens"].copy_(new_tokens_tensor)
                        batch_graph["graph"].replay()
                        out = batch_graph["static_output"]
                    else:
                        prev_out = out
                        out = self.model.forward_batch(new_tokens_tensor, state)
                        del prev_out

                    del new_tokens_tensor
                    new_tokens_tensor = None

                    max_length -= 1

                    for i in range(batch_size):
                        if finished[i]:
                            continue

                        tok = (
                            new_tokens[i][0]
                            if isinstance(new_tokens[i], list)
                            else new_tokens[i]
                        )
                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[i], tok
                        )
                        if content:
                            text_buffers[i] += content

                        if should_stop:
                            finished[i] = True

                for i in range(batch_size):
                    text_buffers[i] += self._flush_stop_state(stop_states[i], final=True)

                return text_buffers
        finally:
            if new_tokens_tensor is not None:
                del new_tokens_tensor
            if out is not None:
                del out
            if state is not None:
                del state
            if encoded_prompts is not None:
                del encoded_prompts
            if finished is not None:
                del finished
            if text_buffers is not None:
                del text_buffers
            if stop_states is not None:
                del stop_states
            if new_tokens is not None:
                del new_tokens
            self._cleanup_cuda_memory()

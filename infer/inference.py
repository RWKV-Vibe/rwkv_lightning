import asyncio
import gc
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from queue import Empty, Queue
from threading import Event, Lock, Thread

import torch

from infer.rwkv_batch.sampler import sample
from infer.rwkv_batch.utils import sampler_gumbel_batch
from infer.xgrammar_utils import XGrammarRuntime


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
        self.page_table_buffer = torch.empty(
            (self.total_pages,), dtype=torch.int32, device=device
        )
        self.page_index_buffer = torch.empty(
            (self.total_pages,), dtype=torch.long, device=device
        )
        self.decode_token_buffer = torch.empty(
            (max(1, decode_capacity), 1), dtype=torch.long, device=device
        )
        self.prefill_token_buffer = torch.empty(
            (max(1, prefill_capacity), 512), dtype=torch.long, device=device
        )

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

    def _ensure_prefill_buffer(self, step_len):
        if step_len <= self.prefill_token_buffer.shape[1]:
            return
        self.prefill_token_buffer = torch.empty(
            (max(1, self.prefill_capacity), step_len),
            dtype=torch.long,
            device=self.device,
        )

    def build_page_table(self, tasks):
        count = len(tasks)
        for idx, task in enumerate(tasks):
            state_pos = int(task["state_pos"])
            self.page_table_buffer[idx] = state_pos
            self.page_index_buffer[idx] = state_pos
        return self.page_table_buffer[:count], self.page_index_buffer[:count]

    def prepare_decode_batch(self, tasks):
        count = len(tasks)
        for idx, task in enumerate(tasks):
            state_pos = int(task["state_pos"])
            self.page_table_buffer[idx] = state_pos
            self.page_index_buffer[idx] = state_pos
            self.decode_token_buffer[idx, 0] = task["input_token"].popleft()
        return (
            self.decode_token_buffer[:count],
            self.page_table_buffer[:count],
            self.page_index_buffer[:count],
        )

    def prepare_prefill_batch(self, tasks, step_len):
        self._ensure_prefill_buffer(step_len)
        count = len(tasks)
        batch_tokens = self.prefill_token_buffer[:count, :step_len]
        for idx, task in enumerate(tasks):
            state_pos = int(task["state_pos"])
            self.page_table_buffer[idx] = state_pos
            self.page_index_buffer[idx] = state_pos
            for token_idx, token in enumerate(islice(task["input_token"], 0, step_len)):
                batch_tokens[idx, token_idx] = token
        return batch_tokens, self.page_table_buffer[:count], self.page_index_buffer[:count]

    @staticmethod
    def consume_prefill_tokens(tasks, step_len):
        for task in tasks:
            for _ in range(step_len):
                task["input_token"].popleft()


class InferenceEngine:
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.enable_cuda_graphs = torch.cuda.is_available() and os.getenv(
            "RWKV_USE_CUDA_GRAPHS", "0"
        ) == "1"
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="model_inference"
        )
        self._cuda_graph_sessions = {}
        self._batch_cuda_graphs = {}
        self._continuous_prefill_candidates = (256, 128, 64, 32, 16, 8, 4, 2)
        self._no_penalty_token_ids = frozenset(
            (33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58)
        )
        self._token_bytes_cache = {}
        self._token_text_cache = {}
        self._xgrammar_runtime = None

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
            "pending_bytes": bytearray(),
            "pending_lengths": deque(),
            "window_size": 6,
        }

    def _ingest_token_with_stop(self, stop_state, token):
        if token == 0:
            return "", True

        pending_bytes = stop_state["pending_bytes"]
        pending_lengths = stop_state["pending_lengths"]
        token_bytes = self._token_raw_bytes(token)
        pending_bytes.extend(token_bytes)
        pending_lengths.append(len(token_bytes))

        decoded_window = bytes(pending_bytes).decode("utf-8", errors="ignore")
        matched_stop = self._match_stop_suffix(
            decoded_window, stop_state["stop_strings"]
        )
        if matched_stop is not None:
            pending_bytes.clear()
            pending_lengths.clear()
            return decoded_window[: -len(matched_stop)], True

        if len(pending_lengths) > stop_state["window_size"]:
            del pending_bytes[: pending_lengths.popleft()]
            trailing_text = bytes(pending_bytes).decode("utf-8", errors="ignore")
            if not trailing_text:
                return decoded_window, False
            if decoded_window.endswith(trailing_text):
                return decoded_window[: -len(trailing_text)], False

            return token_bytes.decode("utf-8", errors="ignore"), False

        return "", False

    def _flush_stop_state(self, stop_state, final=False):
        pending_bytes = stop_state["pending_bytes"]
        pending_lengths = stop_state["pending_lengths"]
        if not pending_lengths:
            return ""

        if not final and len(pending_lengths) <= stop_state["window_size"]:
            return ""

        decoded = bytes(pending_bytes).decode("utf-8", errors="ignore")
        pending_bytes.clear()
        pending_lengths.clear()
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

    def _create_sampling_state(self, batch_size, device, vocab_size=None, seed=None):
        state_seed = int(seed) if seed is not None else random.randint(0, 2**63 - 1)
        return {
            "penalties": torch.zeros(
                (batch_size, vocab_size or self.args.vocab_size),
                dtype=torch.float32,
                device=device,
            ),
            "rand_states": self._create_rand_states(batch_size, seed=state_seed),
            "seed": state_seed,
        }

    def _create_rand_states(self, batch_size, seed=None):
        return sample.setup_rand(
            int(seed) if seed is not None else random.randint(0, 2**63 - 1),
            batch_size,
        )

    @staticmethod
    def _maybe_contiguous(tensor):
        contiguous = getattr(tensor, "contiguous", None)
        return contiguous() if callable(contiguous) else tensor

    def _get_xgrammar_runtime(self):
        if self._xgrammar_runtime is None:
            self._xgrammar_runtime = XGrammarRuntime(self.tokenizer, self.args.vocab_size)
        return self._xgrammar_runtime

    def build_response_format_constraint(self, response_format):
        if not response_format:
            return None
        return self._get_xgrammar_runtime().build_constraint(response_format)

    @staticmethod
    def _normalize_grammar_constraints(grammar_constraints):
        if grammar_constraints is None:
            return []
        if isinstance(grammar_constraints, (list, tuple)):
            return [constraint for constraint in grammar_constraints if constraint is not None]
        return [grammar_constraints]

    def _apply_grammar_constraints_inplace(self, logits, grammar_constraints=None):
        constraints = self._normalize_grammar_constraints(grammar_constraints)
        if not constraints:
            return

        logits_2d = logits if logits.dim() == 2 else logits.unsqueeze(0)
        if len(constraints) == 1 and logits_2d.shape[0] == 1:
            constraints[0].apply(logits_2d)
            return

        if len(constraints) != logits_2d.shape[0]:
            raise ValueError(
                f"Grammar constraint count {len(constraints)} does not match batch size {logits_2d.shape[0]}"
            )

        for row_idx, constraint in enumerate(constraints):
            if constraint is not None:
                constraint.apply(logits_2d[row_idx])

    @staticmethod
    def _apply_logit_bias_inplace(logits, logit_bias=None):
        if not logit_bias:
            return
        logits_2d = logits if logits.dim() == 2 else logits.unsqueeze(0)
        for token_id, bias in logit_bias.items():
            logits_2d[:, int(token_id)] += float(bias)

    def _accept_grammar_tokens(self, grammar_constraints, token_ids):
        constraints = self._normalize_grammar_constraints(grammar_constraints)
        if not constraints:
            return

        if isinstance(token_ids, torch.Tensor):
            token_values = token_ids.view(-1).tolist()
        elif isinstance(token_ids, (list, tuple)):
            token_values = list(token_ids)
        else:
            token_values = [token_ids]

        if len(constraints) != len(token_values):
            raise ValueError(
                f"Grammar constraint count {len(constraints)} does not match token count {len(token_values)}"
            )

        for constraint, token_id in zip(constraints, token_values):
            if constraint is not None and not constraint.accept_token(int(token_id)):
                raise RuntimeError(f"Rejected token {token_id} by grammar matcher")

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
        grammar_constraints=None,
        logit_bias=None,
    ):
        logits_reshaped = self._maybe_contiguous(
            logits.unsqueeze(0).float() if logits.dim() == 1 else logits.float()
        )
        self._apply_logit_bias_inplace(logits_reshaped, logit_bias)
        self._apply_grammar_constraints_inplace(logits_reshaped, grammar_constraints)
        if temperature <= 0:
            temperature = 1.0
            top_k = 1
            top_p = 1.0
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

    def _sample_top_k_top_p(
        self,
        logits,
        top_k,
        top_p,
        temperature=1.0,
        grammar_constraints=None,
        rand_states=None,
        logit_bias=None,
    ):
        logits_reshaped = self._maybe_contiguous(
            logits.unsqueeze(0).float() if logits.dim() == 1 else logits.float()
        )
        self._apply_logit_bias_inplace(logits_reshaped, logit_bias)
        self._apply_grammar_constraints_inplace(logits_reshaped, grammar_constraints)
        if temperature <= 0:
            temperature = 1.0
            top_k = 1
            top_p = 1.0

        if rand_states is None:
            rand_states = self._create_rand_states(logits_reshaped.shape[0])
        elif rand_states.size(0) < logits_reshaped.shape[0]:
            raise ValueError(
                f"Random state batch size {rand_states.size(0)} is smaller than logits batch size {logits_reshaped.shape[0]}"
            )

        return sample.batch_sampling_temperature_topk_topp(
            logits_reshaped,
            rand_states,
            temperature,
            top_k,
            top_p,
        )

    def _sample_throughput_tokens(
        self, logits, temperature, grammar_constraints=None, logit_bias=None
    ):
        logits_reshaped = logits.unsqueeze(0) if logits.dim() == 1 else logits
        self._apply_logit_bias_inplace(logits_reshaped, logit_bias)
        self._apply_grammar_constraints_inplace(logits_reshaped, grammar_constraints)
        if temperature <= 0:
            return torch.argmax(logits_reshaped, dim=-1, keepdim=True)
        return sampler_gumbel_batch(logits=logits_reshaped, temp=temperature)

    @staticmethod
    def _filter_logits_top_k_top_p(logits, top_k, top_p):
        filtered = logits.clone()
        if top_k > 0:
            top_k = min(top_k, filtered.size(-1))
            threshold = torch.topk(filtered, top_k, dim=-1)[0][..., -1, None]
            filtered = filtered.masked_fill(filtered < threshold, -float("Inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., :1] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove,
            )
            filtered = filtered.masked_fill(indices_to_remove, -float("Inf"))
        return filtered

    @staticmethod
    def _safe_float(value):
        if torch.isinf(value):
            return float("-inf") if value < 0 else float("inf")
        return float(value.item())

    def _token_raw_bytes(self, token_id):
        token_id = int(token_id)
        cached = self._token_bytes_cache.get(token_id)
        if cached is not None:
            return cached

        raw = getattr(self.tokenizer, "idx2token", {}).get(token_id)
        if isinstance(raw, bytes):
            payload = raw
        elif isinstance(raw, str):
            payload = raw.encode("utf-8")
        else:
            try:
                payload = self.tokenizer.decode([token_id], utf8_errors="ignore").encode(
                    "utf-8"
                )
            except Exception:
                payload = b""

        self._token_bytes_cache[token_id] = payload
        return payload

    def _token_bytes(self, token_id):
        return list(self._token_raw_bytes(token_id))

    def _token_text(self, token_id):
        token_id = int(token_id)
        cached = self._token_text_cache.get(token_id)
        if cached is not None:
            return cached

        raw = getattr(self.tokenizer, "idx2token", {}).get(token_id)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="ignore")
        elif isinstance(raw, str):
            text = raw
        else:
            try:
                text = self.tokenizer.decode([token_id], utf8_errors="ignore")
            except Exception:
                text = ""

        self._token_text_cache[token_id] = text
        return text

    def _prepare_logprob_logits(
        self,
        logits,
        sampling_state,
        alpha_presence,
        alpha_frequency,
        temperature,
        top_k,
        top_p,
        grammar_constraint=None,
        logit_bias=None,
    ):
        logits_2d = logits.unsqueeze(0).float() if logits.dim() == 1 else logits.float()
        adjusted = logits_2d.clone()
        self._apply_logit_bias_inplace(adjusted, logit_bias)
        penalties = sampling_state["penalties"]
        if penalties.shape[0] == adjusted.shape[0]:
            penalty_mask = (penalties > 0).float()
            adjusted = adjusted - penalty_mask * alpha_presence - penalties * alpha_frequency
        if temperature != 1.0:
            adjusted = adjusted / temperature
        self._apply_grammar_constraints_inplace(adjusted, grammar_constraint)
        return self._filter_logits_top_k_top_p(adjusted, top_k, top_p)

    def _build_logprob_entry(self, filtered_logits, token_id, top_logprobs):
        log_probs = torch.log_softmax(filtered_logits, dim=-1)
        token_id = int(token_id)
        entry = {
            "token": self._token_text(token_id),
            "logprob": self._safe_float(log_probs[token_id]),
            "bytes": self._token_bytes(token_id),
            "top_logprobs": [],
        }

        if top_logprobs > 0:
            top_vals, top_ids = torch.topk(log_probs, min(top_logprobs, log_probs.shape[-1]))
            entry["top_logprobs"] = [
                {
                    "token": self._token_text(idx),
                    "logprob": self._safe_float(val),
                    "bytes": self._token_bytes(idx),
                }
                for val, idx in zip(top_vals, top_ids.tolist())
            ]
        return entry

    def _create_logprob_stop_state(self, stop_tokens):
        return {
            "stop_strings": self._normalize_stop_strings(stop_tokens),
            "pending_bytes": bytearray(),
            "pending_lengths": deque(),
            "pending_entries": deque(),
            "window_size": 6,
        }

    def _ingest_token_with_logprobs(self, stop_state, token, logprob_entry):
        if token == 0:
            return "", [], True

        pending_bytes = stop_state["pending_bytes"]
        pending_lengths = stop_state["pending_lengths"]
        pending_entries = stop_state["pending_entries"]
        token_bytes = self._token_raw_bytes(token)
        pending_bytes.extend(token_bytes)
        pending_lengths.append(len(token_bytes))
        pending_entries.append(logprob_entry)

        decoded_window = bytes(pending_bytes).decode("utf-8", errors="ignore")
        matched_stop = self._match_stop_suffix(
            decoded_window, stop_state["stop_strings"]
        )
        if matched_stop is not None:
            content = decoded_window[: -len(matched_stop)]
            pending_bytes.clear()
            pending_lengths.clear()
            # We cannot precisely map trailing stop-string bytes back to token boundaries here.
            # Drop buffered logprobs on the matched stop boundary rather than returning wrong metadata.
            pending_entries.clear()
            return content, [], True

        if len(pending_lengths) > stop_state["window_size"]:
            oldest_entry = pending_entries.popleft()
            del pending_bytes[: pending_lengths.popleft()]
            trailing_text = bytes(pending_bytes).decode("utf-8", errors="ignore")
            if not trailing_text:
                return decoded_window, [oldest_entry], False
            if decoded_window.endswith(trailing_text):
                return decoded_window[: -len(trailing_text)], [oldest_entry], False

            return token_bytes.decode("utf-8", errors="ignore"), [logprob_entry], False

        return "", [], False

    def _flush_logprob_stop_state(self, stop_state, final=False):
        pending_bytes = stop_state["pending_bytes"]
        pending_lengths = stop_state["pending_lengths"]
        pending_entries = stop_state["pending_entries"]
        if not pending_lengths:
            return "", []

        if not final and len(pending_lengths) <= stop_state["window_size"]:
            return "", []

        decoded = bytes(pending_bytes).decode("utf-8", errors="ignore")
        entries = list(pending_entries)
        pending_bytes.clear()
        pending_lengths.clear()
        pending_entries.clear()
        return decoded, entries

    def _select_continuous_prefill_group(self, task_pool):
        prefill_tasks = [
            task
            for task in task_pool
            if task["pending_out"] is None
            and task["new_token"] is None
            and task["generated_count"] == 0
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

    def _apply_continuous_prefill(self, tasks, state_manager, step_len):
        if not tasks or step_len <= 0:
            return

        batch_tokens, page_table, _ = state_manager.prepare_prefill_batch(tasks, step_len)
        out = self.model.forward_batch_paged(batch_tokens, state_manager.states, page_table)
        state_manager.consume_prefill_tokens(tasks, step_len)

        for idx, task in enumerate(tasks):
            if not task["input_token"]:
                task["pending_out"] = out[idx]

    @staticmethod
    def _ready_for_decode(task):
        if len(task["input_token"]) == 0:
            return False
        if len(task["input_token"]) == 1:
            return True
        return bool(task["generated_count"] or task["new_token"] is not None)

    def _prepare_continuous_decode_batch(self, decode_tasks, state_manager):
        ready_tasks = [task for task in decode_tasks if self._ready_for_decode(task)]
        if not ready_tasks:
            return [], None, None, None
        token_tensor, page_table, page_index = state_manager.prepare_decode_batch(
            ready_tasks
        )
        return ready_tasks, token_tensor, page_table, page_index

    def _apply_continuous_penalties(
        self,
        out,
        state_manager,
        page_index,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
    ):
        decode_occurrence = torch.index_select(
            state_manager.occurrence, 0, page_index
        )
        decode_presence = torch.index_select(
            state_manager.alpha_presence_vector, 0, page_index
        )

        if alpha_presence != 0 or alpha_frequency != 0:
            mask = (decode_occurrence > 0).float()
            out -= mask * alpha_presence + decode_occurrence * alpha_frequency

        decode_occurrence *= alpha_decay
        out -= decode_presence + decode_occurrence * alpha_frequency
        state_manager.occurrence.index_copy_(0, page_index, decode_occurrence)
        return out

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
    def _row_index_tensor(row_indices, device):
        if torch.is_tensor(row_indices):
            return row_indices.to(device=device, dtype=torch.long)
        return torch.tensor(row_indices, device=device, dtype=torch.long)

    def _select_rows(self, tensor, row_indices):
        if tensor is None:
            return None
        row_index = self._row_index_tensor(row_indices, tensor.device)
        return torch.index_select(tensor, 0, row_index).contiguous()

    def _select_state_rows(self, state, row_indices):
        row_index = self._row_index_tensor(row_indices, state[2].device)
        return [
            torch.index_select(state[0], 2, row_index).contiguous(),
            torch.index_select(state[1], 1, row_index).contiguous(),
            torch.index_select(state[2], 0, row_index).contiguous(),
        ]

    def _select_sampling_state_rows(self, sampling_state, row_indices):
        penalties = self._select_rows(sampling_state["penalties"], row_indices)
        seed = sampling_state.get("seed")
        return {
            "penalties": penalties,
            "rand_states": self._create_rand_states(penalties.shape[0], seed=seed),
            "seed": seed,
        }

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
            "input_token": deque(prompt_tokens),
            "state_pos": state_pos,
            "generated_count": 0,
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

    @staticmethod
    def _snapshot_paged_state_rows(states, page_index):
        return [
            torch.index_select(states[0], 2, page_index).contiguous(),
            torch.index_select(states[1], 1, page_index).contiguous(),
            torch.index_select(states[2], 0, page_index).contiguous(),
        ]

    @staticmethod
    def _restore_paged_state_rows(states, page_index, snapshot):
        states[0].index_copy_(2, page_index, snapshot[0])
        states[1].index_copy_(1, page_index, snapshot[1])
        states[2].index_copy_(0, page_index, snapshot[2])

    def _get_or_create_paged_batch_cuda_graph(
        self, graph_cache, batch_size, token_tensor, states, page_table, page_index
    ):
        if batch_size in graph_cache:
            sess = graph_cache[batch_size]
            sess["static_tokens"].copy_(token_tensor)
            sess["static_page_table"].copy_(page_table)
            return sess

        static_tokens = torch.empty_like(token_tensor, device=token_tensor.device)
        static_page_table = torch.empty_like(page_table, device=page_table.device)
        static_tokens.copy_(token_tensor)
        static_page_table.copy_(page_table)

        snapshot = self._snapshot_paged_state_rows(states, page_index)
        static_output = self.model.forward_batch_paged(
            static_tokens, states, static_page_table
        )

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.model.forward_batch_paged(
                static_tokens, states, static_page_table
            )

        self._restore_paged_state_rows(states, page_index, snapshot)

        sess = {
            "static_tokens": static_tokens,
            "static_page_table": static_page_table,
            "static_output": static_output,
            "graph": g,
        }
        graph_cache[batch_size] = sess
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
    def _cleanup_paged_batch_cuda_graphs(graph_cache):
        for batch_size in list(graph_cache.keys()):
            sess = graph_cache.pop(batch_size, None)
            if sess is None:
                continue
            del sess["graph"]
            del sess["static_output"]
            del sess["static_page_table"]
            del sess["static_tokens"]

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
        if not self.enable_cuda_graphs:
            return self.batch_generate_state(
                prompts=prompts,
                state=state,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_presence=alpha_presence,
                alpha_frequency=alpha_frequency,
                alpha_decay=alpha_decay,
                stop_tokens=stop_tokens,
                prefix_cache_manager=prefix_cache_manager,
                session_id=session_id,
            )

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
        if not self.enable_cuda_graphs:
            async for chunk in self.batch_infer_stream_state(
                prompts=prompts,
                state=state,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_presence=alpha_presence,
                alpha_frequency=alpha_frequency,
                alpha_decay=alpha_decay,
                stop_tokens=stop_tokens,
                chunk_size=chunk_size,
                session_id=session_id,
                state_manager=state_manager,
                prefix_cache_manager=prefix_cache_manager,
            ):
                yield chunk
            return

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
        total_batch_size = len(prompts)
        batch_size = total_batch_size
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()
        sampling_state = self._create_sampling_state(batch_size, out.device)

        active_indices = list(range(total_batch_size))
        stop_states = [self._create_stop_state(stop_tokens) for _ in range(total_batch_size)]
        generated_text = [""] * total_batch_size
        batch_graph = None

        for _ in range(max_length):
            if not active_indices:
                break
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

            if batch_graph is None and self.enable_cuda_graphs:
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

            surviving_local = []
            surviving_indices = []
            for local_idx, prompt_idx in enumerate(active_indices):
                tok = new_tokens[local_idx]

                content, should_stop = self._ingest_token_with_stop(
                    stop_states[prompt_idx], tok
                )
                if content:
                    generated_text[prompt_idx] += content

                if should_stop:
                    continue

                surviving_local.append(local_idx)
                surviving_indices.append(prompt_idx)

            if not surviving_indices:
                break

            if len(surviving_indices) != batch_size:
                state = self._select_state_rows(state, surviving_local)
                out = self._select_rows(out, surviving_local)
                sampling_state = self._select_sampling_state_rows(
                    sampling_state, surviving_local
                )
                batch_size = len(surviving_indices)
                batch_graph = None

            active_indices = surviving_indices

        del state
        gc.collect()

        decoded = []
        for i in range(total_batch_size):
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
        total_batch_size = len(prompts)
        batch_size = total_batch_size
        state = self.model.generate_zero_state(batch_size)
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        out = self.model.forward_batch(encoded_prompts, state).float()

        active_indices = list(range(total_batch_size))
        stop_states = [self._create_stop_state(stop_tokens) for _ in range(total_batch_size)]
        chunk_token_counts = [0] * total_batch_size
        text_buffers = [""] * total_batch_size
        sampling_state = self._create_sampling_state(batch_size, out.device)
        batch_graph = None

        try:
            while active_indices and max_length > 0:
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

                if batch_graph is None and self.enable_cuda_graphs:
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

                contents_to_send = {}
                surviving_local = []
                surviving_indices = []

                for local_idx, prompt_idx in enumerate(active_indices):
                    tok = new_tokens[local_idx]

                    content, should_stop = self._ingest_token_with_stop(
                        stop_states[prompt_idx], tok
                    )
                    if content:
                        text_buffers[prompt_idx] += content

                    if should_stop:
                        flushed = self._flush_stop_state(
                            stop_states[prompt_idx], final=True
                        )
                        if flushed:
                            text_buffers[prompt_idx] += flushed
                        if text_buffers[prompt_idx]:
                            contents_to_send[prompt_idx] = (
                                contents_to_send.get(prompt_idx, "")
                                + text_buffers[prompt_idx]
                            )
                            text_buffers[prompt_idx] = ""
                        continue

                    surviving_local.append(local_idx)
                    surviving_indices.append(prompt_idx)
                    chunk_token_counts[prompt_idx] += 1
                    if (
                        chunk_token_counts[prompt_idx] >= chunk_size
                        and text_buffers[prompt_idx]
                    ):
                        contents_to_send[prompt_idx] = (
                            contents_to_send.get(prompt_idx, "")
                            + text_buffers[prompt_idx]
                        )
                        text_buffers[prompt_idx] = ""
                        chunk_token_counts[prompt_idx] = 0

                if contents_to_send:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": i, "delta": {"content": content}}
                            for i, content in sorted(contents_to_send.items())
                            if content
                        ],
                    }
                    if chunk["choices"]:
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                if not surviving_indices:
                    break

                if len(surviving_indices) != batch_size:
                    state = self._select_state_rows(state, surviving_local)
                    out = self._select_rows(out, surviving_local)
                    sampling_state = self._select_sampling_state_rows(
                        sampling_state, surviving_local
                    )
                    batch_size = len(surviving_indices)
                    batch_graph = None

                active_indices = surviving_indices

                await asyncio.sleep(0)

            remaining_contents = [""] * total_batch_size
            for i in range(total_batch_size):
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
        seed=None,
        grammar_constraint=None,
        logit_bias=None,
    ):
        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        finish_reason = "length"

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            sampling_state = self._create_sampling_state(1, out.device, seed=seed)

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
                    grammar_constraints=grammar_constraint,
                    logit_bias=logit_bias,
                ).tolist()

                tok = new_tokens[0]
                self._accept_grammar_tokens(grammar_constraint, tok)
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

    async def singe_infer_with_metadata(
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
        seed=None,
        grammar_constraint=None,
        top_logprobs=0,
        logit_bias=None,
    ):
        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        finish_reason = "length"
        logprob_entries = []

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            sampling_state = self._create_sampling_state(1, out.device, seed=seed)

            while max_length > 0:
                max_length -= 1
                filtered_logits = self._prepare_logprob_logits(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    temperature,
                    top_k,
                    top_p,
                    grammar_constraint,
                    logit_bias=logit_bias,
                )
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                    grammar_constraints=grammar_constraint,
                    logit_bias=logit_bias,
                ).tolist()

                tok = new_tokens[0]
                self._accept_grammar_tokens(grammar_constraint, tok)
                if tok != 0:
                    logprob_entries.append(
                        self._build_logprob_entry(filtered_logits[0], tok, top_logprobs)
                    )
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    finish_reason = "stop"
                    break

                out = self.model.forward(tok, state).float()
                await asyncio.sleep(0)

            generated_text += self._flush_stop_state(stop_state, final=True)
            return generated_text, finish_reason, logprob_entries
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
        seed=None,
        grammar_constraint=None,
        logit_bias=None,
    ):
        finish_reason = "length"

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""
            sampling_state = self._create_sampling_state(1, out.device, seed=seed)

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
                    grammar_constraints=grammar_constraint,
                    logit_bias=logit_bias,
                ).tolist()

                tok = new_tokens[0]
                self._accept_grammar_tokens(grammar_constraint, tok)
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

    async def singe_infer_stream_with_metadata(
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
        seed=None,
        grammar_constraint=None,
        top_logprobs=0,
        logit_bias=None,
    ):
        finish_reason = "length"

        try:
            _, state, out = self._prefill_prompt_from_state_or_prefix(
                prompt, prefix_cache_manager=prefix_cache_manager
            )
            stop_state = self._create_logprob_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""
            logprob_buffer = []
            sampling_state = self._create_sampling_state(1, out.device, seed=seed)

            while max_length > 0:
                max_length -= 1
                filtered_logits = self._prepare_logprob_logits(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    temperature,
                    top_k,
                    top_p,
                    grammar_constraint,
                    logit_bias=logit_bias,
                )
                new_tokens = self._sample_with_repetition(
                    out,
                    sampling_state,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                    grammar_constraints=grammar_constraint,
                    logit_bias=logit_bias,
                ).tolist()

                tok = new_tokens[0]
                self._accept_grammar_tokens(grammar_constraint, tok)
                logprob_entry = (
                    self._build_logprob_entry(filtered_logits[0], tok, top_logprobs)
                    if tok != 0
                    else None
                )
                content, new_entries, should_stop = self._ingest_token_with_logprobs(
                    stop_state, tok, logprob_entry
                )
                if content:
                    text_buffer += content
                if new_entries:
                    logprob_buffer.extend(new_entries)

                if should_stop:
                    finish_reason = "stop"
                    flushed_text, flushed_entries = self._flush_logprob_stop_state(
                        stop_state, final=True
                    )
                    if flushed_text:
                        text_buffer += flushed_text
                    if flushed_entries:
                        logprob_buffer.extend(flushed_entries)
                    if text_buffer or logprob_buffer:
                        yield {
                            "type": "delta",
                            "text": text_buffer,
                            "logprobs": logprob_buffer,
                        }
                        text_buffer = ""
                        logprob_buffer = []
                    break

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and (text_buffer or logprob_buffer):
                    yield {
                        "type": "delta",
                        "text": text_buffer,
                        "logprobs": logprob_buffer,
                    }
                    text_buffer = ""
                    logprob_buffer = []
                    buffered_tokens = 0

                out = self.model.forward(tok, state).float()
                await asyncio.sleep(0)

            flushed_text, flushed_entries = self._flush_logprob_stop_state(
                stop_state, final=True
            )
            if flushed_text:
                text_buffer += flushed_text
            if flushed_entries:
                logprob_buffer.extend(flushed_entries)
            if text_buffer or logprob_buffer:
                yield {
                    "type": "delta",
                    "text": text_buffer,
                    "logprobs": logprob_buffer,
                }

            yield {"type": "done", "finish_reason": finish_reason}
        finally:
            del state
            gc.collect()

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
        total_inputs = len(inputs)

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
        topk_rand_states = self._create_rand_states(decode_capacity)
        paged_decode_graphs = {}
        input_queue = deque(inputs)
        prefill_tasks = []
        decode_tasks = []
        stop_states = [None] * total_inputs
        chunk_token_counts = [0] * total_inputs
        text_buffers = [""] * total_inputs

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
                    self._apply_continuous_prefill(active_prefill_tasks, state_manager, prefill_step)

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
                            task["generated_count"] >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_count"] += 1
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

                            stop_states[prompt_id] = None
                            chunk_token_counts[prompt_id] = 0
                            text_buffers[prompt_id] = ""

                            state_manager.release_page(task["state_pos"])
                            accomplished_task_indices.append(task_idx)
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in self._no_penalty_token_ids else 1.0
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

                decode_ready_tasks, token_tensor, page_table, page_index = (
                    self._prepare_continuous_decode_batch(decode_tasks, state_manager)
                )
                if not decode_ready_tasks:
                    continue

                if self.enable_cuda_graphs and token_tensor.device.type == "cuda":
                    batch_graph = self._get_or_create_paged_batch_cuda_graph(
                        paged_decode_graphs,
                        token_tensor.shape[0],
                        token_tensor,
                        state_manager.states,
                        page_table,
                        page_index,
                    )
                    batch_graph["graph"].replay()
                    out = batch_graph["static_output"].float()
                else:
                    out = self.model.forward_batch_paged(
                        token_tensor, state_manager.states, page_table
                    )
                out = self._apply_continuous_penalties(
                    out,
                    state_manager,
                    page_index,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                )

                if temperature != 1.0:
                    out /= temperature

                new_tokens = self._sample_top_k_top_p(
                    out,
                    top_k,
                    top_p,
                    rand_states=topk_rand_states,
                ).tolist()

                for idx, task in enumerate(decode_ready_tasks):
                    task["new_token"] = new_tokens[idx]

        finally:
            self._cleanup_paged_batch_cuda_graphs(paged_decode_graphs)
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
        total_inputs = len(inputs)

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
        topk_rand_states = self._create_rand_states(decode_capacity)
        paged_decode_graphs = {}
        input_queue = deque(inputs)
        prefill_tasks = []
        decode_tasks = []
        results = [""] * total_inputs
        stop_states = [None] * total_inputs
        result_parts = [[] for _ in range(total_inputs)]

        def _register_prompt(prompt_id):
            stop_states[prompt_id] = self._create_stop_state(stop_tokens)
            result_parts[prompt_id].clear()

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
                    self._apply_continuous_prefill(active_prefill_tasks, state_manager, prefill_step)

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
                            task["generated_count"] >= max_generate_tokens
                        )

                        if not is_finished:
                            task["generated_count"] += 1

                        if is_finished:
                            result_parts[prompt_id].append(
                                self._flush_stop_state(
                                    stop_states[prompt_id], final=True
                                )
                            )
                            results[prompt_id] = "".join(result_parts[prompt_id])
                            stop_states[prompt_id] = None
                            result_parts[prompt_id] = []

                            state_manager.release_page(task["state_pos"])
                            accomplished_task_indices.append(task_idx)
                        else:
                            task["input_token"].append(new_token)
                            www = 0.0 if new_token in self._no_penalty_token_ids else 1.0
                            state_manager.occurrence[task["state_pos"], new_token] += www
                            state_manager.alpha_presence_vector[task["state_pos"], new_token] = (
                                alpha_presence_val
                            )

                if accomplished_task_indices:
                    self._release_finished_tasks(decode_tasks, accomplished_task_indices)

                if not decode_tasks and not prefill_tasks and not input_queue:
                    break

                decode_ready_tasks, token_tensor, page_table, page_index = (
                    self._prepare_continuous_decode_batch(decode_tasks, state_manager)
                )
                if not decode_ready_tasks:
                    continue

                if self.enable_cuda_graphs and token_tensor.device.type == "cuda":
                    batch_graph = self._get_or_create_paged_batch_cuda_graph(
                        paged_decode_graphs,
                        token_tensor.shape[0],
                        token_tensor,
                        state_manager.states,
                        page_table,
                        page_index,
                    )
                    batch_graph["graph"].replay()
                    out = batch_graph["static_output"].float()
                else:
                    out = self.model.forward_batch_paged(
                        token_tensor, state_manager.states, page_table
                    )
                out = self._apply_continuous_penalties(
                    out,
                    state_manager,
                    page_index,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                )

                if temperature != 1.0:
                    out /= temperature

                new_tokens = self._sample_top_k_top_p(
                    out,
                    top_k,
                    top_p,
                    rand_states=topk_rand_states,
                ).tolist()

                for idx, task in enumerate(decode_ready_tasks):
                    task["new_token"] = new_tokens[idx]

        finally:
            self._cleanup_paged_batch_cuda_graphs(paged_decode_graphs)
            del state_manager
            gc.collect()

        return results

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
        output_queue = Queue()

        loop = asyncio.get_running_loop()

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
                data = await loop.run_in_executor(None, output_queue.get)
                if data == "EOF":
                    await future
                    yield "data: [DONE]\n\n"
                    return
                yield data
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
        total_batch_size = len(prompts)
        batch_size = total_batch_size
        state = None
        encoded_prompts = None
        out = None
        active_indices = None
        stop_states = None
        chunk_token_counts = None
        text_buffers = None
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

                active_indices = list(range(total_batch_size))
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(total_batch_size)]
                chunk_token_counts = [0] * total_batch_size
                text_buffers = [""] * total_batch_size

                while active_indices and max_length > 0:
                    new_tokens_tensor = self._sample_throughput_tokens(out, temperature)
                    new_tokens = new_tokens_tensor.tolist()

                    if batch_graph is None and self.enable_cuda_graphs:
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

                    contents_to_send = {}
                    surviving_local = []
                    surviving_indices = []

                    for local_idx, prompt_idx in enumerate(active_indices):
                        tok = (
                            new_tokens[local_idx][0]
                            if isinstance(new_tokens[local_idx], list)
                            else new_tokens[local_idx]
                        )

                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[prompt_idx], tok
                        )
                        if content:
                            text_buffers[prompt_idx] += content

                        if should_stop:
                            flushed = self._flush_stop_state(
                                stop_states[prompt_idx], final=True
                            )
                            if flushed:
                                text_buffers[prompt_idx] += flushed
                            if text_buffers[prompt_idx]:
                                contents_to_send[prompt_idx] = (
                                    contents_to_send.get(prompt_idx, "")
                                    + text_buffers[prompt_idx]
                                )
                                text_buffers[prompt_idx] = ""
                            continue

                        surviving_local.append(local_idx)
                        surviving_indices.append(prompt_idx)
                        chunk_token_counts[prompt_idx] += 1
                        if (
                            chunk_token_counts[prompt_idx] >= chunk_size
                            and text_buffers[prompt_idx]
                        ):
                            contents_to_send[prompt_idx] = (
                                contents_to_send.get(prompt_idx, "")
                                + text_buffers[prompt_idx]
                            )
                            text_buffers[prompt_idx] = ""
                            chunk_token_counts[prompt_idx] = 0

                    if contents_to_send:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [
                                {"index": i, "delta": {"content": content}}
                                for i, content in sorted(contents_to_send.items())
                                if content
                            ],
                        }
                        if chunk["choices"]:
                            yield (
                                f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            )

                    if not surviving_indices:
                        break

                    if len(surviving_indices) != batch_size:
                        state = self._select_state_rows(state, surviving_local)
                        out = self._select_rows(out, surviving_local)
                        batch_size = len(surviving_indices)
                        batch_graph = None

                    active_indices = surviving_indices

                    new_tokens = None
                    await asyncio.sleep(0)

                remaining_contents = [""] * total_batch_size
                for i in range(total_batch_size):
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
            if active_indices is not None:
                del active_indices
            if stop_states is not None:
                del stop_states
            if chunk_token_counts is not None:
                del chunk_token_counts
            if text_buffers is not None:
                del text_buffers
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
        total_batch_size = len(prompts)
        batch_size = total_batch_size
        state = None
        encoded_prompts = None
        out = None
        active_indices = None
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

                active_indices = list(range(total_batch_size))
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(total_batch_size)]
                text_buffers = [""] * total_batch_size

                while active_indices and max_length > 0:
                    new_tokens_tensor = self._sample_throughput_tokens(out, temperature)
                    new_tokens = new_tokens_tensor.tolist()

                    if batch_graph is None and self.enable_cuda_graphs:
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

                    surviving_local = []
                    surviving_indices = []
                    for local_idx, prompt_idx in enumerate(active_indices):
                        tok = (
                            new_tokens[local_idx][0]
                            if isinstance(new_tokens[local_idx], list)
                            else new_tokens[local_idx]
                        )
                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[prompt_idx], tok
                        )
                        if content:
                            text_buffers[prompt_idx] += content

                        if should_stop:
                            continue

                        surviving_local.append(local_idx)
                        surviving_indices.append(prompt_idx)

                    if not surviving_indices:
                        break

                    if len(surviving_indices) != batch_size:
                        state = self._select_state_rows(state, surviving_local)
                        out = self._select_rows(out, surviving_local)
                        batch_size = len(surviving_indices)
                        batch_graph = None

                    active_indices = surviving_indices

                for i in range(total_batch_size):
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
            if active_indices is not None:
                del active_indices
            if text_buffers is not None:
                del text_buffers
            if stop_states is not None:
                del stop_states
            if new_tokens is not None:
                del new_tokens
            self._cleanup_cuda_memory()

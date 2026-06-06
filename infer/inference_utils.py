import gc
from collections import deque

from infer import inference_deps


class InferenceUtilsMixin:
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

        static_input = inference_deps.get_torch().empty_like(x_emb, device="cuda")
        static_state = [None, None, None]
        static_state[0] = inference_deps.get_torch().empty_like(state[0], device="cuda")
        static_state[1] = inference_deps.get_torch().empty_like(state[1], device="cuda")
        static_state[2] = inference_deps.get_torch().empty_like(state[2], device="cuda")
        static_output = inference_deps.get_torch().empty_like(out, device="cuda")

        static_output = self.model.forward(static_input, static_state)

        g = inference_deps.get_torch().cuda.CUDAGraph()
        with inference_deps.get_torch().cuda.graph(g):
            static_output = self.model.forward(static_input, static_state)

        static_input.copy_(x_emb)
        static_state[0].copy_(state[0])
        static_state[1].copy_(state[1])
        static_state[2].copy_(state[2])
        static_output.copy_(out)

        return static_input, static_state, static_output, g

    @staticmethod
    def _cleanup_cuda_state(state):
        del state
        gc.collect()
        inference_deps.get_torch().cuda.empty_cache()

    @staticmethod
    def _cleanup_cuda_memory():
        gc.collect()
        if not inference_deps.get_torch().cuda.is_available():
            return
        try:
            inference_deps.get_torch().cuda.synchronize()
        except Exception:
            pass
        inference_deps.get_torch().cuda.empty_cache()

    @staticmethod
    def _torch_top_k_top_p(logits, top_k, top_p):
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = (
                logits < inference_deps.get_torch().topk(logits, top_k, dim=-1)[0][..., -1, None]
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = inference_deps.get_torch().sort(logits, descending=True, dim=-1)
            cumulative_probs = inference_deps.get_torch().cumsum(
                inference_deps.get_torch().softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., :1] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        probabilities = inference_deps.get_torch().softmax(logits, dim=-1)
        sampled_tokens = inference_deps.get_torch().multinomial(probabilities, 1).squeeze(-1)

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

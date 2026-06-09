import asyncio
import gc
import json
import random

from infer import inference_deps


class BatchInferenceMixin:
### batch generation for V1 endpoint ###

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
        cancel_token=None,
    ):
        state = None
        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            finished = [False] * batch_size
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            generated_text = [""] * batch_size
            sample_rand_states = inference_deps.get_sample().setup_rand(
                random.randint(0, 2**63 - 1), batch_size
            )
            penalties = inference_deps.get_torch().zeros(
                batch_size, out.size(-1), device=out.device
            )

            for _ in range(max_length):
                self._raise_if_cancelled(cancel_token)
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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

                    content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                    if content:
                        generated_text[i] += content

                    if should_stop:
                        finished[i] = True
                        continue

                if all(finished):
                    break

            decoded = []
            for i in range(batch_size):
                generated_text[i] += self._flush_stop_state(stop_states[i], final=True)
                decoded.append(generated_text[i])
            return decoded
        finally:
            if state is not None:
                del state
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

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
        cancel_token=None,
    ):
        state = None

        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            finished = [False] * batch_size
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            chunk_token_counts = [0] * batch_size
            text_buffers = [""] * batch_size
            sample_rand_states = inference_deps.get_sample().setup_rand(
                random.randint(0, 2**63 - 1), batch_size
            )
            penalties = inference_deps.get_torch().zeros(
                batch_size, out.size(-1), device=out.device
            )

            while not all(finished) and max_length > 0:
                self._raise_if_cancelled(cancel_token)
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

### generation for state reuse endpoint ### 

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
        cancel_token=None,
    ):
        try:
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]

            tokens = encoded_prompts[0]
            out = self._forward_tokens_chunked(tokens, state, cancel_token=cancel_token)
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            stop_state = self._create_stop_state(stop_tokens)
            generated_text = ""
            for _ in range(max_length):
                self._raise_if_cancelled(cancel_token)
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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

                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    break

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
            generated_text += self._flush_stop_state(stop_state, final=True)
            return [generated_text]
        finally:
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

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
        cancel_token=None,
    ):
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        chunk_size = max(1, int(chunk_size))
        should_store_state = True

        try:
            tokens = encoded_prompts[0]
            out = self._forward_tokens_chunked(tokens, state, cancel_token=cancel_token)
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)

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
            if cancel_token is not None and cancel_token.is_cancelled():
                should_store_state = False

            if state_manager and session_id and should_store_state:
                state_manager.put_state(session_id, state)
                print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")

            del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

### generation for OpenAI compatible bsz=1 endpoint ###

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
        cancel_token=None,
    ):
        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        finish_reason = "length"
        state = None

        try:
            _, state, out, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                logits_reshaped = out.unsqueeze(0) if out.dim() == 1 else out
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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

                tok = new_tokens[0]
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    finish_reason = "stop"
                    break

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
                await asyncio.sleep(0)

            generated_text += self._flush_stop_state(stop_state, final=True)
            return generated_text, finish_reason
        finally:
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
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
        cancel_token=None,
    ):
        finish_reason = "length"
        state = None

        try:
            _, state, out, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                logits_reshaped = out.unsqueeze(0) if out.dim() == 1 else out
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
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

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
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
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

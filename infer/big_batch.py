import asyncio
import json

from infer import inference_deps


class BigBatchMixin:
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

        try:
            with inference_deps.get_torch().inference_mode():
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
                out = self.model.forward_batch(encoded_prompts, state)

                finished = [False] * batch_size
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
                chunk_token_counts = [0] * batch_size
                text_buffers = [""] * batch_size

                step_count = 0
                cleanup_interval = 100

                while not all(finished) and max_length > 0:
                    new_tokens_tensor = inference_deps.get_sampler_gumbel_batch()(
                        logits=out, temp=temperature
                    )
                    new_tokens = new_tokens_tensor.tolist()
                    del new_tokens_tensor
                    new_tokens_tensor = None

                    prev_out = out
                    out = self.model.forward_batch(new_tokens, state)
                    del prev_out

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

                    if step_count % cleanup_interval == 0:
                        self._cleanup_cuda_memory()

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

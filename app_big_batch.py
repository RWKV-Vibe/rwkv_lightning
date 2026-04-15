import argparse
import atexit
import gc
import json
import os
import signal
import sys

import torch
from pydantic import BaseModel, Field
from robyn import Response, Robyn, StreamingResponse

from infer.rwkv_batch.utils import sampler_gumbel_batch
from model_load.model_loader import load_model_and_tokenizer


class BigBatchRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 1.0
    stop_tokens: list[int] = Field(default_factory=lambda: [0, 261, 24281])
    chunk_size: int = 32
    stream: bool = True
    password: str | None = None


class BigBatchEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    async def batch_stream(
        self,
        prompts: list[str],
        max_length: int,
        temperature: float,
        stop_tokens: tuple[int, ...],
        chunk_size: int,
    ):
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        finished = None
        token_buffers = None
        new_tokens_tensor = None
        new_tokens = None

        try:
            with torch.inference_mode():
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(prompt) for prompt in prompts]
                out = self.model.forward_batch(encoded_prompts, state)
                finished = [False] * batch_size
                token_buffers = [[] for _ in range(batch_size)]

                while not all(finished) and max_length > 0:
                    new_tokens_tensor = sampler_gumbel_batch(
                        logits=out.clone(), temp=temperature
                    )
                    new_tokens = new_tokens_tensor.tolist()
                    del new_tokens_tensor
                    new_tokens_tensor = None

                    prev_out = out
                    out = self.model.forward_batch(new_tokens, state)
                    del prev_out

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

                        if len(token_buffers[i]) >= chunk_size:
                            contents_to_send[i] = self.tokenizer.decode(
                                token_buffers[i], utf8_errors="ignore"
                            )
                            token_buffers[i].clear()

                    if any(contents_to_send):
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": i,
                                            "delta": {"content": contents_to_send[i]},
                                        }
                                        for i in range(batch_size)
                                        if contents_to_send[i]
                                    ],
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )

                    new_tokens = None

                remaining_contents = [""] * batch_size
                for i in range(batch_size):
                    if token_buffers[i]:
                        remaining_contents[i] = self.tokenizer.decode(
                            token_buffers[i], utf8_errors="ignore"
                        )
                        token_buffers[i].clear()

                if any(remaining_contents):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": i,
                                        "delta": {"content": remaining_contents[i]},
                                    }
                                    for i in range(batch_size)
                                    if remaining_contents[i]
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
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
            if token_buffers is not None:
                del token_buffers
            if new_tokens is not None:
                del new_tokens

        yield "data: [DONE]\n\n"
        self.cleanup()

    def batch_generate(
        self,
        prompts: list[str],
        max_length: int,
        temperature: float,
        stop_tokens: tuple[int, ...],
    ):
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        finished = None
        generated_tokens = None
        new_tokens_tensor = None
        new_tokens = None

        try:
            with torch.inference_mode():
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(prompt) for prompt in prompts]
                out = self.model.forward_batch(encoded_prompts, state)
                finished = [False] * batch_size
                generated_tokens = [[] for _ in range(batch_size)]

                while not all(finished) and max_length > 0:
                    new_tokens_tensor = sampler_gumbel_batch(
                        logits=out.clone(), temp=temperature
                    )
                    new_tokens = new_tokens_tensor.tolist()
                    del new_tokens_tensor
                    new_tokens_tensor = None

                    prev_out = out
                    out = self.model.forward_batch(new_tokens, state)
                    del prev_out

                    max_length -= 1

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
                            continue

                        generated_tokens[i].append(tok)

                return [
                    self.tokenizer.decode(tokens, utf8_errors="ignore")
                    for tokens in generated_tokens
                ]
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
            if new_tokens is not None:
                del new_tokens
            self.cleanup()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", type=str, default=None)
    return parser.parse_args()


def create_app(engine: BigBatchEngine, model_name: str, password: str | None):
    app = Robyn(__file__)

    def json_response(status_code: int, payload: dict):
        return Response(
            status_code=status_code,
            description=json.dumps(payload, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    @app.after_request()
    def after_request(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.options("/big_batch/completions")
    @app.options("/healthz")
    async def options_handler():
        return Response(
            status_code=204,
            description="",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        )

    @app.get("/healthz")
    async def healthz():
        return json_response(200, {"status": "ok", "model": model_name})

    @app.post("/big_batch/completions")
    async def big_batch_completions(request):
        try:
            req = BigBatchRequest(**json.loads(request.body))
        except Exception as exc:
            return json_response(400, {"error": f"invalid request: {exc}"})

        if password and req.password != password:
            return json_response(401, {"error": "Unauthorized: invalid or missing password"})

        if not req.contents:
            return json_response(400, {"error": "contents is empty"})

        if req.stream:
            return StreamingResponse(
                engine.batch_stream(
                    prompts=req.contents,
                    max_length=max(1, req.max_tokens),
                    temperature=max(req.temperature, 1e-5),
                    stop_tokens=tuple(req.stop_tokens or [0, 261, 24281]),
                    chunk_size=max(1, req.chunk_size),
                ),
                media_type="text/event-stream",
            )

        results = engine.batch_generate(
            prompts=req.contents,
            max_length=max(1, req.max_tokens),
            temperature=max(req.temperature, 1e-5),
            stop_tokens=tuple(req.stop_tokens or [0, 261, 24281]),
        )
        return json_response(
            200,
            {
                "object": "chat.completion",
                "model": req.model,
                "choices": [
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                    for i, text in enumerate(results)
                ],
            },
        )

    return app


def main():
    cli_args = parse_args()
    model, tokenizer, args, _ = load_model_and_tokenizer(cli_args.model_path)
    engine = BigBatchEngine(model=model, tokenizer=tokenizer)
    model_name = os.path.basename(f"{args.MODEL_NAME}")
    app = create_app(engine, model_name, cli_args.password)

    def cleanup_handler(signum=None, frame=None):
        engine.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(engine.cleanup)
    app.start(host="0.0.0.0", port=cli_args.port)


if __name__ == "__main__":
    main()

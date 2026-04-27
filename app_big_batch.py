import argparse
import asyncio
import atexit
import gc
import json
import os
import queue
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

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

    def batch_stream_sync(
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


class RequestQueueManager:
    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._next_ticket = 0
        self._current_ticket = 0
        self._skipped_tickets = set()

    def acquire_ticket(self) -> int:
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            return ticket

    def get_status(self, ticket: int) -> dict:
        with self._lock:
            ahead = max(ticket - self._current_ticket, 0)
            waiting = max(self._next_ticket - self._current_ticket - 1, 0)
            return {
                "ticket": ticket,
                "ahead": ahead,
                "waiting": waiting,
                "current_ticket": self._current_ticket,
                "status": "processing" if ahead == 0 else "queued",
            }

    def is_turn(self, ticket: int) -> bool:
        with self._lock:
            return ticket == self._current_ticket

    def finish(self, ticket: int):
        with self._lock:
            if ticket == self._current_ticket:
                self._current_ticket += 1
                while self._current_ticket in self._skipped_tickets:
                    self._skipped_tickets.remove(self._current_ticket)
                    self._current_ticket += 1
                return

            if ticket > self._current_ticket:
                self._skipped_tickets.add(ticket)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", type=str, default=None)
    return parser.parse_args()


def create_app(
    engine: BigBatchEngine,
    model_name: str,
    password: str | None,
    queue_manager: RequestQueueManager,
):
    app = Robyn(__file__)

    def json_response(status_code: int, payload: dict):
        return Response(
            status_code=status_code,
            description=json.dumps(payload, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    def sse_payload(payload: dict):
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

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

        ticket = queue_manager.acquire_ticket()

        if req.stream:
            async def queued_stream():
                try:
                    while not queue_manager.is_turn(ticket):
                        status = queue_manager.get_status(ticket)
                        yield sse_payload(
                            {
                                "object": "queue.status",
                                "status": status["status"],
                                "ticket": status["ticket"],
                                "queue_position": status["ahead"] + 1,
                                "requests_ahead": status["ahead"],
                                "waiting_requests": status["waiting"],
                            }
                        )
                        await asyncio.sleep(1)

                    status = queue_manager.get_status(ticket)
                    yield sse_payload(
                        {
                            "object": "queue.status",
                            "status": "processing",
                            "ticket": ticket,
                            "queue_position": 1,
                            "requests_ahead": 0,
                            "waiting_requests": status["waiting"],
                        }
                    )

                    chunk_queue = queue.Queue()

                    def produce_stream():
                        try:
                            for chunk in engine.batch_stream_sync(
                                prompts=req.contents,
                                max_length=max(1, req.max_tokens),
                                temperature=max(req.temperature, 1e-5),
                                stop_tokens=tuple(req.stop_tokens or [0, 261, 24281]),
                                chunk_size=max(1, req.chunk_size),
                            ):
                                chunk_queue.put(("chunk", chunk))
                        except Exception as exc:
                            chunk_queue.put(("error", str(exc)))
                        finally:
                            chunk_queue.put(("done", None))

                    queue_manager.executor.submit(produce_stream)

                    while True:
                        kind, payload = await asyncio.to_thread(chunk_queue.get)
                        if kind == "chunk":
                            yield payload
                            continue
                        if kind == "error":
                            yield sse_payload({"object": "error", "message": payload})
                        break
                finally:
                    queue_manager.finish(ticket)

            return StreamingResponse(
                queued_stream(),
                media_type="text/event-stream",
            )

        try:
            while not queue_manager.is_turn(ticket):
                await asyncio.sleep(1)

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                queue_manager.executor,
                engine.batch_generate,
                req.contents,
                max(1, req.max_tokens),
                max(req.temperature, 1e-5),
                tuple(req.stop_tokens or [0, 261, 24281]),
            )
        finally:
            queue_manager.finish(ticket)

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
    queue_manager = RequestQueueManager(max_workers=1)
    model_name = os.path.basename(f"{args.MODEL_NAME}")
    app = create_app(engine, model_name, cli_args.password, queue_manager)

    def cleanup_handler(signum=None, frame=None):
        queue_manager.shutdown()
        engine.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(queue_manager.shutdown)
    atexit.register(engine.cleanup)
    app.start(host="0.0.0.0", port=cli_args.port)


if __name__ == "__main__":
    main()

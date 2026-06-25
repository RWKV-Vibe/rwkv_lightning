import asyncio
from contextlib import suppress

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from infer.cancellation import CancellationToken, InferenceCancelled
from infer.high_throughput import (
    encode_high_throughput_prompts,
    run_high_throughput_generate,
    run_high_throughput_stream,
)

from API_servers.router.common import (
    SSE_HEADERS,
    check_password,
    client_closed_response,
    cleanup_disconnect_watcher,
    json_response,
    run_sync_with_disconnect_watch,
    watch_disconnect,
)
from API_servers.router.schemas import ChatRequest


router = APIRouter(tags=["high_throughput"])


class HighThroughputInputItem(BaseModel):
    id: int
    text: str


class HighThroughputChatRequest(ChatRequest):
    client_id: str | None = None
    batch_id: str | None = None
    items: list[HighThroughputInputItem] = Field(default_factory=list)
    max_batch_size: int | None = None
    decode_max_batch_size: int | None = None
    prefill_area: int | None = None
    batch_size: int | None = None
    target_batch_size: int | None = None
    prefill_target_batch_size: int | None = None
    stop_tokens: list[int | str] = Field(default_factory=lambda: ["\nUser:"])


def _extract_prompts(req: HighThroughputChatRequest):
    if req.items:
        return [item.text for item in req.items], [item.id for item in req.items]
    return req.contents, [None] * len(req.contents)


async def _cleanup_high_throughput_stream(stream, cancel_token, stream_state: dict):
    current_task = asyncio.current_task()
    cleanup_task = stream_state.get("cleanup_task")
    if stream_state.get("cleanup_done"):
        return
    if cleanup_task is not None and cleanup_task is not current_task:
        with suppress(asyncio.CancelledError, Exception):
            await cleanup_task
        return

    stream_state["cleanup_task"] = current_task
    try:
        cancel_token.cancel()
        watcher = stream_state.get("watcher")
        if watcher is not None:
            await cleanup_disconnect_watcher(watcher)
            stream_state["watcher"] = None

        try:
            await stream.aclose()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    finally:
        stream_state["cleanup_done"] = True


def _schedule_high_throughput_stream_cleanup(stream, cancel_token, stream_state: dict):
    if stream_state.get("cleanup_done") or stream_state.get("cleanup_task") is not None:
        return

    stream_state["cleanup_task"] = asyncio.create_task(
        _cleanup_high_throughput_stream(stream, cancel_token, stream_state)
    )


def high_throughput_sse_response(request: Request, stream, cancel_token: CancellationToken):
    stream_state = {
        "watcher": None,
        "cleanup_task": None,
        "cleanup_done": False,
    }

    async def body():
        try:
            stream_state["watcher"] = asyncio.create_task(
                watch_disconnect(request, cancel_token)
            )
            async for chunk in stream:
                if cancel_token.is_cancelled():
                    break
                yield chunk
                if isinstance(chunk, str) and chunk.strip() == "data: [DONE]":
                    break
        except InferenceCancelled:
            cancel_token.cancel()
        except asyncio.CancelledError:
            cancel_token.cancel()
            raise
        finally:
            _schedule_high_throughput_stream_cleanup(
                stream,
                cancel_token,
                stream_state,
            )

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(
            _cleanup_high_throughput_stream,
            stream,
            cancel_token,
            stream_state,
        ),
    )


@router.post("/high_throughput/chat/completions")
async def high_throughput_chat_completions(request: Request):
    app_state = request.app.state
    runtime = getattr(app_state, "high_throughput_runtime", None)
    if runtime is None:
        return json_response(
            404,
            {"error": "high_throughput endpoint is disabled"},
        )

    body = await request.json()
    req = HighThroughputChatRequest(**body)

    auth_error = check_password(req.password, app_state.password)
    if auth_error is not None:
        return auth_error

    prompts, item_ids = _extract_prompts(req)
    if not prompts:
        return json_response(
            400,
            {"error": "contents or items must contain at least one prompt"},
        )

    plan = runtime.make_plan(
        total_items=len(prompts),
        decode_max_batch_size=req.decode_max_batch_size or req.max_batch_size,
        prefill_area=req.prefill_area,
        prefill_target_batch_size=(
            req.prefill_target_batch_size
            or req.target_batch_size
            or req.batch_size
        ),
    )
    encoded_prompts = encode_high_throughput_prompts(runtime.engine, prompts)

    if req.stream:
        cancel_token = CancellationToken()
        stream = run_high_throughput_stream(
            runtime,
            plan=plan,
            prompts=prompts,
            encoded_prompts=encoded_prompts,
            item_ids=item_ids,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
            chunk_size=req.chunk_size,
            cancel_token=cancel_token,
        )
        return high_throughput_sse_response(request, stream, cancel_token)

    try:
        choices, usages = await run_sync_with_disconnect_watch(
            request,
            run_high_throughput_generate,
            runtime=runtime,
            plan=plan,
            prompts=prompts,
            encoded_prompts=encoded_prompts,
            item_ids=item_ids,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
        )
    except InferenceCancelled:
        return client_closed_response()

    schedule = plan.as_dict()
    schedule["uses_v1_prefill_queue"] = False

    response = {
        "id": "rwkv7-high-throughput",
        "object": "chat.completion",
        "model": req.model,
        "endpoint": "high_throughput",
        "schedule": schedule,
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(item["prompt_tokens"] for item in usages if item),
            "completion_tokens": sum(item["completion_tokens"] for item in usages if item),
            "total_tokens": sum(
                item["prompt_tokens"] + item["completion_tokens"]
                for item in usages
                if item
            ),
            "items": usages,
        },
    }
    if req.client_id is not None:
        response["client_id"] = req.client_id
    if req.batch_id is not None:
        response["batch_id"] = req.batch_id
    return response

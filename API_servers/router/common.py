import asyncio
import json
from contextlib import asynccontextmanager
from contextlib import suppress

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from infer.cancellation import CancellationToken, InferenceCancelled


def json_response(status_code: int, payload: dict):
    return JSONResponse(status_code=status_code, content=payload)


def client_closed_response():
    return Response(status_code=499)


def check_password(body_password, password):
    if password and body_password != password:
        return json_response(401, {"error": "Unauthorized: invalid or missing password"})
    return None


def normalize_state_prompts(prompts: list[str], reuse_existing_state: bool) -> list[str]:
    if not reuse_existing_state:
        return prompts

    normalized_prompts = []
    for prompt in prompts:
        if prompt and not prompt.startswith("\n\n"):
            normalized_prompts.append(f"\n\n{prompt}")
        else:
            normalized_prompts.append(prompt)
    return normalized_prompts


def collect_session_indices(state_manager, session_index: str) -> list[int]:
    prefix = f"{session_index}:"
    all_states = state_manager.list_all_states()
    indices = []
    for key in all_states["l1_cache"] + all_states["l2_cache"] + all_states["database"]:
        if key.startswith(prefix):
            tail = key[len(prefix) :]
            if tail.isdigit():
                indices.append(int(tail))
    return indices


def allocate_next_dialogue_idx(app_state, state_manager, session_index: str) -> int:
    with app_state.dialogue_idx_lock:
        if session_index in app_state.dialogue_idx_counters:
            next_idx = app_state.dialogue_idx_counters[session_index]
            app_state.dialogue_idx_counters[session_index] = next_idx + 1
            return next_idx

        indices = collect_session_indices(state_manager, session_index)
        max_idx = max(indices) if indices else 0
        next_idx = max_idx + 1
        app_state.dialogue_idx_counters[session_index] = next_idx + 1
        return next_idx


async def watch_disconnect(request: Request, cancel_token: CancellationToken):
    try:
        while not cancel_token.is_cancelled():
            if await request.is_disconnected():
                cancel_token.cancel()
                return
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        raise


async def cleanup_disconnect_watcher(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def reserve_prefill_capacity(request: Request, request_bsz: int):
    engine = request.app.state.engine
    permit = await engine.acquire_prefill_permit(
        request_bsz=request_bsz,
        request_label=str(request.url.path),
    )
    try:
        yield permit
    finally:
        await engine.release_prefill_permit(
            request_bsz=request_bsz,
            request_label=str(request.url.path),
            ticket=permit["ticket"],
        )


async def run_sync_with_disconnect_watch(request: Request, func, **kwargs):
    cancel_token = CancellationToken()
    watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
    try:
        result = await run_in_threadpool(func, cancel_token=cancel_token, **kwargs)
        if cancel_token.is_cancelled():
            raise InferenceCancelled("request disconnected")
        return result
    finally:
        await cleanup_disconnect_watcher(watcher)


async def stream_with_disconnect_watch(request: Request, stream, cancel_token: CancellationToken):
    watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
    try:
        async for chunk in stream:
            if cancel_token.is_cancelled():
                break
            yield chunk
    except InferenceCancelled:
        cancel_token.cancel()
    finally:
        await stream.aclose()
        await cleanup_disconnect_watcher(watcher)


async def stream_with_prefill_queue(
    request: Request,
    stream,
    cancel_token: CancellationToken,
    request_bsz: int,
):
    async with reserve_prefill_capacity(request, request_bsz):
        async for chunk in stream_with_disconnect_watch(request, stream, cancel_token):
            yield chunk


def extract_sse_payload(item: str) -> str | None:
    if not isinstance(item, str) or not item.startswith("data: "):
        return None
    return item[6:].strip()


def emit_finish_reason_chunk(response_id: str, created: int, model_name: str, finish_reason: str):
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

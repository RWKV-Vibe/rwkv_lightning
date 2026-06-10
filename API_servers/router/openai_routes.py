import json
import os
import time
from typing import Any
import asyncio

from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager

from API_servers.router.common import (
    cleanup_disconnect_watcher,
    client_closed_response,
    emit_finish_reason_chunk,
    extract_sse_payload,
    json_response,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    watch_disconnect,
)
from API_servers.router.schemas import ChatRequest


router = APIRouter()


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def sanitize_text_block(content) -> str:
    normalized = normalize_message_content(content)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def collect_openai_prompt_parts(body: dict) -> tuple[str, list[str]]:
    messages = body.get("messages") or []
    contents = body.get("contents") or []

    system_parts = []
    transcript_parts = []

    system_field = sanitize_text_block(body.get("system"))
    if system_field:
        system_parts.append(system_field)

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "user")).lower()
        content = sanitize_text_block(message.get("content", ""))

        if role in {"system", "developer"}:
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            if content:
                transcript_parts.append(f"User: {content}")
            continue

        if role == "assistant":
            if content:
                transcript_parts.append(f"Assistant: {content}")
            continue

    if contents:
        content_prompt = sanitize_text_block(contents[0])
        if content_prompt:
            transcript_parts.append(f"User: {content_prompt}")

    system_text = "\n".join(part for part in system_parts if part).strip()
    transcript_parts = [part for part in transcript_parts if part]
    return system_text, transcript_parts


def extract_openai_prompt(body: dict) -> str:
    system_text, transcript_parts = collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(system_text)
    prompt_parts.extend(transcript_parts)
    return "\n".join(part for part in prompt_parts if part).strip()


def format_openai_prompt(body: dict, enable_think: bool) -> str:
    system_text, transcript_parts = collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(f"System: {system_text}")
    prompt_parts.extend(transcript_parts)

    prompt_text = "\n\n".join(part for part in prompt_parts if part).strip()
    if not prompt_text:
        raise ValueError("OpenAI chat completions require system or user text")

    if enable_think:
        return f"{prompt_text}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>\n"


def build_openai_usage(tokenizer, prompt_text: str, completion_text: str) -> dict:
    prompt_tokens = len(tokenizer.encode(prompt_text))
    completion_tokens = len(tokenizer.encode(completion_text)) if completion_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_internal_chat_request(body: dict, prompt: str) -> dict:
    stream = body.get("stream", False)
    chunk_size = body.get("chunk_size")
    if chunk_size is None:
        chunk_size = 1 if stream else 16

    return {
        "model": body.get("model", "rwkv7"),
        "contents": [prompt],
        "messages": body.get("messages", []),
        "system": body.get("system"),
        "max_tokens": body.get("max_tokens", 4096),
        "stop_tokens": body.get("stop_tokens", ["\nUser:"]),
        "temperature": body.get("temperature", 1.0),
        "top_k": body.get("top_k", 20),
        "top_p": body.get("top_p", 0.6),
        "stream": stream,
        "pad_zero": body.get("pad_zero", False),
        "alpha_presence": body.get("alpha_presence", 1),
        "alpha_frequency": body.get("alpha_frequency", 0.1),
        "alpha_decay": body.get("alpha_decay", 0.996),
        "enable_think": body.get("enable_think", False),
        "chunk_size": chunk_size,
        "password": body.get("password"),
        "session_id": body.get("session_id"),
        "use_prefix_cache": body.get("use_prefix_cache", True),
    }


def build_openai_message_response(
    result_text: str, finish_reason: str, body: dict
) -> tuple[dict[str, Any], str]:
    return {"role": "assistant", "content": result_text}, finish_reason


def extract_bearer_token(request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip()


def check_openai_auth(request: Request, body: dict, password):
    if not password:
        return None
    bearer_token = extract_bearer_token(request)
    body_password = body.get("password")
    if bearer_token == password or body_password == password:
        return None
    return json_response(401, {"error": "Unauthorized: invalid or missing password"})


async def stream_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
    cancel_token: CancellationToken,
    prefix_cache_manager=None,
):
    emitted_finish_reason = False
    start_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

    stream = engine.singe_infer_stream(
        prompt=prompt_formatted,
        max_length=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        alpha_presence=req.alpha_presence,
        alpha_frequency=req.alpha_frequency,
        alpha_decay=req.alpha_decay,
        stop_tokens=req.stop_tokens,
        chunk_size=req.chunk_size,
        prefix_cache_manager=prefix_cache_manager,
        cancel_token=cancel_token,
    )

    try:
        async for item in stream:
            payload = extract_sse_payload(item)
            if payload is None:
                continue

            if payload == "[DONE]":
                if not emitted_finish_reason and not cancel_token.is_cancelled():
                    yield emit_finish_reason_chunk(response_id, created, model_name, "stop")
                break

            try:
                chunk_payload = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk_payload.get("choices") or []
            if not choices:
                continue

            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                if not cancel_token.is_cancelled():
                    yield emit_finish_reason_chunk(
                        response_id,
                        created,
                        model_name,
                        finish_reason,
                    )
                    emitted_finish_reason = True
                continue

            content = choices[0].get("delta", {}).get("content")
            if not content or cancel_token.is_cancelled():
                continue

            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finally:
        await stream.aclose()

    if not cancel_token.is_cancelled():
        yield "data: [DONE]\n\n"


@router.get("/openai/v1/models")
async def openai_list_models(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    auth_error = check_openai_auth(request, {}, password)
    if auth_error is not None:
        return auth_error

    model_name = os.path.basename(f"{engine.args.MODEL_NAME}")
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "owned_by": "rwkv_lightning"}],
    }


@router.post("/openai/v1/chat/completions")
async def openai_chat_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    try:
        body = await request.json()
        auth_error = check_openai_auth(request, body, password)
        if auth_error is not None:
            return auth_error

        prompt = extract_openai_prompt(body)
        if not prompt and not (body.get("messages") or []):
            return json_response(400, {"error": "Empty prompt"})

        req = ChatRequest(**build_internal_chat_request(body, prompt))
        prompt_formatted = format_openai_prompt(body, req.enable_think)

        response_id = f"chatcmpl-{os.urandom(12).hex()}"
        created = int(time.time())
        model_name = os.path.basename(f"{engine.args.MODEL_NAME}")
        prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None

        if req.stream:
            cancel_token = CancellationToken()
            stream = stream_openai_chunks(
                engine,
                req,
                prompt_formatted,
                response_id,
                created,
                model_name,
                cancel_token,
                prefix_cache_manager,
            )
            return prefill_sse_response(request, stream, cancel_token, 1)

        async with reserve_prefill_capacity(request, 1):
            cancel_token = CancellationToken()
            watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
            try:
                result_text, finish_reason = await engine.singe_infer(
                    prompt=prompt_formatted,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                    prefix_cache_manager=prefix_cache_manager,
                    cancel_token=cancel_token,
                )
                if cancel_token.is_cancelled():
                    raise InferenceCancelled("request disconnected")
            finally:
                await cleanup_disconnect_watcher(watcher)

        message, response_finish_reason = build_openai_message_response(
            result_text, finish_reason, body
        )
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response_finish_reason,
                }
            ],
            "usage": build_openai_usage(engine.tokenizer, prompt_formatted, result_text),
        }
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except json.JSONDecodeError as exc:
        return json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
    except Exception as exc:
        print(f"[ERROR] /openai/v1/chat/completions: {exc}")
        return json_response(500, {"error": str(exc)})

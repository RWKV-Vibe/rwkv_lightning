import json
import os
import time
import traceback
import uuid
import json
from typing import Any

from pydantic import ValidationError
from robyn import Response, StreamingResponse

from state_manager.state_pool import get_state_manager


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


def _sanitize_text_block(content) -> str:
    normalized = normalize_message_content(content)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _collect_openai_prompt_parts(body: dict) -> tuple[str, list[str]]:
    messages = body.get("messages") or []
    contents = body.get("contents") or []

    system_parts = []
    transcript_parts = []

    system_field = _sanitize_text_block(body.get("system"))
    if system_field:
        system_parts.append(system_field)

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "user")).lower()
        content = _sanitize_text_block(message.get("content", ""))

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
        content_prompt = _sanitize_text_block(contents[0])
        if content_prompt:
            transcript_parts.append(f"User: {content_prompt}")

    system_text = "\n".join(part for part in system_parts if part).strip()
    transcript_parts = [part for part in transcript_parts if part]
    return system_text, transcript_parts


def extract_openai_prompt(body: dict) -> str:
    system_text, transcript_parts = _collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(system_text)
    prompt_parts.extend(transcript_parts)
    return "\n".join(part for part in prompt_parts if part).strip()


def format_openai_prompt(body: dict, enable_think: bool) -> str:
    system_text, transcript_parts = _collect_openai_prompt_parts(body)
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


def format_openai_state_prompt(body: dict, enable_think: bool) -> str:
    contents = body.get("contents") or []
    if len(contents) > 1:
        raise ValueError("State mode only supports a single contents item")
    return format_openai_prompt(body, enable_think)


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
        "runtime": body.get("runtime"),
        "scheduler": body.get("scheduler", "auto"),
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


def _emit_finish_reason_chunk(
    response_id: str,
    created: int,
    model_name: str,
    finish_reason: str,
) -> str:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _extract_sse_payload(item: str) -> str | None:
    if not isinstance(item, str) or not item.startswith("data: "):
        return None
    return item[6:].strip()


def _json_response(status_code: int, payload: dict):
    return Response(
        status_code=status_code,
        description=json.dumps(payload, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
    )


def _extract_bearer_token(request):
    headers = getattr(request, "headers", {}) or {}
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip()


def _check_openai_auth(request, body: dict, password):
    if not password:
        return None
    bearer_token = _extract_bearer_token(request)
    body_password = body.get("password")
    if bearer_token == password or body_password == password:
        return None
    return _json_response(401, {"error": "Unauthorized: invalid or missing password"})


async def _stream_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
    prefix_cache_manager=None,
):
    emitted_finish_reason = False
    start_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

    async for item in engine.singe_infer_stream(
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
    ):
        payload = _extract_sse_payload(item)
        if payload is None:
            continue

        if payload == "[DONE]":
            if not emitted_finish_reason:
                yield _emit_finish_reason_chunk(
                    response_id, created, model_name, "stop"
                )
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
            yield _emit_finish_reason_chunk(
                response_id,
                created,
                model_name,
                finish_reason,
            )
            emitted_finish_reason = True
            continue

        content = choices[0].get("delta", {}).get("content")
        if not content:
            continue

        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def register_openai_routes(app, engine, password, chat_request_model):
    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request):
        try:
            body = json.loads(request.body)
            auth_error = _check_openai_auth(request, body, password)
            if auth_error is not None:
                return auth_error

            served_model_id = getattr(
                engine.args,
                "served_model_id",
                os.path.basename(f"{engine.args.MODEL_NAME}"),
            )
            served_aliases = set(getattr(engine.args, "model_aliases", (served_model_id,)))
            requested_model = str(body.get("model", "rwkv7")).strip()
            if requested_model and requested_model not in served_aliases:
                return _json_response(
                    404,
                    {
                        "error": f"Model '{requested_model}' is not served by this process",
                        "available_models": [served_model_id],
                    },
                )

            requested_runtime = body.get("runtime")
            served_runtime = getattr(engine.args, "runtime", "fp16")
            if requested_runtime and str(requested_runtime).strip().lower() != served_runtime:
                return _json_response(
                    400,
                    {
                        "error": f"Runtime '{requested_runtime}' is not available on this server",
                        "served_runtime": served_runtime,
                    },
                )

            prompt = extract_openai_prompt(body)
            if not prompt and not (body.get("messages") or []):
                return _json_response(400, {"error": "Empty prompt"})

            req = chat_request_model(**build_internal_chat_request(body, prompt))

            # print(f"[OpenAI] Request: {req}")

            prompt_formatted = format_openai_prompt(body, req.enable_think)

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = served_model_id
            prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None

            if req.stream:
                return StreamingResponse(
                    _stream_openai_chunks(
                        engine,
                        req,
                        prompt_formatted,
                        response_id,
                        created,
                        model_name,
                        prefix_cache_manager,
                    ),
                    media_type="text/event-stream",
                )

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
            )

            message, response_finish_reason = build_openai_message_response(
                result_text, finish_reason, body
            )
            response = {
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
                "usage": build_openai_usage(
                    engine.tokenizer, prompt_formatted, result_text
                ),
            }
            return _json_response(200, response)
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except ValidationError as exc:
            return _json_response(400, {"error": exc.errors()})
        except Exception as exc:
            print(f"[ERROR] /openai/v1/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

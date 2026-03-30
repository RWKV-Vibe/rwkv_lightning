import json
import os
import time
import traceback
import uuid

from robyn import Response, StreamingResponse

from API_servers.openai_adapter import (
    apply_tool_compatibility,
    build_internal_chat_request,
    build_openai_message_response,
    build_openai_usage,
    extract_openai_prompt,
    format_openai_prompt,
)


def _tool_stream_enabled(body: dict) -> bool:
    return bool(
        body.get("enable_tool_calls", False)
        or body.get("_rwkv_tool_compat_active", False)
    )


def _parse_tool_payload(payload: str) -> dict | None:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    arguments = parsed.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(arguments, dict):
        return None
    return {
        "name": name.strip(),
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


class _ToolCallStreamProcessor:
    def __init__(self):
        self.buffer = ""
        self.in_tool = False
        self.tool_buffer = ""
        self.tool_emitted = False

    def ingest(self, text: str) -> list[tuple[str, str | dict]]:
        if not text:
            return []
        if self.tool_emitted:
            return [("content", text)]
        self.buffer += text
        return self._ingest_tag()

    def flush(self) -> list[tuple[str, str | dict]]:
        if self.in_tool:
            raw = f"<tool_call>{self.tool_buffer}{self.buffer}"
            self.tool_buffer = ""
            self.buffer = ""
            self.in_tool = False
            return [("content", raw)]
        if self.buffer:
            leftover = self.buffer
            self.buffer = ""
            return [("content", leftover)]
        return []

    def _ingest_tag(self) -> list[tuple[str, str | dict]]:
        events: list[tuple[str, str | dict]] = []
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        while True:
            if not self.in_tool:
                start_idx = self.buffer.find(start_tag)
                if start_idx == -1:
                    emit, keep = _split_incomplete_tag(self.buffer, start_tag)
                    if emit:
                        events.append(("content", emit))
                    self.buffer = keep
                    break
                before = self.buffer[:start_idx]
                if before:
                    events.append(("content", before))
                self.buffer = self.buffer[start_idx + len(start_tag) :]
                self.in_tool = True
                self.tool_buffer = ""

            combined = self.tool_buffer + self.buffer
            end_idx = combined.find(end_tag)
            if end_idx == -1:
                self.tool_buffer = combined
                self.buffer = ""
                break

            payload = combined[:end_idx]
            trailing = combined[end_idx + len(end_tag) :]
            self.in_tool = False
            self.tool_buffer = ""
            self.buffer = ""

            parsed = _parse_tool_payload(payload)
            if parsed is not None:
                events.append(("tool", parsed))
                self.tool_emitted = True
                if trailing:
                    events.append(("content", trailing))
                break
            events.append(("content", f"{start_tag}{payload}{end_tag}"))
            self.buffer = trailing
        return events


def _split_incomplete_tag(buffer: str, tag: str) -> tuple[str, str]:
    max_keep = 0
    for i in range(1, len(tag)):
        if buffer.endswith(tag[:i]):
            max_keep = i
    if max_keep:
        return buffer[:-max_keep], buffer[-max_keep:]
    return buffer, ""


def _emit_tool_call_chunk(
    response_id: str,
    created: int,
    model_name: str,
    tool_call: dict,
) -> str:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


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


def _flush_tool_processor_content(tool_processor) -> list[str]:
    if tool_processor is None:
        return []

    flushed_content: list[str] = []
    for event_type, value in tool_processor.flush():
        if event_type == "content" and value:
            flushed_content.append(value)
    return flushed_content


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
    body: dict,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
):
    tool_processor = None
    tool_finish_reason = None
    emitted_finish_reason = False
    if _tool_stream_enabled(body):
        tool_processor = _ToolCallStreamProcessor()

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
    ):
        payload = _extract_sse_payload(item)
        if payload is None:
            continue

        if payload == "[DONE]":
            for value in _flush_tool_processor_content(tool_processor):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": value},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if not emitted_finish_reason:
                finish_reason = tool_finish_reason or "stop"
                yield _emit_finish_reason_chunk(
                    response_id, created, model_name, finish_reason
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
            for value in _flush_tool_processor_content(tool_processor):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": value},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield _emit_finish_reason_chunk(
                response_id,
                created,
                model_name,
                tool_finish_reason or finish_reason,
            )
            emitted_finish_reason = True
            continue

        content = choices[0].get("delta", {}).get("content")
        if not content:
            continue

        if tool_processor is not None:
            for event_type, value in tool_processor.ingest(content):
                if event_type == "content" and value:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": value},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                elif event_type == "tool":
                    if isinstance(value, dict):
                        tool_call = {
                            "id": f"call_{uuid.uuid4().hex}",
                            "name": value["name"],
                            "arguments": value["arguments"],
                        }
                        yield _emit_tool_call_chunk(
                            response_id, created, model_name, tool_call
                        )
                        tool_finish_reason = "tool_calls"
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

            try:
                body = apply_tool_compatibility(body)
            except ValueError as exc:
                return _json_response(400, {"error": str(exc)})

            prompt = extract_openai_prompt(body)
            if not prompt and not (body.get("messages") or []):
                return _json_response(400, {"error": "Empty prompt"})

            req = chat_request_model(**build_internal_chat_request(body, prompt))

            print(f"[OpenAI] Request: {req}")

            prompt_formatted = format_openai_prompt(body, req.enable_think)

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = req.model or os.path.basename(f"{engine.args.MODEL_NAME}")

            if req.stream:
                return StreamingResponse(
                    _stream_openai_chunks(
                        engine,
                        req,
                        body,
                        prompt_formatted,
                        response_id,
                        created,
                        model_name,
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
        except Exception as exc:
            print(f"[ERROR] /openai/v1/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

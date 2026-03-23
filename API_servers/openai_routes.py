import json
import os
import time
import traceback
import uuid

from robyn import Response, StreamingResponse

from API_servers.openai_adapter import (
    build_internal_chat_request,
    build_openai_message_response,
    build_openai_usage,
    extract_openai_prompt,
    format_openai_prompt,
    format_openai_state_prompt,
)
from state_manager.state_pool import get_state_manager


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


def _get_or_init_state(engine, session_id: str):
    state_manager = get_state_manager()
    state = state_manager.get_state(session_id)
    if state is None:
        state = engine.model.generate_zero_state(0)
        state_manager.put_state(session_id, state)
        print(f"[INIT] Created new state for session: {session_id}")
    else:
        print(f"[REUSE] Reusing existing state for session: {session_id}")
    return state_manager, state


async def _stream_state_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    session_id: str,
    response_id: str,
    created: int,
    model_name: str,
):
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

    state_manager, state = _get_or_init_state(engine, session_id)
    async for item in engine.batch_infer_stream_state(
        prompts=[prompt_formatted],
        state=state,
        max_length=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        alpha_presence=req.alpha_presence,
        alpha_frequency=req.alpha_frequency,
        alpha_decay=req.alpha_decay,
        stop_tokens=req.stop_tokens,
        chunk_size=req.chunk_size,
        session_id=session_id,
        state_manager=state_manager,
    ):
        if not item.startswith("data: "):
            continue

        payload = item[6:]
        if payload == "[DONE]":
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            break

        chunk_payload = json.loads(payload)
        choices = chunk_payload.get("choices") or []
        if not choices:
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


async def _stream_stateless_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
):
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

    saw_done = False
    emitted_finish_reason = False
    use_graph_stream = hasattr(engine, "graph_infer_stream") and not getattr(
        engine, "rocm_flag", False
    )
    if use_graph_stream:
        async for item in engine.graph_infer_stream(
            inputs=[prompt_formatted],
            stop_tokens=req.stop_tokens,
            max_generate_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            chunk_size=req.chunk_size,
        ):
            if not isinstance(item, str) or not item.startswith("data: "):
                continue

            payload = item[6:]
            if payload == "[DONE]":
                saw_done = True
                if not emitted_finish_reason:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                break

            chunk_payload = json.loads(payload)
            choices = chunk_payload.get("choices") or []
            if not choices:
                continue

            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                emitted_finish_reason = True
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
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
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
    else:
        async for item in engine.dynamic_batch_infer_stream(
            prompt=prompt_formatted,
            max_generate_tokens=req.max_tokens,
            stop_tokens=req.stop_tokens,
            pad_zero=req.pad_zero,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            chunk_size=req.chunk_size,
        ):
            if item["type"] == "delta" and item["text"]:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item["text"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif item["type"] == "done":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": item["finish_reason"],
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                break

    yield "data: [DONE]\n\n"


def register_openai_routes(app, engine, password, chat_request_model):
    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request):
        try:
            body = json.loads(request.body)
            auth_error = _check_openai_auth(request, body, password)
            if auth_error is not None:
                return auth_error

            prompt = extract_openai_prompt(body)
            if not prompt and not (body.get("messages") or []):
                return _json_response(400, {"error": "Empty prompt"})

            req = chat_request_model(**build_internal_chat_request(body, prompt))

            print(f"[OpenAI] Request: {req}")

            if req.session_id:
                try:
                    prompt_formatted = format_openai_state_prompt(
                        body, req.enable_think
                    )
                except ValueError as exc:
                    return _json_response(400, {"error": str(exc)})
            else:
                prompt_formatted = format_openai_prompt(body, req.enable_think)

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = req.model or os.path.basename(f"{engine.args.MODEL_NAME}")

            if req.stream:
                return StreamingResponse(
                    _stream_state_openai_chunks(
                        engine,
                        req,
                        prompt_formatted,
                        req.session_id,
                        response_id,
                        created,
                        model_name,
                    )
                    if req.session_id
                    else _stream_stateless_openai_chunks(
                        engine,
                        req,
                        prompt_formatted,
                        response_id,
                        created,
                        model_name,
                    ),
                    media_type="text/event-stream",
                )

            if req.session_id:
                state_manager, state = _get_or_init_state(engine, req.session_id)
                results = engine.batch_generate_state(
                    prompts=[prompt_formatted],
                    state=state,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                )
                state_manager.put_state(req.session_id, state)
                result_text = results[0] if results else ""
                finish_reason = "stop"
            else:
                result_text, finish_reason = await engine.dynamic_batch_generate(
                    prompt=prompt_formatted,
                    max_generate_tokens=req.max_tokens,
                    stop_tokens=req.stop_tokens,
                    pad_zero=req.pad_zero,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    chunk_size=req.chunk_size,
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

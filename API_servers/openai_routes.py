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
)


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

            prompt_formatted = format_openai_prompt(body, req.enable_think)

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = req.model or os.path.basename(f"{engine.args.MODEL_NAME}")

            if req.stream:

                async def stream_openai_chunks():
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

                return StreamingResponse(
                    stream_openai_chunks(), media_type="text/event-stream"
                )

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

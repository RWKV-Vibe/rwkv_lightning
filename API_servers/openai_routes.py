import json
import os
import time
import traceback
import uuid
from typing import Any

from pydantic import ValidationError
from robyn import Response, StreamingResponse

from infer.xgrammar_utils import has_xgrammar, normalize_response_format
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


def _format_assistant_tool_calls(message: dict) -> str:
    tool_calls = message.get("tool_calls") or []
    formatted = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if name:
            formatted.append(f"{name}({arguments or ''})")
    return "; ".join(formatted)


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
            tool_call_text = _format_assistant_tool_calls(message)
            if tool_call_text:
                transcript_parts.append(f"Assistant Tool Calls: {tool_call_text}")
            continue

        if role == "tool":
            if content:
                transcript_parts.append(f"Tool: {content}")
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

    response_format = normalize_response_format(body.get("response_format"))

    return {
        "model": body.get("model", "rwkv7"),
        "runtime": body.get("runtime"),
        "scheduler": body.get("scheduler", "auto"),
        "n": body.get("n", 1),
        "contents": [prompt],
        "messages": body.get("messages", []),
        "system": body.get("system"),
        "max_tokens": _resolve_openai_max_tokens(body),
        "stop_tokens": _normalize_openai_stop(
            body.get("stop") if "stop" in body else body.get("stop_tokens")
        ),
        "temperature": body.get("temperature", 1.0),
        "top_k": body.get("top_k", 20),
        "top_p": body.get("top_p", 0.6),
        "stream": stream,
        "pad_zero": body.get("pad_zero", False),
        "alpha_presence": body.get(
            "alpha_presence", body.get("presence_penalty", 0.0)
        ),
        "alpha_frequency": body.get(
            "alpha_frequency", body.get("frequency_penalty", 0.0)
        ),
        "alpha_decay": body.get("alpha_decay", 0.996),
        "enable_think": body.get("enable_think", False),
        "chunk_size": chunk_size,
        "password": body.get("password"),
        "session_id": body.get("session_id"),
        "use_prefix_cache": body.get("use_prefix_cache", True),
        "seed": body.get("seed"),
        "response_format": response_format,
        "stream_options": body.get("stream_options"),
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
    index: int = 0,
) -> str:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": index,
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


def _openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
):
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return _json_response(status_code, payload)


def _normalize_openai_stop(stop_value):
    if stop_value is None:
        return ["\nUser:"]
    if isinstance(stop_value, str):
        return [stop_value]
    if isinstance(stop_value, list) and all(isinstance(item, str) for item in stop_value):
        return stop_value
    raise ValueError("'stop' must be a string or a list of strings")


def _resolve_openai_max_tokens(body: dict) -> int:
    if body.get("max_tokens") is not None:
        return int(body["max_tokens"])
    if body.get("max_completion_tokens") is not None:
        return int(body["max_completion_tokens"])
    return 4096


def _normalize_openai_n(body: dict) -> int:
    value = int(body.get("n", 1))
    if value < 1:
        raise ValueError("Field 'n' must be >= 1")
    return value


def _normalize_openai_logprobs(body: dict) -> tuple[bool, int]:
    logprobs_enabled = bool(body.get("logprobs", False))
    top_logprobs = int(body.get("top_logprobs", 0) or 0)
    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("Field 'top_logprobs' must be between 0 and 20")
    if top_logprobs and not logprobs_enabled:
        raise ValueError("Field 'top_logprobs' requires 'logprobs=true'")
    return logprobs_enabled, top_logprobs


def _normalize_tools(body: dict) -> list[dict]:
    tools = body.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("Field 'tools' must be a list")

    normalized = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"Tool at index {index} must be an object with type='function'")
        function = tool.get("function") or {}
        name = str(function.get("name", "")).strip()
        if not name:
            raise ValueError(f"Tool at index {index} is missing function.name")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {"type": "object"},
                },
            }
        )
    return normalized


def _normalize_tool_choice(body: dict, tools: list[dict]):
    tool_choice = body.get("tool_choice", "auto")
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized not in {"auto", "none", "required"}:
            raise ValueError("Field 'tool_choice' must be one of: auto, none, required")
        return normalized
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("Object tool_choice currently only supports type='function'")
        function = tool_choice.get("function") or {}
        name = str(function.get("name", "")).strip()
        if not name:
            raise ValueError("Object tool_choice requires function.name")
        if name not in {tool["function"]["name"] for tool in tools}:
            raise ValueError(f"tool_choice function '{name}' is not present in tools")
        return {"type": "function", "function": {"name": name}}
    raise ValueError("Field 'tool_choice' must be a string or function object")


def _tool_choice_requires_tool_call(tool_choice, tools: list[dict]) -> bool:
    if not tools:
        return False
    if tool_choice == "none":
        return False
    return True


def _tool_names_for_choice(tool_choice, tools: list[dict]) -> list[str]:
    if isinstance(tool_choice, dict):
        return [tool_choice["function"]["name"]]
    return [tool["function"]["name"] for tool in tools]


def _build_tool_response_format(tools: list[dict], tool_choice) -> dict:
    allowed_names = _tool_names_for_choice(tool_choice, tools)
    one_of_items = []
    for tool in tools:
        name = tool["function"]["name"]
        if name not in allowed_names:
            continue
        parameters_schema = tool["function"].get("parameters") or {"type": "object"}
        if not isinstance(parameters_schema, dict):
            parameters_schema = {"type": "object"}
        one_of_items.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": parameters_schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not one_of_items:
        raise ValueError("No valid tools available for tool_choice")

    schema = {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": one_of_items[0] if len(one_of_items) == 1 else {"oneOf": one_of_items},
            }
        },
        "required": ["tool_calls"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_call_response",
            "strict": True,
            "schema": schema,
        },
    }


def _build_tool_instruction_block(tools: list[dict], tool_choice) -> str:
    allowed_names = set(_tool_names_for_choice(tool_choice, tools))
    lines = [
        "You must respond by selecting a tool call in JSON.",
        "Return only a JSON object with the shape {'tool_calls': [{'name': ..., 'arguments': {...}}] }.",
        "Do not answer in natural language.",
        "Available tools:",
    ]
    for tool in tools:
        name = tool["function"]["name"]
        if name not in allowed_names:
            continue
        description = tool["function"].get("description") or ""
        params = json.dumps(tool["function"].get("parameters") or {"type": "object"}, ensure_ascii=False)
        lines.append(f"- {name}: {description}")
        lines.append(f"  parameters schema: {params}")
    if isinstance(tool_choice, dict):
        lines.append(f"You must call exactly the function '{tool_choice['function']['name']}'.")
    return "\n".join(lines)


def _inject_instruction_before_assistant(prompt_formatted: str, instruction_block: str) -> str:
    marker = "\n\nAssistant:"
    idx = prompt_formatted.rfind(marker)
    if idx == -1:
        return f"{prompt_formatted}\n\n{instruction_block}"
    return f"{prompt_formatted[:idx]}\n\n{instruction_block}{prompt_formatted[idx:]}"


def _build_openai_logprobs_payload(logprob_entries):
    return {"content": logprob_entries, "refusal": None}


def _build_openai_usage_many(tokenizer, prompt_text: str, completion_texts: list[str]) -> dict:
    prompt_tokens = len(tokenizer.encode(prompt_text))
    completion_tokens = sum(len(tokenizer.encode(text)) for text in completion_texts if text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _parse_tool_response(result_text: str, tool_choice) -> tuple[dict[str, Any], str]:
    payload = json.loads(result_text)
    tool_calls_payload = payload.get("tool_calls")
    if not isinstance(tool_calls_payload, list) or not tool_calls_payload:
        raise ValueError("Tool calling response did not contain a non-empty tool_calls array")

    tool_calls = []
    for tool_call in tool_calls_payload:
        if not isinstance(tool_call, dict):
            raise ValueError("Tool calling response contains an invalid tool_call entry")
        name = str(tool_call.get("name", "")).strip()
        if not name:
            raise ValueError("Tool calling response is missing tool_call.name")
        if isinstance(tool_choice, dict) and name != tool_choice["function"]["name"]:
            raise ValueError(
                f"Tool calling response used '{name}' but tool_choice requires '{tool_choice['function']['name']}'"
            )
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
            }
        )

    return {"role": "assistant", "content": None, "tool_calls": tool_calls}, "tool_calls"


def _validate_openai_features(body: dict):
    _normalize_openai_n(body)
    logprobs_enabled, _ = _normalize_openai_logprobs(body)
    tools = _normalize_tools(body)
    tool_choice = _normalize_tool_choice(body, tools)

    unsupported_fields = {
        "functions": body.get("functions"),
        "function_call": body.get("function_call"),
        "audio": body.get("audio"),
        "modalities": body.get("modalities"),
    }
    for field_name, value in unsupported_fields.items():
        if value not in (None, [], {}, "none"):
            raise ValueError(f"OpenAI field '{field_name}' is not supported yet")

    if body.get("stream") and logprobs_enabled:
        raise ValueError("Streaming logprobs are not supported yet")

    if body.get("stream") and _tool_choice_requires_tool_call(tool_choice, tools):
        raise ValueError("Streaming tool calls are not supported yet")

    response_format = normalize_response_format(body.get("response_format"))
    if response_format and tools:
        raise ValueError("response_format cannot be combined with tools")
    if response_format and response_format.get("type") in {"json_object", "json_schema"} and not has_xgrammar():
        raise RuntimeError(
            "response_format JSON modes require the optional 'xgrammar' package to be installed"
        )
    if tools and not has_xgrammar():
        raise RuntimeError("Tool calling requires the optional 'xgrammar' package to be installed")
    return {
        "response_format": response_format,
        "tools": tools,
        "tool_choice": tool_choice,
        "n": _normalize_openai_n(body),
        "logprobs": logprobs_enabled,
        "top_logprobs": _normalize_openai_logprobs(body)[1],
    }


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
    return _openai_error_response(
        401,
        "Unauthorized: invalid or missing password",
        error_type="authentication_error",
        code="invalid_api_key",
    )


async def _stream_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
    prefix_cache_manager=None,
    grammar_constraint=None,
    choice_index: int = 0,
    usage_collector: list[str] | None = None,
    seed=None,
):
    emitted_finish_reason = False
    accumulated_text = []
    start_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": choice_index,
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
        seed=seed,
        grammar_constraint=grammar_constraint,
    ):
        payload = _extract_sse_payload(item)
        if payload is None:
            continue

        if payload == "[DONE]":
            if not emitted_finish_reason:
                yield _emit_finish_reason_chunk(
                    response_id, created, model_name, "stop", choice_index
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
                choice_index,
            )
            emitted_finish_reason = True
            continue

        content = choices[0].get("delta", {}).get("content")
        if not content:
            continue
        accumulated_text.append(content)
        if usage_collector is not None:
            usage_collector.append(content)

        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    include_usage = bool((req.stream_options or {}).get("include_usage"))
    if include_usage and usage_collector is None:
        usage_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [],
            "usage": build_openai_usage(
                engine.tokenizer, prompt_formatted, "".join(accumulated_text)
            ),
        }
        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def _choice_seed(base_seed, choice_index: int):
    if base_seed is None:
        return None
    return int(base_seed) + int(choice_index)


def _tool_prompt_and_response_format(prompt_formatted: str, tools: list[dict], tool_choice):
    instruction_block = _build_tool_instruction_block(tools, tool_choice)
    prompt_with_tools = _inject_instruction_before_assistant(
        prompt_formatted, instruction_block
    )
    return prompt_with_tools, _build_tool_response_format(tools, tool_choice)


async def _generate_openai_choice(
    engine,
    req,
    prompt_formatted: str,
    prefix_cache_manager,
    *,
    choice_index: int,
    body: dict,
    grammar_constraint,
    top_logprobs: int,
    tool_choice=None,
):
    choice_seed = _choice_seed(req.seed, choice_index)
    if top_logprobs > 0:
        result_text, finish_reason, logprob_entries = await engine.singe_infer_with_metadata(
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
            seed=choice_seed,
            grammar_constraint=grammar_constraint,
            top_logprobs=top_logprobs,
        )
    else:
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
            seed=choice_seed,
            grammar_constraint=grammar_constraint,
        )
        logprob_entries = None

    if tool_choice is not None:
        message, response_finish_reason = _parse_tool_response(result_text, tool_choice)
    else:
        message, response_finish_reason = build_openai_message_response(
            result_text, finish_reason, body
        )

    choice = {
        "index": choice_index,
        "message": message,
        "finish_reason": response_finish_reason,
    }
    if logprob_entries is not None:
        choice["logprobs"] = _build_openai_logprobs_payload(logprob_entries)

    return choice, result_text


def register_openai_routes(app, engine, password, chat_request_model):
    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request):
        try:
            body = json.loads(request.body)
            auth_error = _check_openai_auth(request, body, password)
            if auth_error is not None:
                return auth_error

            validated_features = _validate_openai_features(body)

            served_model_id = getattr(
                engine.args,
                "served_model_id",
                os.path.basename(f"{engine.args.MODEL_NAME}"),
            )
            served_aliases = set(getattr(engine.args, "model_aliases", (served_model_id,)))
            requested_model = str(body.get("model", "rwkv7")).strip()
            if requested_model and requested_model not in served_aliases:
                return _openai_error_response(
                    404,
                    f"Model '{requested_model}' is not served by this process",
                    code="model_not_found",
                )

            requested_runtime = body.get("runtime")
            served_runtime = getattr(engine.args, "runtime", "fp16")
            if requested_runtime and str(requested_runtime).strip().lower() != served_runtime:
                return _openai_error_response(
                    400,
                    f"Runtime '{requested_runtime}' is not available on this server",
                    param="runtime",
                )

            prompt = extract_openai_prompt(body)
            if not prompt and not (body.get("messages") or []):
                return _openai_error_response(400, "Empty prompt")

            req = chat_request_model(**build_internal_chat_request(body, prompt))

            # print(f"[OpenAI] Request: {req}")

            prompt_formatted = format_openai_prompt(body, req.enable_think)
            tools = validated_features["tools"]
            tool_choice = validated_features["tool_choice"]
            if _tool_choice_requires_tool_call(tool_choice, tools):
                prompt_formatted, req.response_format = _tool_prompt_and_response_format(
                    prompt_formatted, tools, tool_choice
                )

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = served_model_id
            prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None
            choice_count = validated_features["n"]
            top_logprobs = validated_features["top_logprobs"] if validated_features["logprobs"] else 0

            if req.stream:
                async def stream_choices():
                    usage_collector = [] if bool((req.stream_options or {}).get("include_usage")) else None
                    for choice_index in range(choice_count):
                        grammar_constraint = engine.build_response_format_constraint(req.response_format)
                        async for chunk in _stream_openai_chunks(
                            engine,
                            req,
                            prompt_formatted,
                            response_id,
                            created,
                            model_name,
                            prefix_cache_manager,
                            grammar_constraint,
                            choice_index=choice_index,
                            usage_collector=usage_collector,
                            seed=_choice_seed(req.seed, choice_index),
                        ):
                            if chunk == "data: [DONE]\n\n":
                                continue
                            yield chunk
                    if usage_collector is not None:
                        usage_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": build_openai_usage_many(
                                engine.tokenizer, prompt_formatted, ["".join(usage_collector)]
                            ),
                        }
                        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_choices(), media_type="text/event-stream")

            choices = []
            completion_texts = []
            for choice_index in range(choice_count):
                grammar_constraint = engine.build_response_format_constraint(req.response_format)
                choice, result_text = await _generate_openai_choice(
                    engine,
                    req,
                    prompt_formatted,
                    prefix_cache_manager,
                    choice_index=choice_index,
                    body=body,
                    grammar_constraint=grammar_constraint,
                    top_logprobs=top_logprobs,
                    tool_choice=tool_choice if _tool_choice_requires_tool_call(tool_choice, tools) else None,
                )
                choices.append(choice)
                completion_texts.append(result_text)

            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": choices,
                "usage": build_openai_usage_many(
                    engine.tokenizer, prompt_formatted, completion_texts
                ),
            }
            return _json_response(200, response)
        except json.JSONDecodeError as exc:
            return _openai_error_response(400, f"Invalid JSON: {str(exc)}")
        except ValidationError as exc:
            return _openai_error_response(400, str(exc), param="request")
        except ValueError as exc:
            return _openai_error_response(400, str(exc))
        except RuntimeError as exc:
            return _openai_error_response(400, str(exc))
        except Exception as exc:
            print(f"[ERROR] /openai/v1/chat/completions: {traceback.format_exc()}")
            return _openai_error_response(
                500,
                str(exc),
                error_type="server_error",
                code="internal_error",
            )

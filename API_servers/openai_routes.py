import json
import math
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
            elif isinstance(item, dict):
                raise ValueError(
                    "OpenAI chat completions currently only support text content parts"
                )
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


def _normalize_legacy_functions(body: dict) -> list[dict] | None:
    functions = body.get("functions")
    if functions in (None, [], {}):
        return None
    if not isinstance(functions, list):
        raise ValueError("Field 'functions' must be a list")

    tools = []
    for index, function in enumerate(functions):
        if not isinstance(function, dict):
            raise ValueError(f"Function at index {index} must be an object")
        name = str(function.get("name", "")).strip()
        if not name:
            raise ValueError(f"Function at index {index} is missing name")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {"type": "object"},
                },
            }
        )
    return tools


def _format_assistant_tool_calls(message: dict) -> str:
    tool_calls = message.get("tool_calls") or []
    formatted = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = str(tool_call.get("id", "")).strip()
        function = tool_call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if name:
            prefix = f"[{tool_call_id}] " if tool_call_id else ""
            formatted.append(f"{prefix}{name}({arguments or ''})")
    return "; ".join(formatted)


def _assistant_tool_call_lookup(message: dict) -> dict[str, str]:
    lookup = {}
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = str(tool_call.get("id", "")).strip()
        if not tool_call_id:
            continue
        function = tool_call.get("function") or {}
        tool_name = str(function.get("name", "")).strip() or "function"
        lookup[tool_call_id] = tool_name
    return lookup


def _collect_openai_prompt_parts(body: dict) -> tuple[str, list[str], dict[str, str]]:
    messages = body.get("messages") or []
    contents = body.get("contents") or []

    system_parts = []
    transcript_parts = []
    known_tool_calls = {}

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
            known_tool_calls.update(_assistant_tool_call_lookup(message))
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            if not tool_call_id:
                raise ValueError("Tool messages require tool_call_id")
            tool_name = known_tool_calls.get(tool_call_id)
            if tool_name is None:
                raise ValueError(f"Unknown tool_call_id '{tool_call_id}' in tool message")
            if content:
                transcript_parts.append(
                    f"Tool Result [{tool_call_id}] {tool_name}: {content}"
                )
            continue

        if role == "function":
            if content:
                transcript_parts.append(f"Tool: {content}")
            continue

    if contents:
        content_prompt = _sanitize_text_block(contents[0])
        if content_prompt:
            transcript_parts.append(f"User: {content_prompt}")

    system_text = "\n".join(part for part in system_parts if part).strip()
    transcript_parts = [part for part in transcript_parts if part]
    return system_text, transcript_parts, known_tool_calls


def extract_openai_prompt(body: dict) -> str:
    system_text, transcript_parts, _ = _collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(system_text)
    prompt_parts.extend(transcript_parts)
    return "\n".join(part for part in prompt_parts if part).strip()


def _reasoning_effort_prompt_suffix(reasoning_effort: str | None, enable_think: bool) -> str:
    if reasoning_effort:
        effort_labels = {
            "none": "",
            "minimal": "(think minimally)",
            "low": "(think a bit)",
            "medium": "(think)",
            "high": "(think a lot)",
            "xhigh": "(think extremely carefully)",
        }
        return effort_labels[reasoning_effort]
    return "(think)" if enable_think else ""


def format_openai_prompt(
    body: dict, enable_think: bool, reasoning_effort: str | None = None
) -> str:
    system_text, transcript_parts, _ = _collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(f"System: {system_text}")
    prompt_parts.extend(transcript_parts)

    prompt_text = "\n\n".join(part for part in prompt_parts if part).strip()
    if not prompt_text:
        raise ValueError("OpenAI chat completions require system or user text")
    think_suffix = _reasoning_effort_prompt_suffix(reasoning_effort, enable_think)
    if think_suffix:
        return f"{prompt_text} {think_suffix}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>\n"


def format_openai_state_prompt(
    body: dict, enable_think: bool, reasoning_effort: str | None = None
) -> str:
    contents = body.get("contents") or []
    if len(contents) > 1:
        raise ValueError("State mode only supports a single contents item")
    return format_openai_prompt(body, enable_think, reasoning_effort)


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
    stop_tokens = []
    if "stop" in body or "stop_tokens" in body:
        stop_tokens = _normalize_openai_stop(
            body.get("stop") if "stop" in body else body.get("stop_tokens")
        )

    return {
        "model": body.get("model", "rwkv7"),
        "runtime": body.get("runtime"),
        "scheduler": body.get("scheduler", "auto"),
        "n": body.get("n", 1),
        "contents": [prompt],
        "messages": body.get("messages", []),
        "system": body.get("system"),
        "max_tokens": _resolve_openai_max_tokens(body),
        "stop_tokens": stop_tokens,
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
        "enable_think": body.get("enable_think", False)
        or body.get("reasoning_effort") not in (None, "none"),
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


def _resolve_openai_model_created(engine) -> int:
    model_root = getattr(engine.args, "MODEL_NAME", "")
    if model_root:
        candidate = f"{model_root}.pth"
        if os.path.exists(candidate):
            return int(os.path.getmtime(candidate))
        if os.path.exists(model_root):
            return int(os.path.getmtime(model_root))
    return int(time.time())


def _build_openai_model_payload(model_id: str, created: int) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "rwkv_lightning",
    }


def _build_openai_models_response(model_ids: list[str], created: int) -> dict:
    unique_ids = []
    seen = set()
    for model_id in model_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        unique_ids.append(model_id)
    return {
        "object": "list",
        "data": [_build_openai_model_payload(model_id, created) for model_id in unique_ids],
    }


def _emit_finish_reason_chunk(
    response_id: str,
    created: int,
    model_name: str,
    finish_reason: str,
    index: int = 0,
    service_tier: str | None = None,
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
    if service_tier is not None:
        chunk["service_tier"] = service_tier
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


def _normalize_reasoning_effort(body: dict) -> str | None:
    value = body.get("reasoning_effort")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    if normalized not in allowed:
        raise ValueError(
            "Field 'reasoning_effort' must be one of: none, minimal, low, medium, high, xhigh"
        )
    return normalized


def _normalize_parallel_tool_calls(body: dict) -> bool:
    value = body.get("parallel_tool_calls")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError("Field 'parallel_tool_calls' must be a boolean")
    return value


def _normalize_logit_bias(body: dict, vocab_size: int) -> dict[int, float] | None:
    value = body.get("logit_bias")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Field 'logit_bias' must be an object")

    normalized = {}
    for token_id_raw, bias_raw in value.items():
        try:
            token_id = int(token_id_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Field 'logit_bias' keys must be token ids") from exc
        if token_id < 0 or token_id >= int(vocab_size):
            raise ValueError(
                f"Field 'logit_bias' contains out-of-range token id {token_id}"
            )
        try:
            bias = float(bias_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Field 'logit_bias' values must be numbers") from exc
        if not math.isfinite(bias):
            raise ValueError("Field 'logit_bias' values must be finite numbers")
        if bias < -100 or bias > 100:
            raise ValueError("Field 'logit_bias' values must be between -100 and 100")
        normalized[token_id] = bias
    return normalized


def _normalize_metadata(body: dict) -> dict[str, str] | None:
    value = body.get("metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Field 'metadata' must be an object")
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Field 'metadata' keys must be strings")
        if not isinstance(item, str):
            raise ValueError("Field 'metadata' values must be strings")
        normalized[key] = item
    return normalized


def _normalize_store(body: dict) -> bool | None:
    value = body.get("store")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Field 'store' must be a boolean")
    return value


def _normalize_user(body: dict) -> str | None:
    value = body.get("user")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Field 'user' must be a string")
    return value


def _normalize_service_tier(body: dict) -> str | None:
    value = body.get("service_tier")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    allowed = {"auto", "default", "flex", "scale", "priority"}
    if normalized not in allowed:
        raise ValueError(
            "Field 'service_tier' must be one of: auto, default, flex, scale, priority"
        )
    return normalized


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


def _normalize_openai_function_call(body: dict, tools: list[dict]):
    function_call = body.get("function_call")
    if function_call in (None, {}, "none", "auto"):
        return function_call
    if not tools:
        raise ValueError("Field 'function_call' requires 'functions' or 'tools'")
    if isinstance(function_call, str):
        normalized = function_call.strip().lower()
        if normalized != "required":
            raise ValueError("Field 'function_call' must be 'auto', 'none', or a function object")
        return "required"
    if isinstance(function_call, dict):
        name = str(function_call.get("name", "")).strip()
        if not name:
            raise ValueError("Field 'function_call' requires name")
        if name not in {tool["function"]["name"] for tool in tools}:
            raise ValueError(f"function_call '{name}' is not present in functions/tools")
        return {"type": "function", "function": {"name": name}}
    raise ValueError("Field 'function_call' must be a string or function object")


def _normalize_tool_choice(body: dict, tools: list[dict]):
    tool_choice = body.get("tool_choice")
    if tool_choice is None:
        return "auto" if tools else "none"
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
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        if tool_choice in {"none", "auto"}:
            return False
        return tool_choice == "required"
    return tool_choice == "required" or isinstance(tool_choice, dict)


def _tool_names_for_choice(tool_choice, tools: list[dict]) -> list[str]:
    if isinstance(tool_choice, dict):
        return [tool_choice["function"]["name"]]
    return [tool["function"]["name"] for tool in tools]


def _build_tool_response_format(
    tools: list[dict], tool_choice, parallel_tool_calls: bool
) -> dict:
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
                    "tool_name": {"const": name},
                    "arguments": parameters_schema,
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not one_of_items:
        raise ValueError("No valid tools available for tool_choice")
    tool_call_schema = (
        one_of_items[0] if len(one_of_items) == 1 else {"oneOf": one_of_items}
    )
    min_items = 1 if _tool_choice_requires_tool_call(tool_choice, tools) else 0
    max_items = None if parallel_tool_calls else 1
    array_schema = {
        "type": "array",
        "items": tool_call_schema,
        "minItems": min_items,
    }
    if max_items is not None:
        array_schema["maxItems"] = max_items
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_call_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"tool_calls": array_schema},
                "required": ["tool_calls"],
                "additionalProperties": False,
            },
        },
    }


def _build_tool_instruction_block(
    tools: list[dict], tool_choice, parallel_tool_calls: bool
) -> str:
    allowed_names = set(_tool_names_for_choice(tool_choice, tools))
    lines = [
        "You must respond in JSON.",
        "Return only a JSON object with the shape {'tool_calls': [{'tool_name': '...', 'arguments': {...}}]}.",
        "Do not answer in natural language.",
        (
            "When multiple tool calls are needed, include them all in tool_calls in the order they should be executed."
            if parallel_tool_calls
            else "Return at most one tool call in tool_calls."
        ),
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


def _build_tool_router_instruction_block(tools: list[dict]) -> str:
    lines = [
        "Decide whether the assistant should answer directly or call a tool.",
        "Return only JSON.",
        "Use {'mode':'answer'} when no tool is necessary.",
        "Use {'mode':'tool'} when tools are necessary.",
        "Use 'answer' for simple arithmetic, general knowledge, summarization, or any request solvable from the prompt alone.",
        "Use 'tool' only when the request explicitly needs one of the provided tools or external side effects/data.",
        "Available tools:",
    ]
    for tool in tools:
        lines.append(
            f"- {tool['function']['name']}: {tool['function'].get('description') or ''}"
        )
    return "\n".join(lines)


def _build_tool_router_response_format(tools: list[dict]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_router_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["answer", "tool"]},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    }


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


def _scan_json_string_end(text: str, start_idx: int):
    escaped = False
    idx = start_idx + 1
    while idx < len(text):
        ch = text[idx]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return idx
        idx += 1
    return None


def _skip_json_whitespace(text: str, idx: int) -> int:
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    return idx


def _extract_json_string_field(text: str, field_name: str):
    marker = f'"{field_name}"'
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return None
    colon_idx = text.find(":", marker_idx + len(marker))
    if colon_idx == -1:
        return None
    value_idx = _skip_json_whitespace(text, colon_idx + 1)
    if value_idx >= len(text) or text[value_idx] != '"':
        return None
    end_idx = _scan_json_string_end(text, value_idx)
    if end_idx is None:
        return None
    return json.loads(text[value_idx : end_idx + 1])


def _find_json_value_start_for_field(text: str, field_name: str):
    marker = f'"{field_name}"'
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return None
    colon_idx = text.find(":", marker_idx + len(marker))
    if colon_idx == -1:
        return None
    value_idx = _skip_json_whitespace(text, colon_idx + 1)
    if value_idx >= len(text):
        return None
    return value_idx


def _find_json_value_end(text: str, start_idx: int):
    if start_idx is None or start_idx >= len(text):
        return None

    first = text[start_idx]
    if first == '"':
        end_idx = _scan_json_string_end(text, start_idx)
        return None if end_idx is None else end_idx + 1

    if first in "[{":
        stack = [first]
        idx = start_idx + 1
        in_string = False
        escaped = False
        closing = {"{": "}", "[": "]"}
        while idx < len(text):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        return None
                    opener = stack.pop()
                    if closing[opener] != ch:
                        return None
                    if not stack:
                        return idx + 1
            idx += 1
        return None

    for idx in range(start_idx, len(text)):
        if text[idx] in ",}]\n\r\t ":
            return idx
    return len(text)


class _StreamingToolCallState:
    def __init__(self, tool_index: int):
        self.tool_index = tool_index
        self.tool_call_id = f"call_{uuid.uuid4().hex}"
        self.tool_name = None
        self.name_emitted = False
        self.arguments_start = None
        self.arguments_emitted_upto = None

    def ingest(self, item_buffer: str):
        deltas = []
        if self.tool_name is None:
            self.tool_name = _extract_json_string_field(item_buffer, "tool_name")

        if self.tool_name is not None and not self.name_emitted:
            self.name_emitted = True
            deltas.append(
                {
                    "index": self.tool_index,
                    "id": self.tool_call_id,
                    "type": "function",
                    "function": {"name": self.tool_name, "arguments": ""},
                }
            )

        current_arg_start = _find_json_value_start_for_field(item_buffer, "arguments")
        if current_arg_start is not None and self.arguments_start is None:
            self.arguments_start = current_arg_start
            self.arguments_emitted_upto = current_arg_start

        if self.arguments_start is not None and self.name_emitted:
            value_end = _find_json_value_end(item_buffer, self.arguments_start)
            emit_upto = len(item_buffer) if value_end is None else value_end
            if emit_upto > self.arguments_emitted_upto:
                fragment = item_buffer[self.arguments_emitted_upto : emit_upto]
                self.arguments_emitted_upto = emit_upto
                if fragment:
                    deltas.append(
                        {
                            "index": self.tool_index,
                            "function": {"arguments": fragment},
                        }
                    )
        return deltas


class _IncrementalToolCallAssembler:
    def __init__(self):
        self.buffer = ""
        self.array_start = None
        self.states = []

    def _locate_array_start(self):
        if self.array_start is not None:
            return self.array_start
        self.array_start = _find_json_value_start_for_field(self.buffer, "tool_calls")
        return self.array_start

    def _iter_item_buffers(self):
        array_start = self._locate_array_start()
        if array_start is None or array_start >= len(self.buffer) or self.buffer[array_start] != "[":
            return []

        item_buffers = []
        object_depth = 0
        in_string = False
        escaped = False
        current_start = None
        idx = array_start + 1
        while idx < len(self.buffer):
            ch = self.buffer[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    object_depth += 1
                    if object_depth == 1:
                        current_start = idx
                elif ch == "}":
                    if object_depth > 0:
                        object_depth -= 1
                        if object_depth == 0 and current_start is not None:
                            item_buffers.append(self.buffer[current_start : idx + 1])
                            current_start = None
                elif ch == "]" and object_depth == 0:
                    break
            idx += 1

        if current_start is not None:
            item_buffers.append(self.buffer[current_start:])
        return item_buffers

    def ingest(self, content: str):
        self.buffer += content
        deltas = []
        item_buffers = self._iter_item_buffers()
        while len(self.states) < len(item_buffers):
            self.states.append(_StreamingToolCallState(len(self.states)))
        for state, item_buffer in zip(self.states, item_buffers):
            deltas.extend(state.ingest(item_buffer))
        return deltas


def _parse_tool_response(
    result_text: str, tool_choice, parallel_tool_calls: bool
) -> tuple[dict[str, Any], str]:
    payload = json.loads(result_text)
    if isinstance(payload, dict) and "tool_calls" in payload:
        tool_call_payloads = payload.get("tool_calls") or []
    elif isinstance(payload, dict) and payload.get("tool_name"):
        tool_call_payloads = [payload]
    else:
        raise ValueError("Tool calling response is missing tool_calls")

    if not isinstance(tool_call_payloads, list):
        raise ValueError("Tool calling response field 'tool_calls' must be a list")
    if not parallel_tool_calls and len(tool_call_payloads) > 1:
        raise ValueError("parallel_tool_calls=false allows at most one tool call")
    if tool_choice in {"required"} and len(tool_call_payloads) < 1:
        raise ValueError("Tool calling response must include at least one tool call")

    tool_calls = []
    for tool_payload in tool_call_payloads:
        if not isinstance(tool_payload, dict):
            raise ValueError("Each tool call must be an object")
        name = str(tool_payload.get("tool_name", "")).strip()
        if not name:
            raise ValueError("Tool calling response is missing tool_name")
        if isinstance(tool_choice, dict) and name != tool_choice["function"]["name"]:
            raise ValueError(
                f"Tool calling response used '{name}' but tool_choice requires '{tool_choice['function']['name']}'"
            )
        arguments = tool_payload.get("arguments", {})
        arguments_json = (
            arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        )
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
    reasoning_effort = _normalize_reasoning_effort(body)
    parallel_tool_calls = _normalize_parallel_tool_calls(body)
    metadata = _normalize_metadata(body)
    store = _normalize_store(body)
    user = _normalize_user(body)
    service_tier = _normalize_service_tier(body)
    legacy_tools = _normalize_legacy_functions(body)
    if legacy_tools is not None:
        if tools:
            raise ValueError("OpenAI fields 'tools' and 'functions' cannot be combined")
        tools = legacy_tools

    if body.get("function_call") is not None:
        tool_choice = _normalize_openai_function_call(body, tools)
    else:
        tool_choice = _normalize_tool_choice(body, tools)

    if tool_choice == "required" and not tools:
        raise ValueError("Field 'tool_choice'='required' requires 'tools'")

    unsupported_fields = {
        "audio": body.get("audio"),
        "modalities": body.get("modalities"),
    }
    for field_name, value in unsupported_fields.items():
        if value not in (None, [], {}, "none"):
            raise ValueError(f"OpenAI field '{field_name}' is not supported yet")

    if body.get("stream") and logprobs_enabled and tools:
        raise ValueError("Streaming logprobs together with streaming tool calls are not supported yet")

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
        "reasoning_effort": reasoning_effort,
        "parallel_tool_calls": parallel_tool_calls,
        "logit_bias": None,
        "metadata": metadata,
        "store": store,
        "user": user,
        "service_tier": service_tier,
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


def _actual_service_tier(requested_tier: str | None) -> str | None:
    if requested_tier is None:
        return None
    return "default"


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
    top_logprobs: int = 0,
    tool_choice=None,
    logit_bias=None,
    service_tier: str | None = None,
    parallel_tool_calls: bool = True,
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
    if service_tier is not None:
        start_chunk["service_tier"] = service_tier
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

    if top_logprobs > 0:
        async for event in engine.singe_infer_stream_with_metadata(
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
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
        ):
            if event["type"] == "done":
                yield _emit_finish_reason_chunk(
                    response_id,
                    created,
                    model_name,
                    event["finish_reason"],
                    choice_index,
                    service_tier,
                )
                emitted_finish_reason = True
                continue

            content = event.get("text", "")
            logprobs = event.get("logprobs", [])
            if not content and not logprobs:
                continue
            if content:
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
                        "logprobs": _build_openai_logprobs_payload(logprobs),
                        "finish_reason": None,
                    }
                ],
            }
            if service_tier is not None:
                chunk["service_tier"] = service_tier
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    else:
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
            logit_bias=logit_bias,
        ):
            payload = _extract_sse_payload(item)
            if payload is None:
                continue

            if payload == "[DONE]":
                if not emitted_finish_reason:
                    yield _emit_finish_reason_chunk(
                        response_id,
                        created,
                        model_name,
                        "stop",
                        choice_index,
                        service_tier,
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
                    service_tier,
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
            if service_tier is not None:
                chunk["service_tier"] = service_tier
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
        if service_tier is not None:
            usage_chunk["service_tier"] = service_tier
        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_openai_tool_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
    *,
    prefix_cache_manager=None,
    grammar_constraint=None,
    choice_index: int = 0,
    usage_collector: list[str] | None = None,
    seed=None,
    tool_choice=None,
    top_logprobs: int = 0,
    logit_bias=None,
    service_tier: str | None = None,
    parallel_tool_calls: bool = True,
):
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
    if service_tier is not None:
        start_chunk["service_tier"] = service_tier
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

    result_text_parts = []
    assembler = _IncrementalToolCallAssembler()
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
        logit_bias=logit_bias,
    ):
        payload = _extract_sse_payload(item)
        if payload is None or payload == "[DONE]":
            continue
        try:
            chunk_payload = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk_payload.get("choices") or []
        if not choices:
            continue
        content = choices[0].get("delta", {}).get("content")
        if not content:
            continue
        result_text_parts.append(content)
        if usage_collector is not None:
            usage_collector.append(content)
        for tool_delta in assembler.ingest(content):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {"tool_calls": [tool_delta]},
                        "finish_reason": None,
                    }
                ],
            }
            if service_tier is not None:
                chunk["service_tier"] = service_tier
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    result_text = "".join(result_text_parts)
    _, finish_reason = _parse_tool_response(
        result_text, tool_choice, parallel_tool_calls=parallel_tool_calls
    )

    yield _emit_finish_reason_chunk(
        response_id,
        created,
        model_name,
        finish_reason,
        choice_index,
        service_tier,
    )


def _choice_seed(base_seed, choice_index: int):
    if base_seed is None:
        return None
    return int(base_seed) + int(choice_index)


def _tool_prompt_and_response_format(
    prompt_formatted: str,
    tools: list[dict],
    tool_choice,
    parallel_tool_calls: bool,
):
    instruction_block = _build_tool_instruction_block(
        tools, tool_choice, parallel_tool_calls
    )
    prompt_with_tools = _inject_instruction_before_assistant(
        prompt_formatted, instruction_block
    )
    return prompt_with_tools, _build_tool_response_format(
        tools, tool_choice, parallel_tool_calls
    )


async def _resolve_auto_tool_choice(
    engine,
    req,
    prompt_formatted: str,
    prefix_cache_manager,
    tools: list[dict],
):
    router_prompt = _inject_instruction_before_assistant(
        prompt_formatted, _build_tool_router_instruction_block(tools)
    )
    router_response_format = _build_tool_router_response_format(tools)
    grammar_constraint = engine.build_response_format_constraint(router_response_format)
    router_text, _ = await engine.singe_infer(
        prompt=router_prompt,
        max_length=64,
        temperature=0.1,
        top_k=1,
        top_p=1.0,
        alpha_presence=0.0,
        alpha_frequency=0.0,
        alpha_decay=req.alpha_decay,
        stop_tokens=req.stop_tokens,
        prefix_cache_manager=prefix_cache_manager,
        seed=req.seed,
        grammar_constraint=grammar_constraint,
    )
    try:
        routing = json.loads(router_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Auto tool routing returned invalid JSON: {router_text}") from exc

    mode = str(routing.get("mode", "answer")).strip().lower()
    if mode == "answer":
        return "none"
    if mode != "tool":
        raise ValueError(f"Unsupported auto tool routing mode '{mode}'")
    return "required"


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
    logit_bias=None,
    parallel_tool_calls: bool = True,
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
            logit_bias=logit_bias,
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
            logit_bias=logit_bias,
        )
        logprob_entries = None

    if tool_choice is not None:
        message, response_finish_reason = _parse_tool_response(
            result_text, tool_choice, parallel_tool_calls
        )
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
    openai_model_created = _resolve_openai_model_created(engine)
    openai_model_aliases = list(getattr(engine.args, "model_aliases", (getattr(engine.args, "served_model_id", "rwkv7"),)))

    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request):
        try:
            body = json.loads(request.body)
            auth_error = _check_openai_auth(request, body, password)
            if auth_error is not None:
                return auth_error

            validated_features = _validate_openai_features(body)
            validated_features["logit_bias"] = _normalize_logit_bias(
                body, getattr(engine.args, "vocab_size", 65536)
            )

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

            prompt_formatted = format_openai_prompt(
                body,
                req.enable_think,
                reasoning_effort=validated_features["reasoning_effort"],
            )
            tools = validated_features["tools"]
            tool_choice = validated_features["tool_choice"]
            parallel_tool_calls = validated_features["parallel_tool_calls"]
            actual_service_tier = _actual_service_tier(
                validated_features["service_tier"]
            )
            if tools and tool_choice == "auto":
                tool_choice = await _resolve_auto_tool_choice(
                    engine,
                    req,
                    prompt_formatted,
                    prefix_cache_manager=get_state_manager() if req.use_prefix_cache else None,
                    tools=tools,
                )
            if _tool_choice_requires_tool_call(tool_choice, tools):
                prompt_formatted, req.response_format = _tool_prompt_and_response_format(
                    prompt_formatted,
                    tools,
                    tool_choice,
                    parallel_tool_calls,
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
                        stream_fn = (
                            _stream_openai_tool_chunks
                            if _tool_choice_requires_tool_call(tool_choice, tools)
                            else _stream_openai_chunks
                        )
                        async for chunk in stream_fn(
                            engine,
                            req,
                            prompt_formatted,
                            response_id,
                            created,
                            model_name,
                            prefix_cache_manager=prefix_cache_manager,
                            grammar_constraint=grammar_constraint,
                            choice_index=choice_index,
                            usage_collector=usage_collector,
                            seed=_choice_seed(req.seed, choice_index),
                            top_logprobs=top_logprobs,
                            tool_choice=tool_choice if _tool_choice_requires_tool_call(tool_choice, tools) else None,
                            logit_bias=validated_features["logit_bias"],
                            service_tier=actual_service_tier,
                            parallel_tool_calls=parallel_tool_calls,
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
                            "usage": _build_openai_usage_many(
                                engine.tokenizer, prompt_formatted, ["".join(usage_collector)]
                            ),
                        }
                        if actual_service_tier is not None:
                            usage_chunk["service_tier"] = actual_service_tier
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
                    logit_bias=validated_features["logit_bias"],
                    parallel_tool_calls=parallel_tool_calls,
                )
                choices.append(choice)
                completion_texts.append(result_text)

            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": choices,
                "usage": _build_openai_usage_many(
                    engine.tokenizer, prompt_formatted, completion_texts
                ),
            }
            if actual_service_tier is not None:
                response["service_tier"] = actual_service_tier
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

    @app.get("/openai/v1/models")
    async def openai_list_models(request):
        auth_error = _check_openai_auth(request, {}, password)
        if auth_error is not None:
            return auth_error
        return _json_response(200, _build_openai_models_response(openai_model_aliases, openai_model_created))

    @app.get("/openai/v1/models/<model_id>")
    async def openai_retrieve_model(request, model_id):
        auth_error = _check_openai_auth(request, {}, password)
        if auth_error is not None:
            return auth_error

        requested_model = str(model_id).strip()
        if requested_model not in set(openai_model_aliases):
            return _openai_error_response(
                404,
                f"Model '{requested_model}' is not served by this process",
                code="model_not_found",
            )
        return _json_response(200, _build_openai_model_payload(requested_model, openai_model_created))

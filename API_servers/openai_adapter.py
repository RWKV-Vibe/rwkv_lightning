import json
import uuid
from typing import Any, Optional


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


def extract_openai_prompt(body: dict) -> str:
    contents = body.get("contents") or []
    if contents:
        return str(contents[0])

    messages = body.get("messages") or []
    if messages:
        return normalize_message_content(messages[-1].get("content", ""))
    return ""


def _normalize_rwkv_tool_compat_mode(mode: Any) -> str:
    if mode is None:
        return "true"
    if isinstance(mode, bool):
        return "true" if mode else "false"

    normalized = str(mode).strip().lower()
    if normalized in {"true", "false"}:
        return normalized
    raise ValueError("rwkv_tool_compat_mode must be one of: true, false")


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        raise ValueError(f"Unsupported tool type: {tool.get('type')}")

    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError("OpenAI chat tools must include a function object")

    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool function name is required")

    parameters = function.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise ValueError("Tool parameters must be a JSON schema object")

    description = function.get("description", "")
    return {
        "name": name.strip(),
        "description": str(description or ""),
        "parameters": parameters,
    }


def render_param(
    name: str, schema: dict[str, Any], required: bool, indent: int = 0
) -> list[str]:
    pad = "  " * indent
    typ = schema.get("type", "any")
    desc = str(schema.get("description", "") or "").strip()
    req_text = "required" if required else "optional"

    line = f"{pad}- {name} ({typ}, {req_text})"
    if desc:
        line += f": {desc}"

    lines = [line]

    if "enum" in schema:
        allowed = ", ".join(map(str, schema["enum"]))
        lines.append(f"{pad}  Allowed values: {allowed}")

    if typ == "object":
        props = schema.get("properties", {})
        reqs = set(schema.get("required", []))
        if isinstance(props, dict) and props:
            lines.append(f"{pad}  Fields:")
            for child_name, child_schema in props.items():
                if not isinstance(child_schema, dict):
                    continue
                lines.extend(
                    render_param(
                        child_name,
                        child_schema,
                        child_name in reqs,
                        indent=indent + 2,
                    )
                )
    elif typ == "array":
        items = schema.get("items", {})
        if not isinstance(items, dict):
            items = {}
        item_type = items.get("type", "any")
        lines.append(f"{pad}  Items type: {item_type}")

        if item_type == "object":
            props = items.get("properties", {})
            reqs = set(items.get("required", []))
            if isinstance(props, dict) and props:
                lines.append(f"{pad}  Item fields:")
                for child_name, child_schema in props.items():
                    if not isinstance(child_schema, dict):
                        continue
                    lines.extend(
                        render_param(
                            child_name,
                            child_schema,
                            child_name in reqs,
                            indent=indent + 2,
                        )
                    )

    return lines


def render_tool(tool: dict[str, Any]) -> str:
    name = tool["name"]
    description = tool.get("description", "").strip() or "No description provided."
    params = tool.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    props = params.get("properties", {})
    if not isinstance(props, dict):
        props = {}
    required = set(params.get("required", []))

    lines = [
        f"Tool: {name}",
        f"Purpose: {description}",
        "",
        "Arguments:",
    ]

    if not props:
        lines.append("- This tool takes no arguments.")
    else:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            lines.extend(render_param(param_name, param_schema, param_name in required))

    example_args = {}
    for param_name, param_schema in props.items():
        if not isinstance(param_schema, dict):
            example_args[param_name] = None
            continue
        typ = param_schema.get("type", "string")
        if typ == "string":
            if "enum" in param_schema and param_schema["enum"]:
                example_args[param_name] = param_schema["enum"][0]
            else:
                example_args[param_name] = f"<{param_name}>"
        elif typ == "integer":
            example_args[param_name] = 1
        elif typ == "number":
            example_args[param_name] = 1
        elif typ == "boolean":
            example_args[param_name] = True
        elif typ == "array":
            example_args[param_name] = []
        elif typ == "object":
            example_args[param_name] = {}
        else:
            example_args[param_name] = None

    example_json = json.dumps(
        {"name": name, "arguments": example_args}, ensure_ascii=False
    )

    lines.extend(
        [
            "",
            "Tool call format:",
            "<tool_call>",
            example_json,
            "</tool_call>",
        ]
    )
    return "\n".join(lines)


def tools_to_system_prompt(tools: list[dict[str, Any]]) -> str:
    normalized = [normalize_tool(tool) for tool in tools]
    header = """You may use tools when needed.

Follow these rules strictly:
1. If a tool is needed, output EXACTLY one tool call.
2. A tool call must be wrapped in <tool_call> and </tool_call>.
3. Inside <tool_call>, output valid JSON only.
4. Do not use markdown code fences.
5. Do not add any explanation before or after a tool call.
6. If no tool is needed, answer normally.
"""
    body = "\n\n".join(render_tool(tool) for tool in normalized)
    return header + "\n\nAvailable tools:\n\n" + body


def apply_tool_compatibility(body: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(body)
    mode = _normalize_rwkv_tool_compat_mode(prepared.get("rwkv_tool_compat_mode", True))
    tools = prepared.get("tools")
    prepared["_rwkv_tool_compat_active"] = False

    if mode == "false":
        prepared.pop("tools", None)
        return prepared

    if tools is None:
        return prepared

    if not isinstance(tools, list):
        raise ValueError("tools must be a list")

    if not tools:
        prepared.pop("tools", None)
        return prepared

    tool_prompt = tools_to_system_prompt(tools)
    existing_system = prepared.get("system")
    if existing_system:
        prepared["system"] = f"{existing_system}\n\n{tool_prompt}"
    else:
        prepared["system"] = tool_prompt

    prepared["_rwkv_tool_compat_active"] = True
    prepared.pop("tools", None)
    return prepared


def format_openai_prompt(body: dict, enable_think: bool) -> str:
    contents = body.get("contents") or []
    messages = body.get("messages") or []
    system_field = body.get("system")
    current_prompt = str(contents[0]).strip() if contents else ""

    system_parts = []
    if system_field:
        system_parts.append(str(system_field))

    history_messages = []
    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
        "developer": "Developer",
    }

    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = normalize_message_content(message.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        dialogue_role = role_map.get(role, role.capitalize() or "User")
        history_messages.append((dialogue_role, content))

    if not current_prompt and history_messages:
        current_prompt = history_messages[-1][1].strip()

    if (
        current_prompt
        and history_messages
        and history_messages[-1][1].strip() == current_prompt
    ):
        history_messages = history_messages[:-1]

    double_newline = "\n\n"
    prompt_parts = []
    if system_parts:
        prompt_parts.append(f"System: {double_newline.join(system_parts)}")
    if history_messages:
        prompt_parts.append(
            double_newline.join(
                f"{role}: {content}" for role, content in history_messages
            )
        )
    if current_prompt:
        prompt_parts.append(f"User: {current_prompt}")

    prompt_text = double_newline.join(part for part in prompt_parts if part).strip()
    if not prompt_text:
        prompt_text = extract_openai_prompt(body)

    if enable_think:
        return f"{prompt_text}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>"


def format_openai_state_prompt(body: dict, enable_think: bool) -> str:
    contents = body.get("contents") or []
    messages = body.get("messages") or []
    system_field = body.get("system")

    if len(contents) > 1:
        raise ValueError("State mode only supports a single contents item")

    system_parts = []
    if system_field:
        system_parts.append(str(system_field).strip())

    role_map = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
        "developer": "Developer",
    }

    prompt_parts = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = normalize_message_content(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        dialogue_role = role_map.get(role, role.capitalize() or "User")
        prompt_parts.append(f"{dialogue_role}: {content}")

    if contents:
        content_prompt = str(contents[0]).strip()
        if content_prompt:
            prompt_parts.append(f"User: {content_prompt}")

    final_parts = []
    if system_parts:
        final_parts.append(
            "System: " + "\n\n".join(part for part in system_parts if part)
        )
    if prompt_parts:
        final_parts.append("\n\n".join(prompt_parts))

    prompt_text = "\n\n".join(part for part in final_parts if part).strip()
    if not prompt_text:
        raise ValueError("State mode requires incremental contents or messages")

    if enable_think:
        return f"{prompt_text}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>"


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
        "stop_tokens": body.get("stop_tokens", [0, 261, 24281]),
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


def _extract_tag_tool_call(text: str) -> Optional[tuple[str, str]]:
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    payload_start = start_idx + len(start_tag)
    end_idx = text.find(end_tag, payload_start)
    if end_idx == -1:
        return None

    payload = text[payload_start:end_idx].strip()
    remaining = (text[:start_idx] + text[end_idx + len(end_tag) :]).strip()
    return payload, remaining


def parse_tool_call_response(
    text: str, parser_mode: str = "auto"
) -> Optional[dict[str, Any]]:
    mode = str(parser_mode or "auto").lower()
    if mode not in {"auto", "tag"}:
        return None

    extracted = _extract_tag_tool_call(text)
    if extracted is None:
        return None

    payload_text, remaining_content = extracted
    try:
        parsed = json.loads(payload_text)
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
        "content": remaining_content or None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
        "finish_reason": "tool_calls",
    }
    return None


def build_openai_message_response(
    result_text: str, finish_reason: str, body: dict
) -> tuple[dict[str, Any], str]:
    if (
        body.get("enable_tool_calls", False)
        or body.get("_rwkv_tool_compat_active", False)
    ) and not body.get("stream", False):
        parsed = parse_tool_call_response(
            result_text, body.get("tool_call_parser", "auto")
        )
        if parsed is not None:
            return {
                "role": "assistant",
                "content": parsed["content"],
                "tool_calls": parsed["tool_calls"],
            }, parsed["finish_reason"]

    return {"role": "assistant", "content": result_text}, finish_reason

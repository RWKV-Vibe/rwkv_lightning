import json
import re
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


def build_openai_usage(tokenizer, prompt_text: str, completion_text: str) -> dict:
    prompt_tokens = len(tokenizer.encode(prompt_text))
    completion_tokens = len(tokenizer.encode(completion_text)) if completion_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_internal_chat_request(body: dict, prompt: str) -> dict:
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
        "stream": body.get("stream", False),
        "pad_zero": body.get("pad_zero", False),
        "alpha_presence": body.get("alpha_presence", 1),
        "alpha_frequency": body.get("alpha_frequency", 0.1),
        "alpha_decay": body.get("alpha_decay", 0.996),
        "enable_think": body.get("enable_think", False),
        "chunk_size": body.get("chunk_size", 2),
        "password": body.get("password"),
    }


def _extract_tag_tool_call(text: str) -> Optional[tuple[str, str]]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None
    remaining = (text[: match.start()] + text[match.end() :]).strip()
    return match.group(1), remaining


def _extract_prefix_tool_call(text: str) -> Optional[tuple[str, str]]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("@@tool "):
            continue
        payload = stripped[len("@@tool ") :].strip()
        remaining_lines = []
        removed = False
        for current_line in text.splitlines():
            current_stripped = current_line.strip()
            if not removed and current_stripped == stripped:
                removed = True
                continue
            remaining_lines.append(current_line)
        remaining = "\n".join(remaining_lines).strip()
        return payload, remaining
    return None


def parse_tool_call_response(
    text: str, parser_mode: str = "auto"
) -> Optional[dict[str, Any]]:
    mode = str(parser_mode or "auto").lower()
    extractors = []
    if mode in {"auto", "tag"}:
        extractors.append(_extract_tag_tool_call)
    if mode in {"auto", "prefix"}:
        extractors.append(_extract_prefix_tool_call)

    for extractor in extractors:
        extracted = extractor(text)
        if extracted is None:
            continue
        payload_text, remaining_content = extracted
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(arguments, dict):
            continue

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
    if body.get("enable_tool_calls", False) and not body.get("stream", False):
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

#!/usr/bin/env python3
"""Unit tests for OpenAI adapter parsing helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from API_servers.openai_adapter import (
    apply_tool_compatibility,
    format_openai_prompt,
    format_openai_state_prompt,
    normalize_tool,
    parse_tool_call_response,
    tools_to_system_prompt,
)


def _assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def _assert_tool_call(
    result: dict, expected_name: str, expected_arguments: dict
) -> None:
    tool_calls = result.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise AssertionError(f"expected exactly one tool call, got {tool_calls!r}")

    tool_call = tool_calls[0]
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise AssertionError(f"expected function object, got {function!r}")

    _assert_equal(function.get("name"), expected_name, "tool name mismatch")
    arguments = json.loads(function.get("arguments", ""))
    _assert_equal(arguments, expected_arguments, "tool arguments mismatch")


def test_parse_tag_tool_call_with_surrounding_text() -> None:
    text = (
        "Need to call a tool first.\n"
        "<tool_call>\n"
        '{"name":"get_weather","arguments":{"location":"Beijing","date":"today"}}\n'
        "</tool_call>\n"
        "Then summarize the result."
    )
    result = parse_tool_call_response(text, parser_mode="tag")
    if result is None:
        raise AssertionError("expected tag-mode parser to return a tool call")

    _assert_equal(result.get("finish_reason"), "tool_calls", "finish reason mismatch")
    _assert_equal(
        result.get("content"),
        "Need to call a tool first.\n\nThen summarize the result.",
        "residual content mismatch",
    )
    _assert_tool_call(
        result,
        "get_weather",
        {"location": "Beijing", "date": "today"},
    )
    print("[PASS] test_parse_tag_tool_call_with_surrounding_text")


def test_parse_prefix_tool_call_is_not_supported() -> None:
    text = (
        "I will use the search tool.\n"
        '@@tool {"name":"web_search","arguments":{"query":"latest news about RWKV"}}\n'
        "Please keep this note as content."
    )
    _assert_equal(
        parse_tool_call_response(text, parser_mode="prefix"),
        None,
        "prefix parser should be unsupported",
    )
    print("[PASS] test_parse_prefix_tool_call_is_not_supported")


def test_parse_tool_call_response_returns_none_for_malformed_json() -> None:
    malformed_tag = (
        "<tool_call>\n"
        '{"name":"calculator","arguments":{"expression":"1 + 2"}\n'
        "</tool_call>"
    )
    _assert_equal(
        parse_tool_call_response(malformed_tag, parser_mode="tag"),
        None,
        "malformed tag JSON should not parse",
    )

    print("[PASS] test_parse_tool_call_response_returns_none_for_malformed_json")


def test_parse_tool_call_response_keeps_first_tool_call_when_multiple_exist() -> None:
    text = (
        "before\n"
        "<tool_call>\n"
        '{"name":"first_tool","arguments":{"step":1}}\n'
        "</tool_call>\n"
        "middle\n"
        "<tool_call>\n"
        '{"name":"second_tool","arguments":{"step":2}}\n'
        "</tool_call>\n"
        "after"
    )
    result = parse_tool_call_response(text, parser_mode="tag")
    if result is None:
        raise AssertionError("expected first tool call to be parsed")

    _assert_tool_call(result, "first_tool", {"step": 1})
    _assert_equal(
        result.get("content"),
        'before\n\nmiddle\n<tool_call>\n{"name":"second_tool","arguments":{"step":2}}\n</tool_call>\nafter',
        "multiple tool call residual content mismatch",
    )
    print(
        "[PASS] test_parse_tool_call_response_keeps_first_tool_call_when_multiple_exist"
    )


def test_parse_tool_call_response_supports_nested_arguments_and_braces() -> None:
    text = (
        "before\n"
        "<tool_call>\n"
        '{"name":"complex_tool","arguments":{"query":"literal } brace","filters":{"tags":["rwkv","tooling"],"limit":2}}}\n'
        "</tool_call>\n"
        "after"
    )
    result = parse_tool_call_response(text, parser_mode="tag")
    if result is None:
        raise AssertionError("expected nested arguments tool call to be parsed")

    _assert_tool_call(
        result,
        "complex_tool",
        {
            "query": "literal } brace",
            "filters": {"tags": ["rwkv", "tooling"], "limit": 2},
        },
    )
    _assert_equal(result.get("content"), "before\n\nafter", "nested residual mismatch")
    print("[PASS] test_parse_tool_call_response_supports_nested_arguments_and_braces")


def test_format_openai_state_prompt_only_includes_incremental_messages() -> None:
    body = {
        "system": "Follow instructions.",
        "messages": [
            {"role": "system", "content": "System repeated"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "New question"},
        ],
    }
    prompt = format_openai_state_prompt(body, enable_think=False)
    if "Previous answer" not in prompt or "New question" not in prompt:
        raise AssertionError("state prompt should include incremental messages")
    if "System:" not in prompt:
        raise AssertionError("state prompt should include system content")
    print("[PASS] test_format_openai_state_prompt_only_includes_incremental_messages")


def test_format_openai_state_prompt_rejects_empty() -> None:
    try:
        format_openai_state_prompt({}, enable_think=False)
    except ValueError:
        print("[PASS] test_format_openai_state_prompt_rejects_empty")
        return
    raise AssertionError("state prompt should reject empty payload")


def test_normalize_tool_rejects_missing_function_object() -> None:
    try:
        normalize_tool({"type": "function", "name": "bad"})
    except ValueError:
        print("[PASS] test_normalize_tool_rejects_missing_function_object")
        return
    raise AssertionError("normalize_tool should reject non-chat-completions format")


def test_tools_to_system_prompt_renders_function_tool() -> None:
    prompt = tools_to_system_prompt(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
    )
    if "Available tools:" not in prompt or "Tool: get_weather" not in prompt:
        raise AssertionError("tool prompt should include tool metadata")
    if "Allowed values: celsius, fahrenheit" not in prompt:
        raise AssertionError("tool prompt should render enum values")
    if '<tool_call>\n{"name": "get_weather"' not in prompt:
        raise AssertionError("tool prompt should include a tool call example")
    print("[PASS] test_tools_to_system_prompt_renders_function_tool")


def test_apply_tool_compatibility_true_converts_tools_and_preserves_stream() -> None:
    body = {
        "system": "Base rules.",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search docs.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    prepared = apply_tool_compatibility(body)
    _assert_equal(prepared.get("stream"), True, "true mode should preserve stream")
    if "Available tools:" not in prepared.get("system", ""):
        raise AssertionError("true mode should append tool prompt into system")
    if prepared.get("tools") is not None:
        raise AssertionError(
            "true mode should drop original tools field after conversion"
        )
    print(
        "[PASS] test_apply_tool_compatibility_true_converts_tools_and_preserves_stream"
    )


def test_apply_tool_compatibility_false_discards_tools_without_conversion() -> None:
    prepared = apply_tool_compatibility(
        {
            "rwkv_tool_compat_mode": False,
            "stream": True,
            "system": "Keep me.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ignored_tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    )
    _assert_equal(prepared.get("stream"), True, "false mode should preserve stream")
    _assert_equal(
        prepared.get("system"), "Keep me.", "false mode should not modify system"
    )
    if prepared.get("tools") is not None:
        raise AssertionError("false mode should discard tools field")
    print(
        "[PASS] test_apply_tool_compatibility_false_discards_tools_without_conversion"
    )


def test_apply_tool_compatibility_defaults_to_true_without_tools() -> None:
    prepared = apply_tool_compatibility({"stream": True, "system": "Base rules."})
    _assert_equal(
        prepared.get("stream"),
        True,
        "default true mode should preserve streaming without tools",
    )
    _assert_equal(
        prepared.get("system"),
        "Base rules.",
        "default true mode should leave system unchanged without tools",
    )
    print("[PASS] test_apply_tool_compatibility_defaults_to_true_without_tools")


def main() -> None:
    print("Running openai_adapter parsing tests...")
    _ = format_openai_prompt({"messages": [{"role": "user", "content": "hi"}]}, False)
    test_parse_tag_tool_call_with_surrounding_text()
    test_parse_prefix_tool_call_is_not_supported()
    test_parse_tool_call_response_returns_none_for_malformed_json()
    test_parse_tool_call_response_keeps_first_tool_call_when_multiple_exist()
    test_parse_tool_call_response_supports_nested_arguments_and_braces()
    test_format_openai_state_prompt_only_includes_incremental_messages()
    test_format_openai_state_prompt_rejects_empty()
    test_normalize_tool_rejects_missing_function_object()
    test_tools_to_system_prompt_renders_function_tool()
    test_apply_tool_compatibility_true_converts_tools_and_preserves_stream()
    test_apply_tool_compatibility_false_discards_tools_without_conversion()
    test_apply_tool_compatibility_defaults_to_true_without_tools()
    print("All openai_adapter parsing tests passed.")


if __name__ == "__main__":
    main()

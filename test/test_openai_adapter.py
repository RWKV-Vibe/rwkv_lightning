#!/usr/bin/env python3
"""Unit tests for OpenAI adapter parsing helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from API_servers.openai_adapter import parse_tool_call_response


def _assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def _assert_tool_call(result: dict, expected_name: str, expected_arguments: dict) -> None:
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


def test_parse_prefix_tool_call_with_surrounding_text() -> None:
    text = (
        "I will use the search tool.\n"
        '@@tool {"name":"web_search","arguments":{"query":"latest news about RWKV"}}\n'
        "Please keep this note as content."
    )
    result = parse_tool_call_response(text, parser_mode="prefix")
    if result is None:
        raise AssertionError("expected prefix-mode parser to return a tool call")

    _assert_equal(
        result.get("content"),
        "I will use the search tool.\nPlease keep this note as content.",
        "prefix residual content mismatch",
    )
    _assert_tool_call(result, "web_search", {"query": "latest news about RWKV"})
    print("[PASS] test_parse_prefix_tool_call_with_surrounding_text")


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

    malformed_prefix = '@@tool {"name":"calculator","arguments":[1,2,3]}'
    _assert_equal(
        parse_tool_call_response(malformed_prefix, parser_mode="prefix"),
        None,
        "prefix arguments must be an object",
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
        "before\n\nmiddle\n<tool_call>\n{\"name\":\"second_tool\",\"arguments\":{\"step\":2}}\n</tool_call>\nafter",
        "multiple tool call residual content mismatch",
    )
    print("[PASS] test_parse_tool_call_response_keeps_first_tool_call_when_multiple_exist")


def main() -> None:
    print("Running openai_adapter parsing tests...")
    test_parse_tag_tool_call_with_surrounding_text()
    test_parse_prefix_tool_call_with_surrounding_text()
    test_parse_tool_call_response_returns_none_for_malformed_json()
    test_parse_tool_call_response_keeps_first_tool_call_when_multiple_exist()
    print("All openai_adapter parsing tests passed.")


if __name__ == "__main__":
    main()

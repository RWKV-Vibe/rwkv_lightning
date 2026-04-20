import json
from typing import Any


class XGrammarConstraint:
    def __init__(self, xgr_module, compiled_grammar, vocab_size: int):
        self._xgr = xgr_module
        self._matcher = xgr_module.GrammarMatcher(compiled_grammar)
        self._token_bitmask = xgr_module.allocate_token_bitmask(1, vocab_size)

    def apply(self, logits):
        logits_2d = logits if logits.dim() == 2 else logits.unsqueeze(0)
        self._matcher.fill_next_token_bitmask(self._token_bitmask)
        self._xgr.apply_token_bitmask_inplace(
            logits_2d, self._token_bitmask.to(logits_2d.device)
        )

    def accept_token(self, token_id: int) -> bool:
        return bool(self._matcher.accept_token(int(token_id)))

    def is_terminated(self) -> bool:
        return bool(self._matcher.is_terminated())


class XGrammarRuntime:
    def __init__(self, tokenizer, vocab_size: int):
        import xgrammar as xgr

        encoded_vocab = [tokenizer.idx2token.get(i, b"") for i in range(vocab_size)]
        self._xgr = xgr
        self._tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab,
            vocab_type=xgr.VocabType.RAW,
            vocab_size=vocab_size,
            stop_token_ids=[0],
        )
        self._compiler = xgr.GrammarCompiler(self._tokenizer_info, max_threads=8)

    def build_constraint(self, response_format: dict[str, Any] | None):
        if not response_format:
            return None

        response_type = str(response_format.get("type", "text")).strip().lower()
        if response_type in {"", "text"}:
            return None

        if response_type == "json_object":
            compiled_grammar = self._compiler.compile_builtin_json_grammar()
            return XGrammarConstraint(
                self._xgr, compiled_grammar, self._tokenizer_info.vocab_size
            )

        if response_type == "json_schema":
            json_schema = response_format.get("json_schema") or {}
            schema = json_schema.get("schema", json_schema)
            strict_mode = bool(json_schema.get("strict", True)) if isinstance(json_schema, dict) else True
            if not schema:
                raise ValueError(
                    "response_format.json_schema must include a non-empty schema"
                )
            compiled_grammar = self._compiler.compile_json_schema(
                schema, strict_mode=strict_mode
            )
            return XGrammarConstraint(
                self._xgr, compiled_grammar, self._tokenizer_info.vocab_size
            )

        raise ValueError(
            f"Unsupported response_format type '{response_type}'. Expected one of: text, json_object, json_schema"
        )


def has_xgrammar() -> bool:
    try:
        import xgrammar  # noqa: F401

        return True
    except Exception:
        return False


def normalize_response_format(response_format: Any) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if isinstance(response_format, str):
        if response_format.strip().lower() == "text":
            return {"type": "text"}
        raise ValueError(
            "response_format string is unsupported; expected an object like {'type': 'json_object'}"
        )
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")

    normalized = dict(response_format)
    normalized["type"] = str(normalized.get("type", "text")).strip().lower()
    if normalized["type"] == "json_schema":
        json_schema = normalized.get("json_schema")
        if isinstance(json_schema, str):
            normalized["json_schema"] = json.loads(json_schema)
    return normalized

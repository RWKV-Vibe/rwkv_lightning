import ast
from functools import lru_cache
from importlib.resources import files

import torch


class TRIE:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        node = self
        ch = key[idx]
        match = None
        while node.to[ch] is not None:
            node = node.to[ch]
            idx += 1
            if node.values:
                match = idx, node.values
            if idx == len(key):
                break
            ch = key[idx]
        if match is None:
            raise ValueError("Failed to match bytes in RWKV tokenizer trie.")
        return match


class RWKVTokenizer:
    eos_token_id = 0

    def __init__(self, vocab_file=None):
        if vocab_file is None:
            vocab_file = files("nanovllm.tokenizers").joinpath("rwkv_vocab_v20230424.txt")
        self.idx2token = {0: b"<|endoftext|>"}
        with vocab_file.open("r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[:line.index(" ")])
                token = ast.literal_eval(line[line.index(" "):line.rindex(" ")])
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == int(line[line.rindex(" "):])
                self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items() if idx != 0}
        self.root = TRIE()
        for token, idx in self.token2idx.items():
            self.root.add(token, val=idx)
        self.vocab_size = len(self.idx2token)

    def encode_bytes(self, src: bytes):
        idx = 0
        token_ids = []
        while idx < len(src):
            prev_idx = idx
            idx, values = self.root.find_longest(src, idx)
            assert idx != prev_idx
            token_ids.append(next(iter(values)))
        return token_ids

    def decode_bytes(self, token_ids):
        return b"".join(self.idx2token[token_id] for token_id in token_ids)

    def encode(self, src: str, **kwargs):
        return self.encode_bytes(src.encode("utf-8"))

    def decode(self, token_ids, utf8_errors: str = "strict", **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.decode_bytes(token_ids).decode("utf-8", errors=utf8_errors)


@lru_cache(maxsize=1)
def get_rwkv_tokenizer():
    return RWKVTokenizer()

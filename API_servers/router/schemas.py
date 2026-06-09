from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = []
    messages: list[dict] = []
    system: Optional[str] = None
    prefix: list[str] = []
    suffix: list[str] = []
    max_tokens: int = 8192
    stop_tokens: list[str] = ["\nUser:"]
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.6
    noise: float = 1.5
    stream: bool = False
    pad_zero: bool = True
    alpha_presence: float = 2
    alpha_frequency: float = 0.2
    alpha_decay: float = 0.996
    enable_think: bool = False
    chunk_size: int = 4
    password: Optional[str] = None
    session_id: Optional[str] = None
    dialogue_idx: Optional[int] = 0
    use_prefix_cache: bool = True


class TranslateRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str
    text_list: list[str]
    placeholders: Optional[list[str]] = None


class TranslateResponse(BaseModel):
    translations: list[dict]

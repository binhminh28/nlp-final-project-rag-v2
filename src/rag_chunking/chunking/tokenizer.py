"""Reproducible tokenizer adapter for chunk accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken
from tiktoken.core import Encoding


DEFAULT_ENCODING = "cl100k_base"


@dataclass(slots=True)
class TiktokenTokenizer:
    """Use ordinary-text encoding so special-looking source text stays content."""

    encoding_name: str = DEFAULT_ENCODING
    _encoding: Encoding = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._encoding = tiktoken.get_encoding(self.encoding_name)

    @property
    def name(self) -> str:
        return f"tiktoken:{self.encoding_name}"

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def token_bytes(self, token: int) -> bytes:
        return self._encoding.decode_single_token_bytes(token)

    def decode_strict(self, tokens: list[int]) -> str:
        """Decode a slice without silently replacing incomplete UTF-8."""

        return b"".join(self.token_bytes(token) for token in tokens).decode("utf-8", errors="strict")


def is_utf8_safe_boundary(
    tokens: list[int], position: int, tokenizer: TiktokenTokenizer
) -> bool:
    """Return whether a token position is also a UTF-8 code-point boundary."""

    if not 0 <= position <= len(tokens):
        raise IndexError(f"Token boundary outside stream: {position}")
    if position in (0, len(tokens)):
        return True
    first_byte = tokenizer.token_bytes(tokens[position])[0]
    return first_byte & 0b1100_0000 != 0b1000_0000


def retreat_to_utf8_safe_boundary(
    tokens: list[int], position: int, tokenizer: TiktokenTokenizer
) -> int:
    """Move backward by the minimum token count needed for UTF-8 safety."""

    while position > 0 and not is_utf8_safe_boundary(tokens, position, tokenizer):
        position -= 1
    return position

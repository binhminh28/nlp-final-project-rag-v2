"""Conservative preflight checks for outbound planner prompts."""

from __future__ import annotations

import re


class OutboundPayloadSafetyError(ValueError):
    """An outbound prompt contains high-confidence sensitive material."""


_CHECKS = {
    "private key marker": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE
    ),
    "credential assignment": re.compile(
        r"(?:api[_-]?key|client[_-]?secret|password)\s*[:=]\s*[\"']?"
        r"(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|[A-Za-z0-9_./+\-=]{32,})",
        re.IGNORECASE,
    ),
    "secret-shaped API token": re.compile(
        r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"
    ),
    "Windows user path": re.compile(r"\b[A-Z]:\\Users\\", re.IGNORECASE),
    # Require a user plus another path component so ordinary application routes
    # such as /home/blog are not mistaken for host filesystem disclosure.
    "Unix home path": re.compile(
        r"(?<!\w)/(?:home|Users)/[^/\s\"']+/[^\s\"']+"
    ),
    "local file URI": re.compile(r"\bfile://(?:/[A-Za-z]:|/(?:home|Users)/)", re.IGNORECASE),
    "dotenv API-key assignment": re.compile(
        r"^\s*(?:OPENROUTER_API_KEY|OPENAI_API_KEY)\s*=", re.MULTILINE
    ),
}


def validate_outbound_payload(payload: str, *, configured_secret: str | None = None) -> None:
    """Reject only high-confidence secret/path leakage; never mutate source text."""

    if configured_secret and configured_secret in payload:
        raise OutboundPayloadSafetyError("outbound payload contains the configured API key")
    for label, pattern in _CHECKS.items():
        if pattern.search(payload):
            raise OutboundPayloadSafetyError(f"outbound payload contains {label}")

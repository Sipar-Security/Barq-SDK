"""Secret redaction for audit records.

The audit log stores the exact request/response of every exchange so a later claim can bind
to precise evidence. That is the point of it; it is also why an unredacted log is a
liability: the agent's own `Authorization: Bearer …` header, provider API keys, and whatever
the response body happened to contain (card numbers, national IDs) all get written verbatim,
fsync'd, hash-chained so they cannot be quietly removed, into a file that lives inside the
workdir the agent itself can write to.

Redaction runs BEFORE hashing, so the chain covers the redacted bytes and evidence
verification still works: you verify what was actually stored.

The default policy masks by header name (an allowlist of well-known credential headers plus
anything whose name contains "key"/"token"/"secret"/"auth"), and by value shape for the
common secret formats that leak through body text. It is deliberately conservative: a
missed secret is worse than an over-redacted one, and the un-redacted value never needs to
be recoverable from the log.
"""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"

# Header names that always carry a credential.
_SECRET_HEADERS = frozenset({
    "authorization", "proxy-authorization", "cookie", "set-cookie",
    "x-api-key", "api-key", "x-auth-token", "x-access-token", "x-csrf-token",
    "x-amz-security-token", "x-goog-api-key", "private-token",
})
# Any header whose NAME contains one of these is treated as a credential too.
_SECRET_NAME_FRAGMENTS = ("key", "token", "secret", "auth", "credential", "password", "session")

# Value shapes worth masking wherever they appear (bodies included).
_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(sk|pk|rk|ak)[-_][A-Za-z0-9\-_]{8,}", re.I), MASK),      # sk-…, ak_live_…
    (re.compile(r"\bAKIA[0-9A-Z]{12,}\b"), MASK),                            # AWS access key id
    (re.compile(r"\bASIA[0-9A-Z]{12,}\b"), MASK),                            # AWS STS key id
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), MASK),                   # GitHub tokens
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), MASK),                # Slack tokens
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"), MASK),  # JWT
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.S), MASK),                                                # PEM private keys
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}"), MASK),   # inline auth
)


def _is_secret_header(name: str) -> bool:
    n = name.lower()
    return n in _SECRET_HEADERS or any(f in n for f in _SECRET_NAME_FRAGMENTS)


def redact_text(text: str) -> str:
    """Mask known secret shapes inside free text (a response body, a command line)."""
    if not text:
        return text
    for pattern, repl in _VALUE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_headers(headers: Any) -> Any:
    """Mask credential-bearing headers by name, keeping the names themselves visible so the
    record still shows WHICH credential was presented."""
    if not isinstance(headers, dict):
        return headers
    return {
        k: (MASK if _is_secret_header(str(k)) else redact_value(v))
        for k, v in headers.items()
    }


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_mapping(data: Any) -> Any:
    """Recursively redact a request/response record.

    A key that names a credential is masked outright; `headers` gets header-aware treatment;
    everything else is scanned for secret-shaped values.
    """
    if not isinstance(data, dict):
        return redact_value(data)
    out: dict = {}
    for key, value in data.items():
        k = str(key)
        if k.lower() in ("headers", "header"):
            out[key] = redact_headers(value)
        elif _is_secret_header(k):
            out[key] = MASK
        else:
            out[key] = redact_value(value)
    return out


def default_redactor(record: Any) -> Any:
    """The redactor AuditLog uses unless a caller supplies its own (or None to disable)."""
    return redact_mapping(record)

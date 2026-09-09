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
    # A URL with inline credentials (postgres://admin:S3cretPw@db/...). The password is the
    # whole secret and it was written verbatim: `_SECRET_HEADERS` only matches header NAMES,
    # and none of the token shapes above look like an arbitrary password.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@"), r"\1" + MASK + "@"),
    # `password=...`, `passwd: ...`, `secret=...`, `token=...`, `pin=...` in a body or a
    # command line. Extremely common, and none of the shape patterns catch an arbitrary
    # value — a secret is only recognisable here by the name of the field carrying it.
    (re.compile(
        r"(?i)\b(pass(?:word|wd|phrase)?|secret|token|api[_-]?key|auth|pin|otp|"
        r"client[_-]?secret|private[_-]?key)"
        r"(\s*[=:]\s*|\"\s*:\s*\"?)([^\s,;&\"'}\]]{3,})"
    ), lambda m: m.group(0) if m.group(3).startswith(MASK[:5]) else m.group(1) + m.group(2) + MASK),
)

# PII/PCI shapes. Separate from credentials because the reason to mask them is different
# (privacy and PCI/GDPR scope, not access control) and because a deployment that must keep
# them — a payments audit trail whose whole purpose is the card reference — turns exactly
# this group off via `default_redactor(record, mask_pii=False)`.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], Any], ...] = (
    # Payment cards: 13-19 digits, optionally spaced/hyphened, validated with Luhn so an
    # order id or a long integer is not mangled.
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "_luhn"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), MASK),  # email
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), MASK),                               # US SSN
    (re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"), MASK),  # phone
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), MASK),                    # IBAN
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> str:
    """Mask personal and payment data. Applied after the credential patterns."""
    if not text:
        return text
    for pattern, repl in _PII_PATTERNS:
        if repl == "_luhn":
            text = pattern.sub(
                lambda m: MASK if _luhn_ok(re.sub(r"\D", "", m.group(0))) else m.group(0),
                text,
            )
        else:
            text = pattern.sub(repl, text)
    return text


def _is_secret_header(name: str) -> bool:
    n = name.lower()
    return n in _SECRET_HEADERS or any(f in n for f in _SECRET_NAME_FRAGMENTS)


def redact_text(text: str, *, mask_pii: bool = True) -> str:
    """Mask known secret shapes inside free text (a response body, a command line).

    `mask_pii` additionally masks emails, payment cards, SSNs, phone numbers and IBANs.
    On by default: the audit log is append-only and hash-chained, so anything written to
    it cannot later be deleted without breaking the chain — which puts a log holding
    personal data in direct conflict with an erasure request. Keeping it out is far
    cheaper than arguing about it afterwards.
    """
    if not text:
        return text
    for pattern, repl in _VALUE_PATTERNS:
        text = pattern.sub(repl, text)
    if mask_pii:
        text = redact_pii(text)
    return text


def redact_headers(headers: Any, *, mask_pii: bool = True) -> Any:
    """Mask credential-bearing headers by name, keeping the names themselves visible so the
    record still shows WHICH credential was presented."""
    if not isinstance(headers, dict):
        return headers
    return {
        k: (MASK if _is_secret_header(str(k)) else redact_value(v, mask_pii=mask_pii))
        for k, v in headers.items()
    }


def redact_value(value: Any, *, mask_pii: bool = True) -> Any:
    if isinstance(value, str):
        return redact_text(value, mask_pii=mask_pii)
    if isinstance(value, dict):
        return redact_mapping(value, mask_pii=mask_pii)
    if isinstance(value, list):
        return [redact_value(v, mask_pii=mask_pii) for v in value]
    return value


def redact_mapping(data: Any, *, mask_pii: bool = True) -> Any:
    """Recursively redact a request/response record.

    A key that names a credential is masked outright; `headers` gets header-aware treatment;
    everything else is scanned for secret-shaped values.
    """
    if not isinstance(data, dict):
        return redact_value(data, mask_pii=mask_pii)
    out: dict = {}
    for key, value in data.items():
        k = str(key)
        if k.lower() in ("headers", "header"):
            out[key] = redact_headers(value, mask_pii=mask_pii)
        elif _is_secret_header(k):
            out[key] = MASK
        else:
            out[key] = redact_value(value, mask_pii=mask_pii)
    return out


def default_redactor(record: Any, *, mask_pii: bool = True) -> Any:
    """The redactor AuditLog uses unless a caller supplies its own (or None to disable).

    Pass `functools.partial(default_redactor, mask_pii=False)` when the record's personal
    or payment data IS the evidence being kept (a payments trail), and you have a lawful
    basis and a retention story for it.
    """
    return redact_mapping(record, mask_pii=mask_pii)

"""Tamper-evidence primitives for the audit log.

The audit log is a compliance artifact: it records what an autonomous agent did on
someone's behalf. "Durable JSONL, fsync'd"
proves the file survived a crash; it does NOT prove the file was not edited after the
fact. Enterprise audit means chain-of-custody: any insertion, deletion, reordering, or
in-place edit is *detectable*, and (with a key held out of band) *unforgeable*.

Three layers, each strictly stronger, each optional so there is a zero-dependency floor:

  1. Hash chain (always on). Every entry carries `prev` = the hash of the entry before
     it and `hash` = H(canonical(entry_core) || prev). The chain is anchored to a random
     `chain_id` per log, so an entry cannot be transplanted from one run's log
     into another's. Detects accidental corruption, reordering, and any edit or deletion
     WITHIN the chain.

     It does NOT detect truncation of the TAIL, and cannot: each entry links to the one
     before it, so dropping the last N leaves a chain that verifies perfectly, and an
     in-file seal is part of the tail that goes with them. The fix is an anchor held
     outside the file -- `AuditLog.anchor()` returns `{chain_id, head, count}`; store it
     where the agent cannot reach and pass it to `verify_audit_file(expected_head=...,
     expected_count=...)`. Failing that, `VerifyReport.unsealed_tail` reports how many
     entries sit past the last seal, which is exactly how many could have been removed
     unnoticed.
     An attacker who holds the file *and* knows the algorithm can recompute the whole
     chain, so this layer alone is tamper-EVIDENT to anyone holding an out-of-band copy
     of any single hash, and tamper-DETECTING against accidents.

  2. Keyed chain (HMAC). If an `hmac_key` is supplied (loaded from a secret store /
     env / file kept outside the workdir), `hash` becomes HMAC(key, ...). Now recomputing
     the chain requires the key, so an attacker who exfiltrates the log cannot forge a
     consistent one. This is the zero-extra-dependency unforgeability story.

  3. Signed seals (Ed25519, optional). `seal()` writes a signed checkpoint over the
     current head + count + chain_id. A third party (the client's triage team) verifies
     it against the operator's known public key: chain-of-custody without sharing a
     secret. Requires `cryptography`; absent it, seals still checkpoint (layers 1/2) but
     carry no asymmetric signature.

Everything here is pure/deterministic and dependency-light so the verifier can run
anywhere, including detached from the engine, on just the JSONL file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Bump only on a breaking change to the canonicalisation or hash-material format.
CHAIN_VERSION = 1
_GENESIS_PREFIX = f"Barq-SDK-audit-v{CHAIN_VERSION}:"
_LEGACY_GENESIS_PREFIX = f"bbengine-audit-v{CHAIN_VERSION}:"


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 preserved.

    Two logically-equal payloads must serialise byte-for-byte identically on every
    platform, or the same evidence would hash differently on the writer and the verifier.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def genesis_hash(chain_id: str, legacy: bool = False) -> str:
    """The `prev` of the very first entry. Binds the whole chain to `chain_id` so no
    entry (or run of entries) can be lifted from another log and still verify."""
    prefix = _LEGACY_GENESIS_PREFIX if legacy else _GENESIS_PREFIX
    return sha256_hex(prefix + chain_id)


def entry_core(entry_id: str, ts: float, kind: str, data: dict) -> dict:
    """The subset of an entry that is covered by the hash. `prev`/`hash`/signature
    fields are intentionally excluded: they are computed FROM this, not over it."""
    return {"id": entry_id, "ts": ts, "kind": kind, "data": data}


def compute_hash(core: dict, prev: str, hmac_key: Optional[bytes] = None) -> str:
    """The chain link. `material = canonical(core) || '\\n' || prev`.

    Plain SHA-256 by default; HMAC-SHA256 when a key is present (layer 2). The newline
    separator is unambiguous because canonical() never emits a bare newline."""
    material = (canonical(core) + "\n" + prev).encode("utf-8")
    if hmac_key is not None:
        return hmac.new(hmac_key, material, hashlib.sha256).hexdigest()
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class VerifyReport:
    """Structured result of verifying a chain. `ok` is the only thing most callers check;
    the rest makes a failure actionable (which entry broke, and why)."""

    ok: bool
    count: int = 0
    chain_id: str = ""
    broken_at: int = -1        # 0-based index of the first bad entry, or -1
    reason: str = ""
    keyed: bool = False        # was this chain verified as HMAC-keyed?
    seals: int = 0             # number of checkpoint seals seen
    seals_verified: int = 0    # of those, how many carried a valid Ed25519 signature
    truncated_tail: bool = False  # final line was a partial write (crash), not a break
    head: str = ""             # the chain head after the last verified entry
    sealed_through: int = 0    # highest entry count attested by a seal in this file
    unsealed_tail: int = 0     # entries past the last seal: how many could be dropped
                               # unnoticed without an external anchor

    def summary(self) -> str:
        if self.ok:
            s = f"OK: {self.count} entries chained (chain_id={self.chain_id[:12]}…"
            s += ", keyed" if self.keyed else ""
            if self.seals:
                s += f", {self.seals_verified}/{self.seals} signed seals"
            if self.unsealed_tail:
                s += f", {self.unsealed_tail} unsealed trailing entries"
            if self.truncated_tail:
                s += ", trailing partial line ignored"
            return s + ")"
        return f"BROKEN at entry {self.broken_at}: {self.reason}"


# ---------------------------------------------------------------------------
# Optional Ed25519 signing (layer 3). Import guarded so the engine's hard
# dependency set stays: rich/prompt_toolkit/mcp/pydantic/pyyaml/httpx.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability is environment-dependent
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization

    _HAVE_ED25519 = True
except Exception:  # pragma: no cover
    _HAVE_ED25519 = False


def ed25519_available() -> bool:
    return _HAVE_ED25519


def seal_material(chain_id: str, head: str, count: int) -> bytes:
    """The bytes an Ed25519 seal signs: a checkpoint of (chain_id, head, count)."""
    return canonical({"chain_id": chain_id, "head": head, "count": count}).encode("utf-8")


def load_private_key(raw: str | bytes):
    """Load an Ed25519 private key from PEM text/bytes or 32-byte hex/raw. Returns None
    if unavailable/unparseable so callers degrade to an unsigned checkpoint."""
    if not _HAVE_ED25519:
        return None
    try:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("-----BEGIN"):
                return serialization.load_pem_private_key(raw.encode(), password=None)
            b = bytes.fromhex(raw)
        else:
            b = raw
        if len(b) == 32:
            return Ed25519PrivateKey.from_private_bytes(b)
        return serialization.load_pem_private_key(b, password=None)
    except Exception:
        return None


def public_key_hex(private_key) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw.hex()


def sign_seal(private_key, chain_id: str, head: str, count: int) -> tuple[str, str]:
    """Return (signature_hex, public_key_hex) for a checkpoint."""
    sig = private_key.sign(seal_material(chain_id, head, count))
    return sig.hex(), public_key_hex(private_key)


def verify_seal_sig(pub_hex: str, sig_hex: str, chain_id: str, head: str, count: int) -> bool:
    if not _HAVE_ED25519:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pub.verify(bytes.fromhex(sig_hex), seal_material(chain_id, head, count))
        return True
    except Exception:
        return False


@dataclass
class ChainState:
    """The append cursor: what the next entry chains onto."""

    chain_id: str
    head: str
    count: int = 0
    keyed: bool = False
    extra: dict = field(default_factory=dict)

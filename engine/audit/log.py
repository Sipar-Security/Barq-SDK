"""Append-only, tamper-evident audit log.

Every request/response exchange and every permission decision is written here,
timestamped, with a stable id. Two jobs:
  1. A durable, tamper-evident record of what the agent did.
  2. An evidence store: `log_exchange()` returns an id that later output can reference to
     bind a claim to the exact request->response that produced it.

JSONL, fsync'd per line, never rewritten. On top of "durable", every entry is now
**chained**: it carries the hash of the entry before it, so any insertion, deletion,
reordering, or in-place edit is detectable (`verify()`), and — with an HMAC key or an
Ed25519 seal — unforgeable by whoever holds the file. See integrity.py for the layers.

The public API (`log_exchange`/`log_decision`/`log_blocked`/`note`/`read_all`/`has`/
`close`) is unchanged, so this is a drop-in for every existing caller; the integrity
fields are additive and legacy (unchained) logs still read back.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from . import integrity as I
from .redact import default_redactor

HEADER_KIND = "audit-open"
SEAL_KIND = "audit-seal"

# --- cross-process append lock ------------------------------------------------
# The hash chain has ONE head. A threading.Lock only serialises writers inside a single
# AuditLog object; two Agents, a worker pool, or two processes on the same file each cache
# their own head, interleave, and fork the chain permanently — verify() then fails forever
# with no error at write time. An OS file lock plus a head re-read under that lock makes
# concurrent appenders safe.
try:  # POSIX
    import fcntl

    def _lock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock_file(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
except ImportError:  # Windows
    import msvcrt

    # msvcrt.locking locks a byte range from the CURRENT position, so every writer must lock
    # the SAME offset (byte 0) or they exclude nothing. LK_LOCK gives up after ~10s, so retry.
    def _lock_file(fh) -> None:
        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.01)

    def _unlock_file(fh) -> None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


def _tail_hash(path: Path) -> Optional[str]:
    """The `hash` of the last complete entry on disk, or None. Reads only the tail, so a
    head re-sync before every append does not cost O(file)."""
    try:
        size = path.stat().st_size
        if size == 0:
            return None
        with path.open("rb") as fh:
            window = min(size, 65536)
            fh.seek(size - window)
            chunk = fh.read(window)
    except OSError:
        return None
    for line in reversed(chunk.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial final write
        h = d.get("hash")
        if h:
            return h
    return None


@dataclass(frozen=True)
class AuditEntry:
    id: str
    ts: float
    kind: str  # "exchange" | "decision" | "blocked" | "note" | "audit-open" | "audit-seal"
    data: dict = field(default_factory=dict)
    prev: str = ""   # hash of the previous entry (genesis for the header)
    hash: str = ""   # H(core || prev); the chain link

    def core(self) -> dict:
        return I.entry_core(self.id, self.ts, self.kind, self.data)


def _coerce_key(key: str | bytes | None) -> Optional[bytes]:
    if key is None:
        return None
    return key.encode("utf-8") if isinstance(key, str) else key


class AuditLog:
    def __init__(
        self,
        path: str | Path,
        *,
        hmac_key: str | bytes | None = None,
        signing_key: str | bytes | None = None,
        chain_id: str | None = None,
        redactor: Any = default_redactor,
    ) -> None:
        """`hmac_key` (layer 2) makes the chain unforgeable without the key; keep it in a
        secret store, not the workdir. `signing_key` (layer 3, Ed25519) is used by
        `seal()` for third-party-verifiable checkpoints. Both are optional — with neither,
        the plain SHA-256 chain still detects tampering."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Applied to every record BEFORE hashing, so the chain covers the redacted bytes and
        # evidence verification still checks what was actually stored. Pass redactor=None to
        # keep raw payloads (only appropriate when the log itself is a secret store).
        self._redactor = redactor
        self._hmac_key = _coerce_key(hmac_key)
        self._signing_raw = signing_key
        self._fh = None
        # The hash chain is a single serial thread: prev/head must advance atomically with the
        # write. Callers today append sequentially (one Coordinator at a time, subagents run to
        # completion before the parent resumes), but a shared AuditLog across concurrent
        # subagents would otherwise interleave writes and fork the chain. This makes _append
        # atomic so that stays impossible regardless of how it's driven.
        self._lock = threading.Lock()
        # The OS lock is taken on a dedicated sidecar file, never on the log itself: on
        # Windows a byte-range lock on the data file makes concurrent READERS fail with
        # PermissionError, which would break verify()/read_all() while a write is in flight.
        self._lockpath = self.path.with_name(self.path.name + ".lock")
        self._lockfh = self._lockpath.open("a+", encoding="utf-8")

        state = self._bootstrap(chain_id)
        self.chain_id = state.chain_id
        self._head = state.head
        self._count = state.count
        self._keyed = state.keyed
        self._fh = self.path.open("a", encoding="utf-8")

        # A brand-new (or legacy, unheadered) file needs its anchor written now, so the
        # first real entry chains onto a header that records chain_id + keying.
        if state.extra.get("needs_header"):
            self._append(
                HEADER_KIND,
                {
                    "chain_id": self.chain_id,
                    "version": I.CHAIN_VERSION,
                    "created": time.time(),
                    "hmac": self._hmac_key is not None,
                    "legacy_prefix": state.extra.get("legacy_prefix", 0),
                },
            )

    # ---- bootstrap / resume ------------------------------------------------
    def _bootstrap(self, chain_id: str | None) -> "I.ChainState":
        """Decide where the chain resumes. Three cases:
          * empty/new file            -> open a fresh chain (write header on first append)
          * chained file (has header) -> resume from the last entry's hash
          * legacy file (no header)   -> keep it, open a NEW chain segment after it
        """
        entries = self._read_raw()
        if not entries:
            cid = chain_id or uuid.uuid4().hex
            return I.ChainState(
                chain_id=cid, head=I.genesis_hash(cid), count=0,
                extra={"needs_header": True},
            )

        headers = [e for e in entries if e.kind == HEADER_KIND and e.hash]
        last = entries[-1]
        if headers and last.hash:
            hdr = headers[-1]
            cid = str(hdr.data.get("chain_id") or "")
            keyed = bool(hdr.data.get("hmac"))
            if keyed and self._hmac_key is None:
                raise ValueError(
                    f"{self.path} is an HMAC-keyed audit chain; its key must be supplied "
                    "to append to it (audit integrity would otherwise break)."
                )
            if not keyed and self._hmac_key is not None:
                raise ValueError(
                    f"{self.path} is an unkeyed audit chain; appending with an HMAC key "
                    "would fork the chain. Open it without hmac_key."
                )
            return I.ChainState(chain_id=cid, head=last.hash, count=len(entries), keyed=keyed)

        # Legacy (no chain header): don't rewrite history — start a new chain after it.
        cid = chain_id or uuid.uuid4().hex
        return I.ChainState(
            chain_id=cid, head=I.genesis_hash(cid), count=0,
            extra={"needs_header": True, "legacy_prefix": len(entries)},
        )

    # ---- append path -------------------------------------------------------
    def _append(self, kind: str, data: dict) -> str:
        """Append one chained entry.

        Serialised twice over: a thread lock for writers inside this process, and an OS file
        lock for writers outside it. Under the file lock the head is re-read from disk, so an
        entry appended by another AuditLog object (another Agent, another process) is chained
        onto rather than overwritten — which is what used to fork the chain irrecoverably.
        """
        if self._redactor is not None:
            data = self._redactor(data)
        with self._lock:
            _lock_file(self._lockfh)
            try:
                disk_head = _tail_hash(self.path)
                if disk_head is not None and disk_head != self._head:
                    self._head = disk_head  # another writer advanced the chain; follow it
                entry_id = uuid.uuid4().hex
                ts = time.time()
                core = I.entry_core(entry_id, ts, kind, data)
                h = I.compute_hash(core, self._head, self._hmac_key)
                entry = AuditEntry(
                    id=entry_id, ts=ts, kind=kind, data=data, prev=self._head, hash=h
                )
                self._fh.seek(0, os.SEEK_END)
                self._fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
                self._fh.flush()
                os.fsync(self._fh.fileno())
            finally:
                _unlock_file(self._lockfh)
            self._head = h
            self._count += 1
            return entry_id

    # ---- public logging API (unchanged signatures) -------------------------
    def log_exchange(self, method: str, url: str, request: dict, response: dict) -> str:
        """Record a reproduced request/response. Returns the evidence id.

        The exact bytes of `request` and `response` are content-addressed (sha256 of their
        canonical form) and stored *inside* the hashed entry. A claim can therefore bind to
        the precise request->response that produced it, not to a URL substring: tampering
        with either payload breaks both the content hash and the chain."""
        req_sha = I.sha256_hex(I.canonical(request))
        resp_sha = I.sha256_hex(I.canonical(response))
        return self._append(
            "exchange",
            {
                "method": method,
                "url": url,
                "request": request,
                "response": response,
                "req_sha256": req_sha,
                "resp_sha256": resp_sha,
            },
        )

    def log_decision(self, tool: str, behavior: str, reason: str) -> str:
        return self._append("decision", {"tool": tool, "behavior": behavior, "reason": reason})

    def log_blocked(self, tool: str, target: str, reason: str) -> str:
        """A policy block — feeds alerting/metrics on denied tool calls."""
        return self._append("blocked", {"tool": tool, "target": target, "reason": reason})

    def note(self, text: str) -> str:
        return self._append("note", {"text": text})

    # ---- sealing (layer 3) -------------------------------------------------
    def seal(self, signing_key: str | bytes | None = None) -> str:
        """Write a checkpoint over the current head+count. If an Ed25519 signing key is
        available (arg, constructor, or $BBENGINE_AUDIT_SIGNING_KEY) the checkpoint is
        signed so a third party can verify chain-of-custody against the operator's public
        key. Without a key it is still a chained, timestamped checkpoint. Returns its id."""
        raw = signing_key or self._signing_raw or os.environ.get("BBENGINE_AUDIT_SIGNING_KEY")
        head, count = self._head, self._count
        data: dict = {"head": head, "count": count}
        priv = I.load_private_key(raw) if raw else None
        if priv is not None:
            sig, pub = I.sign_seal(priv, self.chain_id, head, count)
            data.update({"alg": "ed25519", "sig": sig, "pub": pub})
        else:
            data.update({"alg": "hmac-sha256" if self._hmac_key else "none"})
        return self._append(SEAL_KIND, data)

    # ---- verification ------------------------------------------------------
    def verify(self) -> "I.VerifyReport":
        """Re-walk the on-disk chain and confirm every link. O(n), reads the file fresh."""
        return verify_audit_file(self.path, hmac_key=self._hmac_key)

    def evidence_digest(self, evidence_id: str) -> Optional[dict]:
        """The content-addresses to record to bind to this exact evidence:
        the entry's chain hash plus the request/response sha256s."""
        for e in self._read_raw():
            if e.id == evidence_id:
                return {
                    "chain_id": self.chain_id,
                    "entry_hash": e.hash,
                    "req_sha256": e.data.get("req_sha256"),
                    "resp_sha256": e.data.get("resp_sha256"),
                }
        return None

    def verify_evidence(
        self, evidence_id: str, *, request: dict | None = None, response: dict | None = None
    ) -> bool:
        """Confirm that the given request/response bytes are byte-identical to what the
        audit entry recorded — structural evidence binding, not a URL match. Passing only
        one of request/response verifies just that side."""
        dig = self.evidence_digest(evidence_id)
        if dig is None:
            return False
        if request is not None and I.sha256_hex(I.canonical(request)) != dig.get("req_sha256"):
            return False
        if response is not None and I.sha256_hex(I.canonical(response)) != dig.get("resp_sha256"):
            return False
        return True

    # ---- reading -----------------------------------------------------------
    def _read_raw(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        out: list[AuditEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(
                    AuditEntry(
                        id=d["id"], ts=d["ts"], kind=d["kind"],
                        data=d.get("data", {}), prev=d.get("prev", ""), hash=d.get("hash", ""),
                    )
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                # A partial write (crash mid-fsync) can leave one bad line. Skip it rather
                # than making the whole evidence log unreadable.
                continue
        return out

    def read_all(self) -> list[AuditEntry]:
        """Every non-structural entry, in order (hides the chain header/seal bookkeeping
        so existing callers see exactly the exchanges/decisions/blocks/notes they did)."""
        return [e for e in self._read_raw() if e.kind not in (HEADER_KIND, SEAL_KIND)]

    def has(self, evidence_id: str) -> bool:
        return any(e.id == evidence_id for e in self._read_raw())

    @property
    def head(self) -> str:
        return self._head

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        lockfh = getattr(self, "_lockfh", None)
        if lockfh is not None and not lockfh.closed:
            lockfh.close()


# ---------------------------------------------------------------------------
# Detached verifier: works on just the JSONL file, no live AuditLog needed.
# ---------------------------------------------------------------------------
def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        s = line.strip()
        if s:
            yield i, s


def verify_audit_file(
    path: str | Path, *, hmac_key: str | bytes | None = None
) -> "I.VerifyReport":
    """Verify a chained audit file end-to-end. Detects any edit/insert/delete/reorder,
    tolerates a legacy (pre-chain) prefix and a truncated final line, and checks any
    Ed25519 seals against their embedded public key.

    Returns a VerifyReport whose `.ok` is True only if every chained entry links.
    """
    path = Path(path)
    key = _coerce_key(hmac_key)
    if not path.exists():
        return I.VerifyReport(ok=False, reason="file does not exist")

    raw = list(_iter_lines(path))
    if not raw:
        return I.VerifyReport(ok=True, count=0, reason="empty")

    # Parse; remember if the FINAL line is unparseable (crash truncation, tolerated).
    parsed: list[tuple[int, dict]] = []
    truncated_tail = False
    for idx, (lineno, s) in enumerate(raw):
        try:
            parsed.append((lineno, json.loads(s)))
        except json.JSONDecodeError:
            if idx == len(raw) - 1:
                truncated_tail = True  # only the very last line may be a partial write
            else:
                return I.VerifyReport(
                    ok=False, broken_at=lineno,
                    reason=f"unparseable entry at line {lineno} (not the final line)",
                )

    # Locate the chain header. Everything before it is a legacy prefix we don't verify.
    header_pos = next(
        (i for i, (_, d) in enumerate(parsed) if d.get("kind") == HEADER_KIND and d.get("hash")),
        None,
    )
    if header_pos is None:
        return I.VerifyReport(
            ok=False, count=len(parsed), truncated_tail=truncated_tail,
            reason="no chain header (audit-open); log is legacy/unchained",
        )

    header = parsed[header_pos][1]
    chain_id = str(header.get("data", {}).get("chain_id") or "")
    keyed = bool(header.get("data", {}).get("hmac"))
    if keyed and key is None:
        return I.VerifyReport(
            ok=False, chain_id=chain_id, keyed=True,
            reason="chain is HMAC-keyed; supply hmac_key to verify it",
        )
    if not keyed:
        key = None  # an unkeyed chain must be verified with plain SHA-256

    expected_prev = I.genesis_hash(chain_id)
    count = 0
    seals = 0
    seals_verified = 0
    for _, d in parsed[header_pos:]:
        entry = AuditEntry(
            id=d.get("id", ""), ts=d.get("ts", 0.0), kind=d.get("kind", ""),
            data=d.get("data", {}), prev=d.get("prev", ""), hash=d.get("hash", ""),
        )
        if entry.prev != expected_prev:
            return I.VerifyReport(
                ok=False, count=count, chain_id=chain_id, broken_at=count, keyed=keyed,
                reason=f"entry {entry.id[:12]} prev={entry.prev[:12]}… != expected "
                       f"{expected_prev[:12]}… (insert/delete/reorder)",
                truncated_tail=truncated_tail,
            )
        recomputed = I.compute_hash(entry.core(), entry.prev, key)
        if recomputed != entry.hash:
            return I.VerifyReport(
                ok=False, count=count, chain_id=chain_id, broken_at=count, keyed=keyed,
                reason=f"entry {entry.id[:12]} hash mismatch (content edited)",
                truncated_tail=truncated_tail,
            )
        if entry.kind == SEAL_KIND:
            seals += 1
            sd = entry.data
            # The seal must attest the head that immediately precedes it.
            if sd.get("head") == entry.prev and sd.get("alg") == "ed25519":
                if I.verify_seal_sig(
                    sd.get("pub", ""), sd.get("sig", ""), chain_id,
                    sd.get("head", ""), int(sd.get("count", 0)),
                ):
                    seals_verified += 1
        expected_prev = entry.hash
        count += 1

    return I.VerifyReport(
        ok=True, count=count, chain_id=chain_id, keyed=keyed,
        seals=seals, seals_verified=seals_verified, truncated_tail=truncated_tail,
    )

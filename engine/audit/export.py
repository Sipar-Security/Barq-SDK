"""SIEM export for the audit log.

An enterprise audit trail is not a private file: it forwards to the client's SIEM
(Splunk / Elastic / Datadog / QRadar) where it becomes tamper-evident by being outside
the operator's reach, and searchable next to the client's own telemetry. Two lingua
francas cover essentially every SIEM:

  * ECS: Elastic Common Schema, JSON. Splunk, Elastic, Datadog, OpenSearch ingest it.
  * CEF: ArcSight Common Event Format, key=value. ArcSight, QRadar, and most legacy
           SIEMs speak it.

Each exported record carries the entry's chain hash (`event.hash`) so a SIEM-side
correlation can prove the forwarded copy matches the on-disk chain. Structural
bookkeeping entries (chain header/seal) are emitted too (a seal in the SIEM is exactly
the chain-of-custody checkpoint a triage team wants).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .log import AuditEntry, SEAL_KIND, HEADER_KIND, _iter_lines

# Map audit kinds to ECS event.category / a numeric CEF severity (0-10).
_ECS_CATEGORY = {
    "exchange": ["web"],
    "decision": ["configuration"],
    "blocked": ["network", "intrusion_detection"],
    "note": ["process"],
    HEADER_KIND: ["configuration"],
    SEAL_KIND: ["configuration"],
}
_CEF_SEVERITY = {"exchange": 3, "decision": 2, "blocked": 6, "note": 1,
                 HEADER_KIND: 1, SEAL_KIND: 4}


def _ts_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def to_ecs(entry: AuditEntry) -> dict:
    """One audit entry as an Elastic Common Schema document."""
    d = entry.data
    doc: dict = {
        "@timestamp": _ts_iso(entry.ts),
        "event": {
            "id": entry.id,
            "kind": "event",
            "action": entry.kind,
            "category": _ECS_CATEGORY.get(entry.kind, ["process"]),
            "hash": entry.hash,
        },
        "labels": {"chain_prev": entry.prev},
        "observer": {"vendor": "agent-engine", "product": "engine"},
    }
    if entry.kind == "exchange":
        doc["url"] = {"full": d.get("url", "")}
        doc["http"] = {
            "request": {"method": d.get("method", "")},
            "response": {"status_code": (d.get("response") or {}).get("status")},
        }
        doc["labels"].update(
            {"req_sha256": d.get("req_sha256", ""), "resp_sha256": d.get("resp_sha256", "")}
        )
    elif entry.kind == "blocked":
        doc["event"]["outcome"] = "failure"
        doc["destination"] = {"address": d.get("target", "")}
        doc["message"] = f"{d.get('tool','')} blocked {d.get('target','')}: {d.get('reason','')}"
    elif entry.kind == "decision":
        doc["message"] = f"{d.get('tool','')}: {d.get('behavior','')} ({d.get('reason','')})"
    elif entry.kind == "note":
        doc["message"] = d.get("text", "")
    elif entry.kind == SEAL_KIND:
        doc["message"] = f"audit seal alg={d.get('alg','')} count={d.get('count','')}"
        doc["labels"]["seal_head"] = d.get("head", "")
    return doc


# CEF extension values must escape backslash, equals, and newlines; the header fields
# (after the 6 pipes) escape backslash and pipe.
def _cef_ext(v: object) -> str:
    return str(v).replace("\\", "\\\\").replace("=", "\\=").replace("\n", "\\n").replace("\r", "")


def _cef_hdr(v: object) -> str:
    return str(v).replace("\\", "\\\\").replace("|", "\\|")


def to_cef(entry: AuditEntry) -> str:
    """One audit entry as an ArcSight CEF line."""
    d = entry.data
    sev = _CEF_SEVERITY.get(entry.kind, 1)
    name = {
        "exchange": "Target HTTP exchange",
        "decision": "Permission decision",
        "blocked": "Out-of-scope egress blocked",
        "note": "Audit note",
        HEADER_KIND: "Audit chain opened",
        SEAL_KIND: "Audit chain seal",
    }.get(entry.kind, entry.kind)

    ext: dict[str, object] = {
        "rt": int(entry.ts * 1000),          # CEF receipt time is epoch millis
        "externalId": entry.id,
        "cs1Label": "chainHash", "cs1": entry.hash,
        "cs2Label": "chainPrev", "cs2": entry.prev,
    }
    if entry.kind == "exchange":
        ext["requestMethod"] = d.get("method", "")
        ext["request"] = d.get("url", "")
        ext["cs3Label"], ext["cs3"] = "reqSha256", d.get("req_sha256", "")
        ext["cs4Label"], ext["cs4"] = "respSha256", d.get("resp_sha256", "")
    elif entry.kind == "blocked":
        ext["dhost"] = d.get("target", "")
        ext["msg"] = d.get("reason", "")
        ext["act"] = "block"
    elif entry.kind == "decision":
        ext["act"] = d.get("behavior", "")
        ext["msg"] = d.get("reason", "")
    elif entry.kind == "note":
        ext["msg"] = d.get("text", "")
    elif entry.kind == SEAL_KIND:
        ext["msg"] = f"alg={d.get('alg','')} count={d.get('count','')}"

    ext_str = " ".join(f"{k}={_cef_ext(v)}" for k, v in ext.items())
    return (
        f"CEF:0|agent-engine|engine|{_cef_hdr(1)}|{_cef_hdr(entry.kind)}|"
        f"{_cef_hdr(name)}|{sev}|{ext_str}"
    )


def _read_entries(path: str | Path) -> Iterator[AuditEntry]:
    for _, s in _iter_lines(Path(path)):
        try:
            o = json.loads(s)
            yield AuditEntry(
                id=o.get("id", ""), ts=o.get("ts", 0.0), kind=o.get("kind", ""),
                data=o.get("data", {}), prev=o.get("prev", ""), hash=o.get("hash", ""),
            )
        except json.JSONDecodeError:
            continue


def export_ecs(path: str | Path) -> Iterator[str]:
    """Yield newline-delimited ECS JSON (one doc per line): pipe to Filebeat/HEC."""
    for e in _read_entries(path):
        yield json.dumps(to_ecs(e), ensure_ascii=False, separators=(",", ":"))


def export_cef(path: str | Path) -> Iterator[str]:
    """Yield CEF lines: pipe to a syslog forwarder."""
    for e in _read_entries(path):
        yield to_cef(e)

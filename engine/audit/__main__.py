"""Detached audit CLI: verify chain-of-custody or forward to a SIEM.

    python -m engine.audit verify   <audit.jsonl> [--hmac-key-env VAR | --hmac-key HEX]
    python -m engine.audit export   <audit.jsonl> --format ecs|cef [-o OUT]

`verify` exits non-zero if the chain is broken — wire it into CI or a pre-report gate.
It reads only the JSONL file, so a client can run it against a log you hand them without
the engine, and (for a keyed chain) with the key you disclose out of band.
"""

from __future__ import annotations

import argparse
import os
import sys

from .export import export_cef, export_ecs
from .log import verify_audit_file


def _resolve_key(args) -> str | None:
    if getattr(args, "hmac_key", None):
        return args.hmac_key
    if getattr(args, "hmac_key_env", None):
        return os.environ.get(args.hmac_key_env)
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m engine.audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify the tamper-evident chain")
    v.add_argument("path")
    v.add_argument("--hmac-key", help="HMAC key (hex/utf-8) for a keyed chain")
    v.add_argument("--hmac-key-env", help="env var holding the HMAC key")

    e = sub.add_parser("export", help="export to a SIEM format")
    e.add_argument("path")
    e.add_argument("--format", choices=["ecs", "cef"], required=True)
    e.add_argument("-o", "--out", help="output file (default: stdout)")

    args = p.parse_args(argv)

    if args.cmd == "verify":
        report = verify_audit_file(args.path, hmac_key=_resolve_key(args))
        print(report.summary())
        return 0 if report.ok else 1

    if args.cmd == "export":
        lines = export_ecs(args.path) if args.format == "ecs" else export_cef(args.path)
        sink = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
        try:
            for line in lines:
                sink.write(line + "\n")
        finally:
            if args.out:
                sink.close()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

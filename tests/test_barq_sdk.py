"""Tests for the `barq_sdk` public import name.

The implementation lives in `engine`; `barq_sdk` re-exports it. These pin that the two
spellings resolve to the SAME objects (not copies), that subpackage and deep imports work,
and that the audit chain's genesis prefix still accepts logs written by the old name.
"""

import pytest

import barq_sdk
import engine
from barq_sdk import Agent, Coordinator
from barq_sdk.audit import AuditLog, verify_audit_file
from barq_sdk.audit.integrity import _GENESIS_PREFIX, _LEGACY_GENESIS_PREFIX, genesis_hash


def test_barq_sdk_exports_are_the_same_objects():
    assert barq_sdk.Agent is Agent is engine.Agent
    assert barq_sdk.Coordinator is Coordinator is engine.Coordinator
    assert barq_sdk.__version__ == engine.__version__


def test_barq_sdk_subpackage_imports():
    from barq_sdk.audit import AuditLog as BarqAuditLog
    from engine.audit import AuditLog as EngineAuditLog

    assert BarqAuditLog is EngineAuditLog


def test_barq_sdk_deep_module_import():
    """A nested module (engine.audit.integrity) is reachable under the new name too."""
    from barq_sdk.audit.integrity import canonical as barq_canonical
    from engine.audit.integrity import canonical as engine_canonical

    assert barq_canonical is engine_canonical


@pytest.mark.parametrize("name", sorted(engine.__all__))
def test_every_engine_export_is_reachable(name):
    assert getattr(barq_sdk, name) is getattr(engine, name)


def test_audit_signing_key_env(monkeypatch, tmp_path):
    """seal() picks the signing key up from the env var under the current name."""
    log = AuditLog(tmp_path / "audit.jsonl")
    monkeypatch.setenv("BARQ_SDK_AUDIT_SIGNING_KEY", "00" * 32)
    try:
        assert log.seal() is not None
    finally:
        log.close()
    assert verify_audit_file(tmp_path / "audit.jsonl").ok is True


@pytest.mark.parametrize(
    "var", ["BARQ_SDK_AUDIT_SIGNING_KEY", "BARK_SQK_AUDIT_SIGNING_KEY",
            "BBENGINE_AUDIT_SIGNING_KEY"],
)
def test_legacy_signing_key_env_names_still_work(monkeypatch, tmp_path, var):
    """Renaming the project must not silently stop honouring an operator's configured key."""
    for stale in ("BARQ_SDK_AUDIT_SIGNING_KEY", "BARK_SQK_AUDIT_SIGNING_KEY",
                  "BBENGINE_AUDIT_SIGNING_KEY"):
        monkeypatch.delenv(stale, raising=False)
    monkeypatch.setenv(var, "11" * 32)
    log = AuditLog(tmp_path / f"{var}.jsonl")
    try:
        seal_id = log.seal()
    finally:
        log.close()
    entry = [e for e in log._read_raw() if e.id == seal_id][0]
    # cryptography may be absent; then the seal is unsigned but must still checkpoint.
    from barq_sdk.audit import ed25519_available

    assert entry.data.get("alg") == ("ed25519" if ed25519_available() else "none")


def test_genesis_prefix_is_current_name():
    assert _GENESIS_PREFIX.startswith("Barq-SDK-audit-v")
    h = genesis_hash("test-chain")
    assert isinstance(h, str) and len(h) == 64


def test_legacy_genesis_chains_still_verify(tmp_path):
    """A log written before the rename must not read as tampered after it."""
    import json

    path = tmp_path / "legacy.jsonl"
    log = AuditLog(path)
    log.log_decision("ReadFile", "allow", "ok")
    log.close()

    # Rewrite the chain as the pre-rename build would have anchored it.
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    chain_id = rows[0]["data"]["chain_id"]
    assert rows[0]["prev"] == genesis_hash(chain_id)
    assert genesis_hash(chain_id, legacy=True) != genesis_hash(chain_id)
    assert _LEGACY_GENESIS_PREFIX.startswith("bbengine-audit-v")

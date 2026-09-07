"""Tests for Bark-SQK package compatibility, imports, environment variables, and genesis prefixing."""

import os
import tempfile
import pytest

import bark_sqk
from bark_sqk import Agent, Coordinator
from bark_sqk.audit import AuditLog, verify_audit_file
from bark_sqk.audit.integrity import genesis_hash, _GENESIS_PREFIX


def test_bark_sqk_exports():
    assert bark_sqk.Agent is Agent
    assert bark_sqk.Coordinator is Coordinator
    assert bark_sqk.__version__ == "0.1.0"


def test_bark_sqk_subpackage_imports():
    from bark_sqk.audit import AuditLog as BarkAuditLog
    from engine.audit import AuditLog as EngineAuditLog

    assert BarkAuditLog is EngineAuditLog


def test_bark_sqk_audit_signing_key_env(monkeypatch, tmp_path):
    log_file = tmp_path / "audit.jsonl"
    log = AuditLog(log_file)

    # Set BARK_SQK_AUDIT_SIGNING_KEY in environment
    # Sample 32-byte hex key for Ed25519 testing
    test_key = "00" * 32
    monkeypatch.setenv("BARK_SQK_AUDIT_SIGNING_KEY", test_key)

    seal_id = log.seal()
    assert seal_id is not None
    log.close()

    report = verify_audit_file(log_file)
    assert report.ok is True


def test_bark_sqk_genesis_prefix():
    assert _GENESIS_PREFIX.startswith("bark-sqk-audit-v")
    h = genesis_hash("test-chain")
    assert isinstance(h, str) and len(h) == 64

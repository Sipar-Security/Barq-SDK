from engine.audit import AuditLog
from engine.memory import Memory, MemoryStore, MemoryType


def test_audit_exchange_returns_referable_id(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    eid = log.log_exchange(
        "GET", "https://api.example.com/x?id=1",
        request={"headers": {}}, response={"status": 200, "body": "ok"},
    )
    assert log.has(eid)
    # the exact request/response bytes are content-addressed and verifiable by id
    assert log.verify_evidence(eid, response={"status": 200, "body": "ok"})
    assert not log.verify_evidence(eid, response={"status": 200, "body": "tampered"})


def test_audit_chain_verifies_and_detects_tamper(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    log.log_decision("HttpRequest", "allow", "in policy")
    log.log_decision("Bash", "deny", "danger")
    log.close()
    assert AuditLog(p).verify().ok
    # flip a byte in the middle of the file -> chain no longer verifies
    raw = p.read_text(encoding="utf-8")
    p.write_text(raw.replace("in policy", "in POLICY", 1), encoding="utf-8")
    from engine.audit import verify_audit_file
    assert not verify_audit_file(p).ok


def test_audit_is_append_only_across_instances(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(p)
    a.log_blocked("Fetch", "evil.example.com", "out of policy")
    a.close()
    b = AuditLog(p)
    b.note("resumed")
    b.close()
    entries = AuditLog(p).read_all()
    assert len(entries) == 2
    assert entries[0].kind == "blocked" and entries[1].kind == "note"


def test_memory_save_load_and_index(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    store.save(Memory(
        name="deploy command",
        description="how the service is deployed",
        type=MemoryType.PROJECT,
        body="Deploy with `make release`.\n**Why:** documented in runbook.",
    ))
    back = store.load("deploy-command")
    assert back.type is MemoryType.PROJECT
    assert "make release" in back.body
    idx = (tmp_path / "mem" / "MEMORY.md").read_text()
    assert "deploy-command.md" in idx


def test_memory_no_duplicate_index_lines(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    store.save(Memory("dup", "first desc", MemoryType.PROJECT, "b"))
    store.save(Memory("dup", "second desc", MemoryType.PROJECT, "b2"))
    idx = (tmp_path / "mem" / "MEMORY.md").read_text()
    assert idx.count("(dup.md)") == 1
    assert "second desc" in idx


def test_memory_find_relevant(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    store.save(Memory("scanner", "the api rejects automated scanners", MemoryType.FEEDBACK, "x"))
    store.save(Memory("tracker", "bugs tracked in the issue tracker", MemoryType.REFERENCE, "y"))
    hits = store.find_relevant("can I scan the api")
    assert hits and hits[0].name == "scanner"

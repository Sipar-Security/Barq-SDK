"""Shared-file stores under concurrency.

`MemoryStore` had no lock at all while `AuditLog` had one, and the `Agent` facade puts both
in the same control-plane directory and exposes `memory_dir=` so a shared store is a
documented configuration. Two reproduced failures:

  * Windows: `os.replace` onto MEMORY.md raised `PermissionError: [WinError 5]` when another
    writer held it. The exception propagated out of `save()` and killed the calling thread.
  * Every platform: `_add_index_line` was an unlocked read-modify-write of the whole index,
    so concurrent saves lost entries. Measured at one dropped pointer per ~80 saves — and a
    memory whose index line is gone still exists on disk but is invisible to anything that
    reads MEMORY.md.

These run real threads against real files. They are the only tests here that would pass by
luck on a single run, so each uses enough writers and iterations to make the pre-fix
failure essentially certain.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from engine.audit import AuditLog
from engine.audit.log import verify_audit_file
from engine.filelock import atomic_write_text, file_lock
from engine.memory import Memory, MemoryStore, MemoryType


def index_lines(root) -> list[str]:
    path = os.path.join(str(root), "MEMORY.md")
    if not os.path.exists(path):
        return []
    return [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]


def memory_files(root) -> list[str]:
    return [f for f in os.listdir(str(root)) if f.endswith(".md") and f != "MEMORY.md"]


def mem(name: str) -> Memory:
    return Memory(name=name, description=f"about {name}", type=MemoryType.PROJECT, body="body")


# --- the lock primitive ------------------------------------------------------
def test_file_lock_serialises_writers(tmp_path):
    target = tmp_path / "shared.txt"
    order: list[str] = []
    barrier = threading.Barrier(4)

    def worker(tag: str) -> None:
        barrier.wait()
        with file_lock(target):
            order.append(f"{tag}-in")
            # If the lock did not hold, another thread would interleave here.
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            atomic_write_text(target, existing + tag + "\n")
            order.append(f"{tag}-out")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, "abcd"))

    # Every in is immediately followed by its own out: no interleaving.
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0], order
    assert sorted(target.read_text(encoding="utf-8").split()) == ["a", "b", "c", "d"]


def test_lock_is_taken_on_a_sidecar_not_the_target(tmp_path):
    """Locking the file being replaced is what makes `os.replace` fail on Windows."""
    target = tmp_path / "data.txt"
    with file_lock(target):
        assert (tmp_path / "data.txt.lock").exists()
        atomic_write_text(target, "written while locked")
    assert target.read_text(encoding="utf-8") == "written while locked"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.txt"
    for i in range(20):
        atomic_write_text(target, str(i))
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]


def test_atomic_write_is_all_or_nothing(tmp_path):
    target = tmp_path / "x.txt"
    atomic_write_text(target, "original")
    with pytest.raises(Exception):
        atomic_write_text(target, object())  # type: ignore[arg-type]
    assert target.read_text(encoding="utf-8") == "original"


# --- MemoryStore -------------------------------------------------------------
def test_concurrent_saves_never_raise(tmp_path):
    """On Windows this raised PermissionError straight out of save()."""
    stores = [MemoryStore(tmp_path) for _ in range(4)]
    errors: list[str] = []

    def worker(args) -> None:
        store, prefix = args
        for i in range(30):
            try:
                store.save(mem(f"{prefix}-{i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, zip(stores, "abcd")))

    assert errors == []


def test_no_index_entry_is_lost_under_concurrency(tmp_path):
    """The lost-update race: both writers read the old index, both append, the second
    replace discards the first."""
    stores = [MemoryStore(tmp_path) for _ in range(4)]

    def worker(args) -> None:
        store, prefix = args
        for i in range(30):
            store.save(mem(f"{prefix}-{i}"))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, zip(stores, "abcd")))

    files = memory_files(tmp_path)
    lines = index_lines(tmp_path)
    assert len(files) == 120
    assert len(lines) == len(files), (
        f"{len(files) - len(lines)} memories exist on disk but have no index pointer"
    )


def test_non_overwrite_collision_is_atomic(tmp_path):
    """`overwrite=False` picks a free `slug-N`. Checking existence outside the lock let two
    concurrent saves choose the same suffix and one silently overwrite the other - the exact
    collision the flag exists to prevent."""
    stores = [MemoryStore(tmp_path) for _ in range(4)]

    def worker(store) -> None:
        for _ in range(15):
            store.save(mem("same-name"), overwrite=False)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, stores))

    assert len(memory_files(tmp_path)) == 60


def test_concurrent_save_and_delete_keep_the_index_consistent(tmp_path):
    store = MemoryStore(tmp_path)
    for i in range(40):
        store.save(mem(f"item-{i}"))

    savers = [MemoryStore(tmp_path) for _ in range(2)]
    deleters = [MemoryStore(tmp_path) for _ in range(2)]

    def save_more(store) -> None:
        for i in range(40, 60):
            store.save(mem(f"item-{i}"))

    def delete_some(store) -> None:
        for i in range(0, 20):
            store.delete(f"item-{i}")

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(save_more, s) for s in savers]
        futures += [pool.submit(delete_some, s) for s in deleters]
        for f in futures:
            f.result()

    files = set(memory_files(tmp_path))
    pointers = {l.split("](")[1].split(")")[0] for l in index_lines(tmp_path)}
    assert pointers == files, (
        f"index and disk disagree: only on disk {files - pointers}, "
        f"only in index {pointers - files}"
    )


def test_recall_still_works_after_concurrent_writes(tmp_path):
    """Correctness of the index is only interesting if recall reads it."""
    stores = [MemoryStore(tmp_path) for _ in range(3)]

    def worker(args) -> None:
        store, prefix = args
        for i in range(20):
            store.save(
                Memory(
                    name=f"{prefix}-deploy-{i}",
                    description="how we deploy to production",
                    type=MemoryType.PROJECT,
                    body="use the blue-green pipeline",
                )
            )

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(worker, zip(stores, "abc")))

    found = MemoryStore(tmp_path).find_relevant("deployment pipeline", limit=100)
    assert len(found) == 60


# --- AuditLog: the property that was already correct, pinned ------------------
def test_audit_chain_survives_concurrent_writers(tmp_path):
    """The audit log's lock predates this work; the shared primitive must not weaken it."""
    path = tmp_path / "audit.jsonl"
    logs = [AuditLog(path) for _ in range(4)]

    def worker(args) -> None:
        log, prefix = args
        for i in range(40):
            log.log_decision(f"{prefix}{i}", "allow", "concurrent")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, zip(logs, "abcd")))
    for log in logs:
        log.close()

    report = verify_audit_file(path)
    assert report.ok, report.summary()
    assert sum(1 for _ in open(path, encoding="utf-8")) == 161  # 160 + the open record

"""Session persistence: crash-resume for the agent loop.

The coordinator's message history lives only in memory: a crash mid-run would lose the
entire reasoning + tool-call transcript and force a restart from turn 0. `SessionStore`
snapshots that transcript to disk after every turn boundary so a later run can pick it up
and continue.

Engine-general: it persists an opaque list of message dicts and a `done` flag. It knows
nothing about the task domain; any agent built on this engine can use it.

Storage format
--------------
Append-only JSONL: a `meta` line, then one line per message. Snapshotting used to serialise
the WHOLE transcript on every turn, so a run wrote O(n²) bytes in total. On a tool-heavy
run with large results that dominates the run's I/O for no benefit. Now a turn appends only
the messages that are actually new.

A full rewrite still happens when the transcript shrinks or diverges (which is exactly what
context compaction does (it replaces the list), so correctness never depends on the
transcript only ever growing. Writes are atomic (temp file + os.replace) so a crash during
a rewrite cannot leave a half-written, unparseable snapshot behind; appends are flushed and
fsync'd per turn.

v1 files (a single JSON object) still load, so an existing session resumes after upgrade.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


class SessionStore:
    VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._written = 0          # messages already on disk
        self._done: bool | None = None
        self._synced = False       # have we reconciled with the file yet?

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    # --- reading ------------------------------------------------------------
    def load(self) -> Optional[tuple[list[dict], bool]]:
        """Return (messages, done) or None if there is nothing usable to resume."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not raw.strip():
            return None

        # v1: one JSON object holding everything.
        stripped = raw.lstrip()
        if stripped.startswith("{") and "\n" not in stripped.rstrip().rstrip("\n"):
            return self._load_v1(raw)

        messages: list[dict] = []
        done = False
        saw_meta = False
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a partial final write
            if not isinstance(rec, dict):
                continue
            if rec.get("_") == "meta":
                saw_meta = True
                done = bool(rec.get("done", False))
            elif "role" in rec:
                messages.append(rec)
        if not saw_meta and not messages:
            return self._load_v1(raw)
        self._written = len(messages)
        self._done = done
        self._synced = True
        return messages, done

    def _load_v1(self, raw: str) -> Optional[tuple[list[dict], bool]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        messages = data.get("messages")
        if not isinstance(messages, list):
            return None
        done = bool(data.get("done", False))
        # Force the next save to rewrite in the new format rather than appending to a v1 file.
        self._written = 0
        self._done = None
        self._synced = False
        return messages, done

    # --- writing ------------------------------------------------------------
    def save(self, messages: list[dict], done: bool = False) -> None:
        if not self._synced:
            self._reconcile()
        # The transcript shrank or was replaced (compaction): the append log no longer
        # describes it, so rewrite from scratch.
        if len(messages) < self._written:
            self._rewrite(messages, done)
            return
        new = messages[self._written:]
        if not new and self._done == done:
            return  # nothing changed; don't touch the file
        self._append(new, done)

    def _reconcile(self) -> None:
        """Establish how much of `path` is already a valid append log for this store."""
        if not self.exists():
            self._written, self._done, self._synced = 0, None, True
            return
        loaded = self.load()
        if loaded is None:
            self._written, self._done, self._synced = 0, None, True

    def _append(self, new_messages: list[dict], done: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.exists()
        with self.path.open("a", encoding="utf-8") as fh:
            if fresh:
                fh.write(json.dumps({"_": "meta", "version": self.VERSION, "done": done},
                                    ensure_ascii=False) + "\n")
                self._done = done
            for m in new_messages:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            if done != self._done:
                # The LAST meta line wins, so flipping `done` costs one line, not a rewrite.
                fh.write(json.dumps({"_": "meta", "version": self.VERSION, "done": done},
                                    ensure_ascii=False) + "\n")
                self._done = done
            fh.flush()
            os.fsync(fh.fileno())
        self._written += len(new_messages)
        self._synced = True

    def _rewrite(self, messages: list[dict], done: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        lines = [json.dumps({"_": "meta", "version": self.VERSION, "done": done},
                            ensure_ascii=False)]
        lines += [json.dumps(m, ensure_ascii=False) for m in messages]
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on POSIX and Windows
        self._written = len(messages)
        self._done = done
        self._synced = True

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._written, self._done, self._synced = 0, None, True


class ToolJournal:
    """Records each tool_use result by its id so a resumed run executes each call AT MOST
    ONCE (exactly-once side effects).

    Message-boundary snapshots alone give at-least-once: a crash after a tool ran but
    before its result was persisted makes the resumed run re-execute it (a replayed POST
    could double-fire). The journal closes that: before dispatching a tool the coordinator
    checks here, and a call already recorded returns its stored result instead of running
    again. Append-only JSONL; the last record for an id wins.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._done: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._done[rec["id"]] = rec["result"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # skip a corrupt line rather than abort resume

    def get(self, tool_use_id: str) -> Optional[dict]:
        return self._done.get(tool_use_id)

    def record(self, tool_use_id: str, result: dict) -> None:
        self._done[tool_use_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": tool_use_id, "result": result}, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._done.clear()

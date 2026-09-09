"""File-based memory: a small durable store of facts worth carrying across runs.

Four types: user, feedback, project, reference. One fact per file with frontmatter; a
MEMORY.md index holds one pointer line per memory. This is durable *context*, not a log of
transient outputs, do not fill it with re-derivable noise (code patterns, architecture,
git history, ephemeral state, anything derivable by reading the current project state).

Recall
------
Scoring covers name, description AND body. Scoring only name+description (the previous
behaviour) meant a fact stated in the body, which is where facts actually live, was
unreachable unless the description happened to repeat it.

Recall is served from an in-memory index built once and refreshed only for files whose
mtime/size changed. A query then costs O(matching terms) -- but only while the cached
directory sweep is still trusted. Past `_STAT_TTL` the store re-scans to notice writes made
by another process, so a recall is O(corpus) at that point too. The sweep is one `scandir`
pass rather than a glob plus a stat per file, and the postings are rebuilt only when
something actually changed, which is what keeps that re-scan in the low milliseconds rather
than the hundreds (measured: 667ms -> 5ms on a 1,600-memory store).

Every method here does blocking filesystem work. Call the `a`-prefixed wrappers
(`asave`/`afind_relevant`/`adelete`/`aall`) from async code -- they run it on a worker
thread. Calling the sync versions inside `async def` blocks the whole event loop, which
stalls every other agent in the process. Term frequencies are weighted by field (name >
description > body) and normalised, so a long memory does not win on length alone.

It is still lexical, not semantic; swap in a vector store for embeddings recall.

Namespacing
-----------
`namespace` scopes a store to one tenant/user/session. Memories are stored under a
subdirectory per namespace and recall never crosses one, so a multi-tenant host gets
isolation from the store rather than from remembering to hand each tenant a different path.

The directory name is the namespace slug plus a digest of the EXACT string (see
`namespace_dir`). Slugging alone was lossy -- `TENANT-A`, `tenant_a` and `tenant a` all
collapsed onto one directory, so two tenants whose ids differed only in case or separator
silently shared a memory store.

Memory is model-authored data, not engine output. Frontmatter fields are flattened to a
single line on write so a value cannot close the block and forge the rest of the document,
and `RecallMemory` output is fenced as untrusted like any other tool result.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from engine.filelock import atomic_write_text, file_lock


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass(frozen=True)
class Memory:
    name: str
    description: str
    type: MemoryType
    body: str


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

# Frontmatter is a line-oriented `key: value` block, so a value containing a newline can
# close the block early and forge the rest of the document. Every field written into it is
# model-supplied, and the model's own input is untrusted tool output — so a memory saved
# from a poisoned web page could previously set its own `type`, replace the body, and come
# back on a LATER run as a stored fact. Values are flattened to one line on the way in;
# `body` is exempt because it lives after the closing `---` where it cannot escape.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_FIELD = 500


def scalar_field(value: str, *, limit: int = _MAX_FIELD) -> str:
    """Coerce a model-supplied value into something that cannot escape a frontmatter line.

    Newlines and carriage returns collapse to spaces, control characters are dropped, and
    the result is length-capped. This is a structural fix, not a denylist: after it there
    is no character left that terminates a frontmatter line or block.
    """
    flat = str(value or "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    flat = _CONTROL_CHARS.sub("", flat).strip()
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat

# Common words with no recall value — excluded from keyword matching (with <3-char terms)
# so they don't substring-match inside unrelated memory text.
_STOPWORDS = frozenset({
    "the", "and", "for", "how", "what", "why", "who", "you", "your", "our", "are", "was",
    "can", "did", "does", "with", "from", "this", "that", "there", "here", "has", "had",
    "have", "into", "out", "about", "when", "where", "which", "would", "should", "could",
})

# Relative weight of a hit per field: a term in the name is a stronger signal than the same
# term buried in the body.
_W_NAME, _W_DESC, _W_BODY = 6.0, 3.0, 1.0


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def namespace_dir(namespace: str) -> str:
    """The directory name for a tenant/user/session namespace.

    Slugging alone is LOSSY: it lowercases and folds every non-alphanumeric run to a
    hyphen, so `TENANT-A`, `tenant_a`, `tenant a` and `../tenant-a` all collapsed onto one
    directory — two tenants whose ids differed only in case or separator silently shared a
    memory store. The slug is kept for readability but disambiguated with a short digest of
    the EXACT namespace string, so distinct namespaces are always distinct directories
    while path traversal stays impossible (the result has no separators).
    """
    import hashlib

    raw = str(namespace)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(raw)[:48] or 'ns'}-{digest}"


def _terms(text: str) -> list[str]:
    return [
        t for t in re.split(r"\W+", (text or "").lower())
        if len(t) >= 3 and t not in _STOPWORDS
    ]


class MemoryStore:
    # How long a cached directory sweep is trusted before re-statting every file. Writes made
    # through this object invalidate immediately, so this only bounds how stale a change made
    # by ANOTHER process can be.
    _STAT_TTL = 2.0

    def __init__(
        self,
        root: str | Path,
        namespace: str | None = None,
        *,
        audit: Any = None,
    ) -> None:
        """`namespace` scopes every read and write to one tenant/user/session.

        `audit` is an optional sink with `note(text)` (an `AuditLog` satisfies it). Memory
        is durable state an agent carries between runs, so a write to it is a consequential
        action — without a record, a poisoned or clobbered fact is undetectable after the
        fact and unattributable to the run that made it.
        """
        self.namespace = namespace
        base = Path(root)
        self.root = base / namespace_dir(namespace) if namespace else base
        self.root.mkdir(parents=True, exist_ok=True)
        self._audit = audit
        self.index = self.root / "MEMORY.md"
        # slug -> (stat signature, Memory, term counts). Rebuilt lazily per changed file.
        self._cache: dict[str, tuple[tuple[float, int], Memory, dict[str, float]]] = {}
        self._dir_sig: float | None = None   # directory mtime at the last sweep
        self._last_sync: float = 0.0
        # Inverted index: term -> {slug: weight}. Query cost is then proportional to the
        # number of MATCHING memories, not to the size of the corpus.
        self._postings: dict[str, dict[str, float]] = {}

    # --- writing ------------------------------------------------------------
    def save(self, mem: Memory, *, overwrite: bool = True) -> Path:
        """Persist one memory. Returns its path.

        With `overwrite=False` a name that slugs onto an existing DIFFERENT memory gets a
        numeric suffix instead of destroying it. Distinct names collapsing onto one slug
        ("Deploy Process" and "deploy-process") used to overwrite silently, with no error and
        no way to notice the first memory was gone.
        """
        slug = _slug(scalar_field(mem.name)) or "memory"
        description = scalar_field(mem.description)
        # ONE lock spans slug resolution, the memory write and the index update. The
        # non-overwrite path picks a free `slug-N`, so checking existence outside the lock
        # let two concurrent saves choose the same suffix and one silently overwrite the
        # other - the exact collision `overwrite=False` exists to prevent.
        with file_lock(self.index):
            path = self.root / f"{slug}.md"
            overwrote = False
            if path.exists():
                if overwrite:
                    overwrote = True
                else:
                    n = 2
                    while (self.root / f"{slug}-{n}.md").exists():
                        n += 1
                    slug = f"{slug}-{n}"
                    path = self.root / f"{slug}.md"
            self._atomic_write(
                path,
                f"---\nname: {slug}\ndescription: {description}\ntype: {mem.type.value}\n"
                f"---\n\n{mem.body}\n",
            )
            self._write_index_line(slug, description)
        self._cache.pop(slug, None)
        self._dir_sig = None  # force the next sweep to pick this up
        self._record(
            "memory.save",
            slug=slug, type=mem.type.value, overwrote=overwrote,
            description=description, body_chars=len(mem.body or ""),
        )
        return path

    def delete(self, slug: str) -> bool:
        """Remove a memory and its index line. Returns False if it was not there.

        A store with no delete cannot have a memory retracted when it turns out to be wrong,
        which is the single most common reason to change one.
        """
        slug = _slug(slug)
        path = self.root / f"{slug}.md"
        existed = path.exists()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._cache.pop(slug, None)
        self._dir_sig = None
        if existed:
            # Same shared read-modify-write as _add_index_line, and the same lock: a delete
            # racing a save otherwise resurrects the deleted pointer or drops the saved one.
            with file_lock(self.index):
                if self.index.exists():
                    lines = [
                        l for l in self.index.read_text(encoding="utf-8").splitlines()
                        if f"({slug}.md)" not in l
                    ]
                    self._atomic_write(self.index, "\n".join(lines) + ("\n" if lines else ""))
        self._record("memory.delete", slug=slug, existed=existed)
        return existed

    def _record(self, action: str, **fields: Any) -> None:
        """Write one memory mutation to the audit sink. Never raises: a logging failure must
        not lose the write that already happened on disk."""
        if self._audit is None:
            return
        note = getattr(self._audit, "note", None)
        if not callable(note):
            return
        parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
        try:
            note(f"{action} ns={self.namespace!r} {parts}")
        except Exception:
            pass

    def _atomic_write(self, path: Path, text: str) -> None:
        """Write via a temp file + os.replace so a crash mid-write cannot leave a memory (or
        the index) half-written and unparseable.

        The replace is RETRIED. On Windows `os.replace` fails with a sharing violation if
        any other handle to the destination is open, so under two concurrent writers this
        raised `PermissionError: [WinError 5]` straight out of `save()` and killed the
        calling thread.
        """
        atomic_write_text(path, text)

    def _write_index_line(self, slug: str, description: str) -> None:
        """Point MEMORY.md at one memory. **The caller must hold `file_lock(self.index)`.**

        This is a read-modify-write of a file EVERY writer shares. Without a cross-process
        lock two concurrent saves both read the old index, both append their own line, and
        the second replace discards the first - a classic lost update, measured at one
        dropped pointer per ~80 concurrent saves. A memory whose index line is lost still
        exists on disk but is invisible to anything that reads MEMORY.md.
        """
        line = f"- [{slug}]({slug}.md) — {description}"
        existing = self.index.read_text(encoding="utf-8") if self.index.exists() else ""
        # replace an existing pointer for this slug rather than duplicating
        lines = [l for l in existing.splitlines() if f"({slug}.md)" not in l]
        lines.append(line)
        self._atomic_write(self.index, "\n".join(lines) + "\n")

    def _add_index_line(self, slug: str, description: str) -> None:
        """Locking wrapper around `_write_index_line`, for callers outside `save()`."""
        with file_lock(self.index):
            self._write_index_line(slug, description)

    # --- reading ------------------------------------------------------------
    def load(self, slug: str) -> Memory:
        text = (self.root / f"{slug}.md").read_text(encoding="utf-8")
        m = _FRONTMATTER.match(text)
        if not m:
            raise ValueError(f"{slug}: no frontmatter")
        fm: dict[str, str] = {}
        for line in m.group(1).splitlines():
            fm_match = re.match(r"\s*(\w+):\s*(.*)", line)
            if fm_match:
                # FIRST occurrence wins. Writes are sanitised so a duplicate key cannot be
                # injected any more, but a file edited by hand (or written by an older
                # build) can still carry one — and last-wins let the injected copy override
                # the real `type`/`description`. Defence in depth, on the read side too.
                fm.setdefault(fm_match.group(1), fm_match.group(2).strip())
        raw_type = fm.get("type", "project")
        try:
            mtype = MemoryType(raw_type)
        except ValueError:
            mtype = MemoryType.PROJECT  # unknown type: keep the memory, don't lose it
        return Memory(
            name=fm.get("name", slug),
            description=fm.get("description", ""),
            type=mtype,
            body=m.group(2).strip(),
        )

    def _safe_load(self, slug: str) -> Optional[Memory]:
        try:
            return self.load(slug)
        except (OSError, ValueError):
            return None

    def _scan(self) -> list[tuple[str, tuple[float, int]]]:
        """(slug, stat-signature) for every memory, in ONE directory pass.

        `glob()` + a separate `Path.stat()` per file cost two syscalls each; `os.scandir`
        carries the stat data with the entry (and on Windows it comes free from the
        directory read). On a 1,600-memory store this is the difference between a sweep
        that dominates every recall and one that does not.
        """
        out: list[tuple[str, tuple[float, int]]] = []
        try:
            with os.scandir(self.root) as it:
                for entry in it:
                    name = entry.name
                    if name == "MEMORY.md" or not name.endswith(".md"):
                        continue
                    try:
                        if not entry.is_file():
                            continue
                        st = entry.stat()
                    except OSError:
                        continue
                    out.append((name[:-3], (st.st_mtime, st.st_size)))
        except OSError:
            return []
        return out

    def _sync(self, force: bool = False) -> None:
        """Refresh the index for changed/new/removed files only.

        Two cheap gates avoid the sweep entirely: the directory's own mtime (which changes
        when a memory is created or deleted) and a short TTL. Writes through this object
        invalidate their own cache entry directly, so the sweep only exists to notice changes
        made by ANOTHER process — which does not need to be detected within microseconds.

        When it does run, it is one `scandir` pass plus a re-parse of only the files whose
        signature changed. It used to be a glob, a stat per file, and a full postings
        rebuild every time, which made recall cost scale with the whole corpus rather than
        with the number of matches.
        """
        now = time.monotonic()
        if not force and self._cache:
            try:
                dir_sig = self.root.stat().st_mtime
            except OSError:
                dir_sig = None
            if dir_sig == self._dir_sig and (now - self._last_sync) < self._STAT_TTL:
                return
        self._last_sync = now
        try:
            self._dir_sig = self.root.stat().st_mtime
        except OSError:
            self._dir_sig = None

        seen: set[str] = set()
        changed = False
        for slug, sig in self._scan():
            seen.add(slug)
            hit = self._cache.get(slug)
            if hit is not None and hit[0] == sig:
                continue
            changed = True
            mem = self._safe_load(slug)
            if mem is None:
                self._cache.pop(slug, None)
                continue
            self._cache[slug] = (sig, mem, self._weights(mem))
        for gone in set(self._cache) - seen:
            del self._cache[gone]
            changed = True
        # Rebuilding the inverted index is O(corpus); only pay it when something moved.
        if changed or not self._postings:
            self._rebuild_postings()

    def _rebuild_postings(self) -> None:
        postings: dict[str, dict[str, float]] = {}
        for slug, (_sig, _mem, weights) in self._cache.items():
            for term, w in weights.items():
                postings.setdefault(term, {})[slug] = w
        self._postings = postings

    @staticmethod
    def _weights(mem: Memory) -> dict[str, float]:
        """Per-term weight for one memory, normalised so a long body cannot outscore a
        precise name match purely by repeating a term."""
        scores: dict[str, float] = {}
        for field_text, weight in (
            (mem.name, _W_NAME), (mem.description, _W_DESC), (mem.body, _W_BODY),
        ):
            terms = _terms(field_text)
            if not terms:
                continue
            seen: set[str] = set()
            for t in terms:
                if t in seen:
                    continue  # count each term once per field
                seen.add(t)
                scores[t] = scores.get(t, 0.0) + weight
        return scores

    def all(self) -> list[Memory]:
        self._sync()
        return [entry[1] for entry in self._cache.values()]

    def find_relevant(self, query: str, limit: int = 5) -> list[Memory]:
        """Weighted keyword overlap over name, description and body.

        Deterministic and dependency-free. Swap in a vector/embeddings store for semantic
        recall; the seam is this method.
        """
        self._sync()
        terms = set(_terms(query))
        if not terms:
            return []
        totals: dict[str, float] = {}
        for t in terms:
            exact = self._postings.get(t)
            if exact:
                for slug, w in exact.items():
                    totals[slug] = totals.get(slug, 0.0) + w
                continue
            # No exact hit: fall back to a substring match over the VOCABULARY (a few
            # thousand strings) rather than over every term of every memory, which made the
            # query cost scale with the whole corpus.
            for vocab_term, slugs in self._postings.items():
                if t in vocab_term:
                    for slug, w in slugs.items():
                        totals[slug] = totals.get(slug, 0.0) + w * 0.25
        scored = [
            (score, slug, self._cache[slug][1])
            for slug, score in totals.items()
            if score > 0 and slug in self._cache
        ]
        # Sort by score desc, then slug asc so equal scores are stable and reproducible.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [m for _s, _slug, m in scored[:limit]]

    # --- async entry points -------------------------------------------------
    # Every method above does synchronous filesystem work. Called directly from an async
    # tool handler that blocks the event loop for the whole sweep (measured at 667 ms on a
    # 1,600-memory store), which stalls every other agent, request and MCP session sharing
    # the process. The tools use these wrappers instead, so the work runs on a worker
    # thread. The sync methods stay public for non-async callers and tests.
    async def asave(self, mem: Memory, *, overwrite: bool = True) -> Path:
        return await asyncio.to_thread(self.save, mem, overwrite=overwrite)

    async def adelete(self, slug: str) -> bool:
        return await asyncio.to_thread(self.delete, slug)

    async def afind_relevant(self, query: str, limit: int = 5) -> list[Memory]:
        return await asyncio.to_thread(self.find_relevant, query, limit)

    async def aall(self) -> list[Memory]:
        return await asyncio.to_thread(self.all)

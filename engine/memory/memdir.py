"""File-based memory: a small durable store of facts worth carrying across runs.

Four types: user, feedback, project, reference. One fact per file with frontmatter; a
MEMORY.md index holds one pointer line per memory. This is durable *context*, not a log of
transient outputs — do not fill it with re-derivable noise (code patterns, architecture,
git history, ephemeral state — anything derivable by reading the current project state).

Recall
------
Scoring covers name, description AND body. Scoring only name+description (the previous
behaviour) meant a fact stated in the body — which is where facts actually live — was
unreachable unless the description happened to repeat it.

Recall is served from an in-memory index built once and refreshed only for files whose
mtime/size changed, so a query costs O(matching terms) instead of re-reading and
re-parsing the entire corpus on every call. Term frequencies are weighted by field (name >
description > body) and normalised, so a long memory does not win on length alone.

It is still lexical, not semantic; swap in a vector store for embeddings recall.

Namespacing
-----------
`namespace` scopes a store to one tenant/user/session. Memories are stored under a
subdirectory per namespace and recall never crosses one, so a multi-tenant host gets
isolation from the store rather than from remembering to hand each tenant a different path.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


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

    def __init__(self, root: str | Path, namespace: str | None = None) -> None:
        """`namespace` scopes every read and write to one tenant/user/session."""
        self.namespace = namespace
        base = Path(root)
        self.root = base / _slug(namespace) if namespace else base
        self.root.mkdir(parents=True, exist_ok=True)
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
        slug = _slug(mem.name) or "memory"
        path = self.root / f"{slug}.md"
        if not overwrite and path.exists():
            n = 2
            while (self.root / f"{slug}-{n}.md").exists():
                n += 1
            slug = f"{slug}-{n}"
            path = self.root / f"{slug}.md"
        self._atomic_write(
            path,
            f"---\nname: {slug}\ndescription: {mem.description}\ntype: {mem.type.value}\n"
            f"---\n\n{mem.body}\n",
        )
        self._add_index_line(slug, mem.description)
        self._cache.pop(slug, None)
        self._dir_sig = None  # force the next sweep to pick this up
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
        if existed and self.index.exists():
            lines = [
                l for l in self.index.read_text(encoding="utf-8").splitlines()
                if f"({slug}.md)" not in l
            ]
            self._atomic_write(self.index, "\n".join(lines) + ("\n" if lines else ""))
        return existed

    def _atomic_write(self, path: Path, text: str) -> None:
        """Write via a temp file + os.replace so a crash mid-write cannot leave a memory (or
        the index) half-written and unparseable."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _add_index_line(self, slug: str, description: str) -> None:
        line = f"- [{slug}]({slug}.md) — {description}"
        existing = self.index.read_text(encoding="utf-8") if self.index.exists() else ""
        # replace an existing pointer for this slug rather than duplicating
        lines = [l for l in existing.splitlines() if f"({slug}.md)" not in l]
        lines.append(line)
        self._atomic_write(self.index, "\n".join(lines) + "\n")

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
                fm[fm_match.group(1)] = fm_match.group(2).strip()
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

    def _files(self) -> list[Path]:
        return [p for p in self.root.glob("*.md") if p.name != "MEMORY.md"]

    def _sync(self, force: bool = False) -> None:
        """Refresh the index for changed/new/removed files only.

        Rescanning means one `stat()` per memory, which on a 2,000-memory store is the whole
        cost of a query. Two cheap gates avoid it: the directory's own mtime (which changes
        when a memory is created or deleted) and a short TTL. Writes through this object
        invalidate their own cache entry directly, so the sweep only exists to notice changes
        made by ANOTHER process — which does not need to be detected within microseconds.
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
        for p in self._files():
            slug = p.stem
            seen.add(slug)
            try:
                st = p.stat()
            except OSError:
                continue
            sig = (st.st_mtime, st.st_size)
            hit = self._cache.get(slug)
            if hit is not None and hit[0] == sig:
                continue
            mem = self._safe_load(slug)
            if mem is None:
                self._cache.pop(slug, None)
                continue
            self._cache[slug] = (sig, mem, self._weights(mem))
        for gone in set(self._cache) - seen:
            del self._cache[gone]
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

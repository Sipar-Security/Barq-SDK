"""Filesystem tools, gated by FilesystemGuard.

The agent can read, write, edit and list files, but every access goes through the guard's
policy: writes are confined to the allowed workdir, and credential paths (~/.ssh, cloud
creds, .env) are unreadable even though the process could technically reach them.

Read/Write alone are not enough to work with a codebase: an agent that cannot list a
directory can only open paths it was already told about, and an agent that can only
overwrite whole files has to reproduce a file verbatim to change one line, expensive and
the most common way a model destroys work. ListDir, FindFiles and EditFile close that.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path

from engine.sandbox import FilesystemGuard

_MAX_READ = 20000

# Filesystem calls block. Inside `async def` they block the whole EVENT LOOP, not just the
# calling task — a FindFiles over 1,800 files was measured freezing it for 1.2 s, which
# stalls every other agent, request and MCP session in the process, and makes the
# coordinator's `max_parallel_tools` fan-out meaningless for the built-in tools. Every
# tool below does its I/O in a worker thread instead.
#
# The guard checks stay on the calling thread: they are pure path arithmetic, and running
# them inline keeps a denial cheap and keeps the decision adjacent to the code that reads
# it, rather than one thread-hop away from it.
async def _off_loop(fn, *args):
    return await asyncio.to_thread(fn, *args)

READ_FILE_SPEC = {
    "name": "ReadFile",
    "description": (
        "Read a UTF-8 text file. Denied for credential paths by policy. Long files are "
        "truncated; use `offset` to continue reading from a given character position."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "character offset to start at"},
            "limit": {"type": "integer", "description": f"max characters (default {_MAX_READ})"},
        },
        "required": ["path"],
    },
}

WRITE_FILE_SPEC = {
    "name": "WriteFile",
    "description": "Write a UTF-8 text file. Confined to the allowed workdir by policy.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}

EDIT_FILE_SPEC = {
    "name": "EditFile",
    "description": (
        "Replace an exact string in a file. `old` must appear exactly once unless "
        "`replace_all` is true. Prefer this over WriteFile for changing part of a file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "exact text to replace"},
            "new": {"type": "string", "description": "replacement text"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old", "new"],
    },
}

LIST_DIR_SPEC = {
    "name": "ListDir",
    "description": "List the entries of a directory. Directories are marked with a trailing /.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

FIND_FILES_SPEC = {
    "name": "FindFiles",
    "description": (
        "Find files under a directory matching a glob pattern (e.g. '*.py', 'src/**/*.ts'). "
        "Returns paths only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "root to search (default: workdir)"},
            "limit": {"type": "integer"},
        },
        "required": ["pattern"],
    },
}


def make_read_file(guard: FilesystemGuard):
    async def read_file(inp: dict) -> str:
        try:
            p = guard.assert_read(inp["path"])
        except Exception as e:  # FilesystemViolation
            return f"DENIED: {e}"
        try:
            text = await _off_loop(p.read_text, "utf-8", "replace")
        except OSError as e:
            return f"READ_ERROR: {e}"
        offset = max(0, int(inp.get("offset") or 0))
        limit = int(inp.get("limit") or _MAX_READ)
        limit = max(1, min(limit, _MAX_READ))
        chunk = text[offset:offset + limit]
        end = offset + len(chunk)
        if end < len(text):
            # Silent truncation let the model believe it had read a whole file it had not,
            # and then reason about content that was never in its context.
            chunk += (
                f"\n\n[TRUNCATED: showed characters {offset}-{end} of {len(text)}. "
                f"Call ReadFile again with offset={end} to continue.]"
            )
        elif offset:
            chunk += f"\n\n[end of file at character {len(text)}]"
        return chunk

    return read_file


def make_write_file(guard: FilesystemGuard):
    async def write_file(inp: dict) -> str:
        content = str(inp.get("content", ""))
        try:
            p = guard.assert_write(inp["path"])
        except Exception as e:
            return f"DENIED: {e}"
        def _write() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        try:
            await _off_loop(_write)
        except OSError as e:
            return f"WRITE_ERROR: {e}"
        return f"wrote {len(content)} bytes to {p}"

    return write_file


def make_edit_file(guard: FilesystemGuard):
    async def edit_file(inp: dict) -> str:
        old, new = str(inp.get("old", "")), str(inp.get("new", ""))
        if not old:
            return "REJECTED: 'old' must be a non-empty string"
        try:
            p = guard.assert_write(inp["path"])   # editing is writing
            guard.assert_read(inp["path"])        # ...and reading
        except Exception as e:
            return f"DENIED: {e}"
        try:
            text = await _off_loop(p.read_text, "utf-8")
        except OSError as e:
            return f"READ_ERROR: {e}"
        count = text.count(old)
        if count == 0:
            return "NOT_FOUND: the 'old' text does not appear in the file; read it and retry"
        if count > 1 and not inp.get("replace_all"):
            return (
                f"AMBIGUOUS: 'old' appears {count} times. Include more surrounding context to "
                "make it unique, or set replace_all=true."
            )
        updated = text.replace(old, new) if inp.get("replace_all") else text.replace(old, new, 1)
        try:
            await _off_loop(p.write_text, updated, "utf-8")
        except OSError as e:
            return f"WRITE_ERROR: {e}"
        return f"replaced {count if inp.get('replace_all') else 1} occurrence(s) in {p}"

    return edit_file


def make_list_dir(guard: FilesystemGuard):
    async def list_dir(inp: dict) -> str:
        try:
            p = guard.assert_read(inp["path"])
        except Exception as e:
            return f"DENIED: {e}"

        def _scan() -> str | None:
            if not p.is_dir():
                return None
            # scandir carries is_dir/stat with each entry, so one pass answers everything;
            # iterdir + a stat() per name was three syscalls per file.
            rows: list[tuple[bool, str, str]] = []
            with os.scandir(p) as it:
                for entry in it:
                    try:
                        if entry.is_dir():
                            rows.append((True, entry.name.lower(), f"{entry.name}/"))
                            continue
                        size = entry.stat().st_size
                        rows.append((False, entry.name.lower(), f"{entry.name}  ({size} bytes)"))
                    except OSError:
                        rows.append((False, entry.name.lower(), entry.name))
            rows.sort(key=lambda r: (not r[0], r[1]))
            return "\n".join(r[2] for r in rows)

        try:
            listing = await _off_loop(_scan)
        except OSError as e:
            return f"READ_ERROR: {e}"
        if listing is None:
            return f"NOT_A_DIRECTORY: {p}"
        return listing or f"{p} is empty"

    return list_dir


def make_find_files(guard: FilesystemGuard, default_root: Path | None = None):
    async def find_files(inp: dict) -> str:
        root_arg = inp.get("path") or (str(default_root) if default_root else ".")
        try:
            root = guard.assert_read(root_arg)
        except Exception as e:
            return f"DENIED: {e}"
        if not root.is_dir():
            return f"NOT_A_DIRECTORY: {root}"
        pattern = str(inp.get("pattern", "*"))
        limit = max(1, min(int(inp.get("limit") or 200), 1000))

        def _walk() -> list[str]:
            # os.walk over rglob: it yields names per directory without building a Path
            # object for every entry, and lets us skip a subtree the policy hides instead
            # of descending into it and filtering afterwards.
            found: list[str] = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d for d in dirnames if guard.can_read(os.path.join(dirpath, d))
                ]
                for name in filenames:
                    if len(found) >= limit:
                        return found
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                        if guard.can_read(full):  # never surface a path policy hides
                            found.append(rel)
            return found

        try:
            hits = await _off_loop(_walk)
        except OSError as e:
            return f"SEARCH_ERROR: {e}"
        if not hits:
            return f"no files matching {pattern!r} under {root}"
        out = "\n".join(sorted(hits))
        if len(hits) >= limit:
            out += f"\n\n[TRUNCATED at {limit} results; narrow the pattern]"
        return out

    return find_files

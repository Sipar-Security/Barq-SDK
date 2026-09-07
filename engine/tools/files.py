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

import fnmatch
from pathlib import Path

from engine.sandbox import FilesystemGuard

_MAX_READ = 20000

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
            text = p.read_text(encoding="utf-8", errors="replace")
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
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
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
            text = p.read_text(encoding="utf-8")
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
            p.write_text(updated, encoding="utf-8")
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
        if not p.is_dir():
            return f"NOT_A_DIRECTORY: {p}"
        try:
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as e:
            return f"READ_ERROR: {e}"
        if not entries:
            return f"{p} is empty"
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"{e.name}/")
            else:
                try:
                    lines.append(f"{e.name}  ({e.stat().st_size} bytes)")
                except OSError:
                    lines.append(e.name)
        return "\n".join(lines)

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
        hits: list[str] = []
        try:
            for p in root.rglob("*"):
                if len(hits) >= limit:
                    break
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                    if guard.can_read(p):  # never surface a path policy hides
                        hits.append(rel)
        except OSError as e:
            return f"SEARCH_ERROR: {e}"
        if not hits:
            return f"no files matching {pattern!r} under {root}"
        out = "\n".join(sorted(hits))
        if len(hits) >= limit:
            out += f"\n\n[TRUNCATED at {limit} results; narrow the pattern]"
        return out

    return find_files

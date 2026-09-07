"""Tests for the built-in file toolset.

Read and Write alone cannot work a codebase: an agent that cannot list a directory can only
open paths it was already told about, and one that can only overwrite whole files must
reproduce a file verbatim to change a line. These cover the discovery and edit tools that
close that gap, and the policy boundary each of them must respect.
"""

from __future__ import annotations

import asyncio

import pytest

from engine.sandbox import FilesystemGuard, FilesystemPolicy
from engine.tools.files import (
    make_edit_file,
    make_find_files,
    make_list_dir,
    make_read_file,
    make_write_file,
)


@pytest.fixture()
def tools(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    guard = FilesystemGuard(FilesystemPolicy.build(wd))
    return {
        "read": make_read_file(guard),
        "write": make_write_file(guard),
        "edit": make_edit_file(guard),
        "ls": make_list_dir(guard),
        "find": make_find_files(guard, wd),
        "wd": wd,
    }


def run(coro):
    return asyncio.run(coro)


# --- read ------------------------------------------------------------------------------
def test_read_returns_content_and_marks_truncation(tools):
    wd = tools["wd"]
    (wd / "small.txt").write_text("hello", encoding="utf-8")
    assert run(tools["read"]({"path": "small.txt"})) == "hello"

    (wd / "big.txt").write_text("z" * 25000, encoding="utf-8")
    out = run(tools["read"]({"path": "big.txt"}))
    assert "[TRUNCATED" in out and "offset=20000" in out

    rest = run(tools["read"]({"path": "big.txt", "offset": 20000}))
    assert "[TRUNCATED" not in rest and "end of file" in rest
    assert rest.count("z") == 5000


def test_read_respects_the_limit_argument(tools):
    (tools["wd"] / "f.txt").write_text("abcdefghij", encoding="utf-8")
    out = run(tools["read"]({"path": "f.txt", "limit": 4}))
    assert out.startswith("abcd") and "[TRUNCATED" in out


def test_read_denies_secret_paths(tools):
    (tools["wd"] / ".env").write_text("SECRET=1", encoding="utf-8")
    out = run(tools["read"]({"path": ".env"}))
    assert out.startswith("DENIED") and "SECRET=1" not in out


def test_read_reports_a_missing_file_without_raising(tools):
    assert run(tools["read"]({"path": "nope.txt"})).startswith("READ_ERROR")


# --- write -----------------------------------------------------------------------------
def test_write_creates_parent_directories(tools):
    out = run(tools["write"]({"path": "a/b/c.txt", "content": "hi"}))
    assert out.startswith("wrote")
    assert (tools["wd"] / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "hi"


def test_write_outside_the_workdir_is_denied(tools, tmp_path):
    out = run(tools["write"]({"path": str(tmp_path / "escape.txt"), "content": "x"}))
    assert out.startswith("DENIED")
    assert not (tmp_path / "escape.txt").exists()


# --- edit ------------------------------------------------------------------------------
def test_edit_replaces_a_unique_string(tools):
    f = tools["wd"] / "code.py"
    f.write_text("def a():\n    return 1\n", encoding="utf-8")
    out = run(tools["edit"]({"path": "code.py", "old": "return 1", "new": "return 2"}))
    assert "replaced 1" in out
    assert f.read_text(encoding="utf-8") == "def a():\n    return 2\n"


def test_edit_refuses_an_ambiguous_match(tools):
    f = tools["wd"] / "d.txt"
    f.write_text("x\nx\n", encoding="utf-8")
    out = run(tools["edit"]({"path": "d.txt", "old": "x", "new": "y"}))
    assert out.startswith("AMBIGUOUS") and "2 times" in out
    assert f.read_text(encoding="utf-8") == "x\nx\n", "an ambiguous edit must change nothing"


def test_edit_replace_all_is_explicit(tools):
    f = tools["wd"] / "d.txt"
    f.write_text("x\nx\n", encoding="utf-8")
    out = run(tools["edit"]({"path": "d.txt", "old": "x", "new": "y", "replace_all": True}))
    assert "replaced 2" in out
    assert f.read_text(encoding="utf-8") == "y\ny\n"


def test_edit_reports_a_missing_target(tools):
    (tools["wd"] / "d.txt").write_text("hello", encoding="utf-8")
    assert run(tools["edit"]({"path": "d.txt", "old": "zzz", "new": "y"})).startswith("NOT_FOUND")


def test_edit_rejects_an_empty_old_string(tools):
    (tools["wd"] / "d.txt").write_text("hello", encoding="utf-8")
    assert run(tools["edit"]({"path": "d.txt", "old": "", "new": "y"})).startswith("REJECTED")


def test_edit_outside_the_workdir_is_denied(tools, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    out = run(tools["edit"]({"path": str(outside), "old": "original", "new": "hacked"}))
    assert out.startswith("DENIED")
    assert outside.read_text(encoding="utf-8") == "original"


# --- list ------------------------------------------------------------------------------
def test_list_dir_marks_directories_and_sizes(tools):
    wd = tools["wd"]
    (wd / "sub").mkdir()
    (wd / "a.txt").write_text("12345", encoding="utf-8")
    out = run(tools["ls"]({"path": "."}))
    assert "sub/" in out
    assert "a.txt" in out and "5 bytes" in out


def test_list_dir_reports_empty_and_non_directories(tools):
    wd = tools["wd"]
    (wd / "empty").mkdir()
    assert "is empty" in run(tools["ls"]({"path": "empty"}))
    (wd / "f.txt").write_text("x", encoding="utf-8")
    assert run(tools["ls"]({"path": "f.txt"})).startswith("NOT_A_DIRECTORY")


# --- find ------------------------------------------------------------------------------
def test_find_files_matches_a_glob_recursively(tools):
    wd = tools["wd"]
    (wd / "src").mkdir()
    (wd / "src" / "a.py").write_text("", encoding="utf-8")
    (wd / "src" / "b.txt").write_text("", encoding="utf-8")
    (wd / "c.py").write_text("", encoding="utf-8")

    out = run(tools["find"]({"pattern": "*.py"}))
    assert "src/a.py" in out and "c.py" in out and "b.txt" not in out


def test_find_files_never_surfaces_a_hidden_secret(tools):
    wd = tools["wd"]
    (wd / ".env").write_text("SECRET=1", encoding="utf-8")
    (wd / "key.pem").write_text("-----BEGIN", encoding="utf-8")
    (wd / "ok.txt").write_text("", encoding="utf-8")

    out = run(tools["find"]({"pattern": "*"}))
    assert "ok.txt" in out
    assert ".env" not in out and "key.pem" not in out


def test_find_files_reports_no_matches(tools):
    assert "no files matching" in run(tools["find"]({"pattern": "*.rs"}))


def test_find_files_truncates_at_the_limit(tools):
    wd = tools["wd"]
    for i in range(30):
        (wd / f"f{i}.txt").write_text("", encoding="utf-8")
    out = run(tools["find"]({"pattern": "*.txt", "limit": 10}))
    assert "[TRUNCATED at 10 results" in out

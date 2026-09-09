"""Cross-process advisory file locking.

A `threading.Lock` only serialises writers inside one Python object. Two `Agent`s, a worker
pool, or two processes pointed at the same directory each hold their own, interleave, and
corrupt whatever they share. The audit log has always taken an OS lock for exactly that
reason; this module is that mechanism, extracted so every store that shares a file can use
the same one instead of each re-deriving it (or, as the memory store did, going without).

The lock is taken on a SIDECAR file, never on the file being written. A store that replaces
its file atomically (`os.replace`) must not be holding that file open while it does so — on
Windows the replace then fails with `PermissionError: [WinError 5]`.

    with file_lock(path):          # locks `path.lock`
        data = read(path)
        write_atomic(path, mutate(data))

Usage is advisory: it protects processes that cooperate by taking the same lock. Nothing
stops an unrelated process from writing the file.
"""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["file_lock", "atomic_write_text", "LOCK_SUFFIX"]

LOCK_SUFFIX = ".lock"

# How long to keep retrying a Windows `os.replace` that loses a race with a reader. Each
# attempt is cheap and the contended window is sub-millisecond; the cap exists so a genuinely
# stuck file surfaces as an error instead of hanging forever.
_REPLACE_ATTEMPTS = 50
_REPLACE_BACKOFF = 0.01

try:  # POSIX
    import fcntl

    def _lock_fd(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock_fd(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

except ImportError:  # Windows
    import msvcrt

    def _lock_fd(fh) -> None:
        # msvcrt.locking locks a byte range from the CURRENT position, so every writer must
        # lock the SAME offset (byte 0) or they exclude nothing. LK_LOCK gives up after
        # ~10s, so retry rather than fail.
        while True:
            try:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.01)

    def _unlock_fd(fh) -> None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextmanager
def file_lock(path: str | Path, *, suffix: str = LOCK_SUFFIX) -> Iterator[None]:
    """Hold an exclusive cross-process lock for `path` for the duration of the block.

    The lock lives on `path + suffix` so the target file itself stays closed and can be
    atomically replaced inside the block.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + suffix)
    fh = open(lock_path, "a+b")
    try:
        _lock_fd(fh)
        try:
            yield
        finally:
            _unlock_fd(fh)
    finally:
        fh.close()


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` via a temp file + `os.replace`, retrying a Windows sharing
    violation.

    `os.replace` is atomic on POSIX and on Windows, but on Windows it FAILS if any other
    handle to the destination is open — including a reader that opened it microseconds
    earlier. An unretried replace therefore raises `PermissionError: [WinError 5]` under
    concurrency, which is how the memory store's index write killed its calling thread
    outright rather than merely racing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        last: OSError | None = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, target)
                return
            except PermissionError as e:  # Windows sharing violation
                last = e
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EBUSY):
                    raise
                last = e
            time.sleep(_REPLACE_BACKOFF * (1 + attempt // 10))
        raise last if last is not None else OSError(f"could not replace {target}")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

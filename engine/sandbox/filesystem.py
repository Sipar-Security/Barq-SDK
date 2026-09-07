"""FilesystemGuard: path read/write policy.

Mirrors CC's sandbox filesystem config (entrypoints/sandboxTypes.ts: allowWrite,
denyWrite, denyRead, allowRead). Semantics match CC:
  - read denied if the path is under a denyRead region AND not under an allowRead region
    (allowRead takes precedence, re-allowing within a denied region);
  - write allowed only if under the workdir or an allowWrite path, AND not under denyWrite.

By default, credential paths (~/.ssh, cloud creds, gnupg, and this project's .env with
our MCP/model keys) are denied for reading even though the process could technically
reach them (exactly CC's denyRead-credentials posture). This is our own enforcement
layer for the agent's file tools; it is NOT an OS sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    for r in roots:
        try:
            if path == r or path.is_relative_to(r):
                return True
        except (ValueError, OSError):
            continue
    return False


def default_credential_denylist() -> tuple[Path, ...]:
    home = Path.home()
    names = [".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker",
             ".config/gcloud", ".config/gh"]
    return tuple(_resolve(home / n) for n in names)


# Secret files that must never be READ by the agent's file tools, matched by name/suffix
# WHEREVER they live, not only under a home credential dir. The region denylist above
# covers ~/.ssh etc.; this closes the ".env is readable anywhere" gap (the region list did
# not cover a project-local .env, and ReadFile was the only tool honouring the guard at all).
# Template envs are
# deliberately readable (they carry keys, not secrets).
SENSITIVE_READ_NAMES = frozenset({
    ".env", ".env.local", ".env.production", ".env.development", ".env.staging",
    ".env.prod", ".env.dev", ".env.test",
    ".git-credentials", "credentials", ".netrc", ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa",
})
_TEMPLATE_ENV_NAMES = frozenset({
    ".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults",
})
SENSITIVE_READ_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".ppk")
# Path segments that denote a credential directory, used to scan raw shell commands
# (readonly.py) where there is no resolved Path to region-match.
CREDENTIAL_DIR_SEGMENTS = frozenset({
    ".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker", "gcloud",
})


def name_is_sensitive_read(path: str | Path) -> bool:
    """True if a filename looks like a secret that must not be read (by name or suffix)."""
    name = Path(path).name.lower()
    if name in _TEMPLATE_ENV_NAMES:
        return False
    if name in SENSITIVE_READ_NAMES:
        return True
    return name.endswith(SENSITIVE_READ_SUFFIXES)


@dataclass(frozen=True)
class FilesystemPolicy:
    workdir: Path                       # workspace dir: writable + readable by default
    allow_write: tuple[Path, ...] = ()
    deny_write: tuple[Path, ...] = ()
    deny_read: tuple[Path, ...] = ()
    allow_read: tuple[Path, ...] = ()
    block_sensitive_names: bool = True  # deny reads of .env/*.pem/id_rsa/... anywhere

    @classmethod
    def build(
        cls,
        workdir: str | Path,
        allow_write: tuple[str, ...] = (),
        deny_write: tuple[str, ...] = (),
        deny_read: tuple[str, ...] = (),
        allow_read: tuple[str, ...] = (),
        include_credential_denylist: bool = True,
        block_sensitive_names: bool = True,
    ) -> "FilesystemPolicy":
        dr = tuple(_resolve(p) for p in deny_read)
        if include_credential_denylist:
            dr = dr + default_credential_denylist()
        return cls(
            workdir=_resolve(workdir),
            allow_write=tuple(_resolve(p) for p in allow_write),
            deny_write=tuple(_resolve(p) for p in deny_write),
            deny_read=dr,
            allow_read=tuple(_resolve(p) for p in allow_read),
            block_sensitive_names=block_sensitive_names,
        )


class FilesystemViolation(PermissionError):
    pass


class FilesystemGuard:
    def __init__(self, policy: FilesystemPolicy) -> None:
        self.policy = policy

    def _abs(self, path: str | Path) -> Path:
        """Resolve a path to canonical absolute form. A RELATIVE path is taken relative to
        the policy's workdir (not the process CWD), so a model that writes ``notes.txt``
        lands inside the agent's workspace. Absolute paths and ``~`` are honored as given."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.policy.workdir / p
        return p.resolve()

    def can_read(self, path: str | Path) -> bool:
        p = self._abs(path)
        # An explicit allow_read region always wins (re-allows within a denied region).
        if _under(p, self.policy.allow_read):
            return True
        # Secret files (by name/suffix) are denied wherever they live; this is the layer
        # ReadFile/Grep/ListDir all consult, so the guarantee is consistent across tools.
        if self.policy.block_sensitive_names and name_is_sensitive_read(p):
            return False
        if _under(p, self.policy.deny_read):
            return False
        return True

    def can_write(self, path: str | Path) -> bool:
        p = self._abs(path)
        writable_roots = (self.policy.workdir,) + self.policy.allow_write
        if not _under(p, writable_roots):
            return False
        if _under(p, self.policy.deny_write):
            return False
        return True

    # convenience: raise instead of returning False (for use inside file tools)
    def assert_read(self, path: str | Path) -> Path:
        if not self.can_read(path):
            raise FilesystemViolation(f"read denied by sandbox policy: {path}")
        return self._abs(path)

    def assert_write(self, path: str | Path) -> Path:
        if not self.can_write(path):
            raise FilesystemViolation(f"write denied by sandbox policy: {path}")
        return self._abs(path)

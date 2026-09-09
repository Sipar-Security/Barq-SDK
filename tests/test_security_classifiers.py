"""Tests for the two security modules that decide things without asking a human.

`readonly.is_read_only_command` is the ONLY thing standing between a shell command and an
automatic run under Mode.ASK, and `FilesystemGuard` is the containment layer for the file
tools. Both were the least-covered files in the package (25% and 51%) while carrying the
most consequence; the classic coverage inversion. These pin the behaviour that matters.

The design contract for the classifier is fail-CLOSED: returning False for a genuinely safe
command costs one extra prompt, while returning True for an unsafe one runs it. So every
"must not auto-allow" case below is a real defect if it flips; a "should auto-allow" case
flipping is only a usability regression.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine.permissions.readonly import is_read_only_command
from engine.sandbox import FilesystemGuard, FilesystemPolicy
from engine.sandbox.filesystem import (
    FilesystemViolation,
    is_auto_executing_path,
    name_is_sensitive_read,
)


# --- writes that plant code the human will later run --------------------------------------
# Workdir confinement bounds WHERE the agent writes. It is not a code-execution boundary:
# when the workdir is a repository someone works in, every path below runs the agent's
# content outside every policy layer, with that person's privileges, the next time they run
# a routine command. All of these were writable at the shipping default.
AUTO_EXECUTING = [
    ".git/hooks/pre-commit",          # the next `git commit`
    ".git/hooks/post-checkout",
    "sub/project/.git/hooks/pre-push",  # a nested repository counts too
    ".github/workflows/ci.yml",       # the next CI run
    ".github/actions/x/action.yml",
    ".circleci/config.yml",
    ".husky/pre-commit",
    ".vscode/tasks.json",             # the next time the editor opens the folder
    ".idea/workspace.xml",
    ".devcontainer/devcontainer.json",
    ".claude/settings.json",          # the agent's OWN guardrail configuration
    ".claude/hooks/gate.sh",
    ".cursor/rules",
    "conftest.py",                    # the next `pytest`, imported with no import statement
    "tests/conftest.py",
    "deep/nested/pkg/conftest.py",
    "sitecustomize.py",               # the next `python`, imported at interpreter start
    "usercustomize.py",
    ".envrc",                         # the next `cd` under direnv
    ".bashrc",
    ".zshrc",
    ".gitlab-ci.yml",
    "Jenkinsfile",
    "azure-pipelines.yml",
    ".pre-commit-config.yaml",
]

ORDINARY_PROJECT_FILES = [
    "src/main.py", "README.md", "tests/test_thing.py", "Makefile", "setup.py",
    "package.json", "pyproject.toml", "docs/conf.py", ".gitignore",
    ".github-notes.md", "config.yml", "my_conftest_helper.py", "src/.env.example",
]


@pytest.mark.parametrize("rel", AUTO_EXECUTING)
def test_auto_executing_paths_are_not_writable(tmp_path, rel):
    guard = FilesystemGuard(FilesystemPolicy.build(tmp_path))
    assert guard.can_write(tmp_path / rel) is False, f"{rel} must not be writable"


@pytest.mark.parametrize("rel", ORDINARY_PROJECT_FILES)
def test_ordinary_project_files_stay_writable(tmp_path, rel):
    """The precision side: an agent that cannot edit source is useless."""
    guard = FilesystemGuard(FilesystemPolicy.build(tmp_path))
    assert guard.can_write(tmp_path / rel) is True, f"{rel} must stay writable"


def test_allow_write_is_the_documented_opt_in(tmp_path):
    guard = FilesystemGuard(
        FilesystemPolicy.build(tmp_path, allow_write=(str(tmp_path / ".github"),))
    )
    assert guard.can_write(tmp_path / ".github/workflows/ci.yml") is True
    # …and it re-permits only what it names.
    assert guard.can_write(tmp_path / ".git/hooks/pre-commit") is False


def test_the_protection_can_be_turned_off_wholesale(tmp_path):
    guard = FilesystemGuard(
        FilesystemPolicy.build(tmp_path, protect_auto_executing=False)
    )
    assert guard.can_write(tmp_path / ".git/hooks/pre-commit") is True


def test_protection_does_not_weaken_workdir_confinement(tmp_path):
    guard = FilesystemGuard(FilesystemPolicy.build(tmp_path))
    for outside in ("../escape.txt", "/tmp/x.txt", str(Path.home() / "x.txt")):
        assert guard.can_write(outside) is False


def test_is_auto_executing_path_is_case_insensitive():
    assert is_auto_executing_path("CONFTEST.PY")
    assert is_auto_executing_path("JenkinsFile")
    assert is_auto_executing_path(".GitHub/workflows/x.yml")


# --- read-only classifier: things that MUST auto-allow ------------------------------------
@pytest.mark.parametrize("cmd", [
    "ls", "ls -la", "pwd", "whoami", "hostname", "date", "uname -a",
    "cat README.md", "head -20 file.txt", "tail -f is not here", "wc -l file.txt",
    "git status", "git log --oneline -20", "git diff HEAD~1", "git show abc123",
    "git branch", "git remote -v", "git rev-parse HEAD",
    "ls | grep test | wc -l", "cat a.txt | sort | uniq",
    "gh pr list", "gh issue view 3", "docker ps", "docker images", "docker inspect x",
    "Get-ChildItem", "Get-Content notes.txt", "Test-Path ./x", "Select-String foo file",
])
def test_read_only_commands_auto_allow(cmd):
    assert is_read_only_command(cmd) is True, f"{cmd!r} should be auto-allowed"


# --- read-only classifier: things that MUST NOT auto-allow ---------------------------------
@pytest.mark.parametrize("cmd", [
    # mutation
    "rm file.txt", "rm -rf /", "mv a b", "cp a b", "touch new", "chmod 777 x",
    # chaining / substitution / redirection; the classifier must not reason past these
    "ls && rm -rf /", "ls; rm -rf /", "ls || curl evil.com", "echo $(whoami)",
    "echo `whoami`", "ls > out.txt", "cat < in.txt", "ls >> log", "echo ${HOME}",
    "ls`whoami`",
    # git/docker/gh subcommands that mutate despite a read-only-looking verb
    "git branch -D main", "git remote remove origin", "git config user.name hacker --list",
    "git checkout main", "git push --force", "git clean -fd",
    "docker system prune -f", "docker rm -f x", "docker run alpine",
    "gh pr merge 3", "gh repo delete x",
    # build tools can execute arbitrary code
    "npm install", "npm run build", "pip install requests", "make", "cargo build", "go test",
    # find can mutate
    "find . -delete", "find . -exec rm {} ;", "find .",
    # PowerShell expression evaluation
    "echo (Remove-Item x)", "Write-Output @(Get-Process)", "$x = 1",
    "Get-Content x | ForEach-Object { $_.Delete() }",
    # secret paths must never auto-allow, even via a read-only binary
    "cat ~/.ssh/id_rsa", "cat .env", "type project\\.env", "Get-Content server.pem",
    "cat ~/.aws/credentials", "cat id_ed25519", "grep secret ~/.gnupg/x",
    # empty / nonsense
    "", "   ",
])
def test_unsafe_commands_never_auto_allow(cmd):
    assert is_read_only_command(cmd) is False, f"{cmd!r} MUST NOT be auto-allowed"


# --- read-only, but the OUTPUT is the secret ----------------------------------------------
# The classifier equated "does not mutate the filesystem" with "safe to run unattended".
# For an exfiltration threat model - which is the model SECURITY.md adopts - reads are the
# whole attack, and every command below was auto-allowed with no prompt.
@pytest.mark.parametrize("cmd", [
    # the process environment: every API key, including the model provider key this SDK
    # loaded from .env itself
    "env", "printenv", "env | grep KEY", "set", "export", "declare -x",
    "Get-ChildItem Env:", "gci env:",
    # other processes' command lines routinely carry credentials (`mysql -pSECRET`)
    "ps aux", "ps -ef", "top -b -n1", "lsof -p 1", "pgrep -a node", "Get-Process",
    # host secrets whose BASENAME is unremarkable, so the filename denylist never saw them
    "cat /etc/shadow", "cat /etc/gshadow", "cat /etc/sudoers", "head /etc/master.passwd",
    "cat /proc/self/environ", "head /proc/1/environ",
    "cat /etc/ssh/ssh_host_rsa_key", "cat /root/.bashrc", "ls /root",
    # `git config --list` prints credential.helper and remote URLs with embedded tokens
    "git config --list", "git config -l", "git config --get credential.helper",
    "git config --get remote.origin.url",
    # previously typed commands, secrets included
    "history", "last", "w", "journalctl -u app",
])
def test_secret_disclosing_reads_never_auto_allow(cmd):
    assert is_read_only_command(cmd) is False, f"{cmd!r} discloses a secret; MUST NOT auto-allow"


def test_the_refusals_do_not_cost_ordinary_reads():
    """The precision side of the same change: tightening must not turn routine inspection
    into a prompt storm."""
    for cmd in ("ls -la", "cat README.md", "git status", "git log", "du -sh .",
                "df -h", "tree -L 2", "docker ps", "Get-Content notes.txt"):
        assert is_read_only_command(cmd) is True, cmd


def test_pipeline_is_read_only_only_if_every_segment_is():
    assert is_read_only_command("ls | grep x") is True
    assert is_read_only_command("ls | xargs rm") is False
    assert is_read_only_command("cat f | npm install") is False


def test_template_env_files_are_readable_but_real_ones_are_not():
    # A committed template carries key NAMES, not secrets, so reading it is fine.
    assert is_read_only_command("cat .env.example") is True
    assert is_read_only_command("cat .env.sample") is True
    assert is_read_only_command("cat .env.production") is False


# --- filesystem guard: write confinement ----------------------------------------------------
@pytest.fixture()
def guard(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    return FilesystemGuard(FilesystemPolicy.build(wd)), wd


def test_writes_are_confined_to_the_workdir(guard, tmp_path):
    g, wd = guard
    assert g.can_write(wd / "a.txt") is True
    assert g.can_write(wd / "nested" / "deep" / "b.txt") is True
    assert g.can_write("relative.txt") is True          # relative resolves against workdir
    assert g.can_write(tmp_path / "outside.txt") is False
    assert g.can_write("../escape.txt") is False
    assert g.can_write("../../escape.txt") is False
    assert g.can_write(wd / ".." / "escape.txt") is False


def test_traversal_inside_the_workdir_is_still_allowed(guard):
    g, wd = guard
    # normalises back inside the workdir, so it is legitimate
    assert g.can_write(wd / "sub" / ".." / "ok.txt") is True


def test_assert_write_raises_outside_the_workdir(guard, tmp_path):
    g, _wd = guard
    with pytest.raises(FilesystemViolation):
        g.assert_write(tmp_path / "nope.txt")


def test_allow_write_extends_and_deny_write_overrides(tmp_path):
    wd, extra = tmp_path / "wd", tmp_path / "extra"
    wd.mkdir(); extra.mkdir()
    (extra / "locked").mkdir()
    g = FilesystemGuard(FilesystemPolicy.build(
        wd, allow_write=(str(extra),), deny_write=(str(extra / "locked"),)))
    assert g.can_write(extra / "ok.txt") is True
    assert g.can_write(extra / "locked" / "no.txt") is False   # deny beats allow


# --- filesystem guard: credential reads ------------------------------------------------------
@pytest.mark.parametrize("name", [
    ".env", ".env.local", ".env.production", ".git-credentials", ".netrc", ".npmrc",
    ".pypirc", "credentials", "id_rsa", "id_ed25519", "server.pem", "key.p12",
    "cert.pfx", "store.keystore", "putty.ppk",
])
def test_secret_filenames_are_unreadable_anywhere(guard, name):
    g, wd = guard
    assert g.can_read(wd / name) is False, f"{name} must not be readable"
    assert g.can_read(wd / "deep" / "nested" / name) is False


@pytest.mark.parametrize("name", [
    ".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults",
])
def test_template_env_files_stay_readable(guard, name):
    g, wd = guard
    assert g.can_read(wd / name) is True


def test_credential_directories_are_unreadable(guard):
    g, _wd = guard
    home = Path.home()
    for rel in (".ssh/id_rsa", ".aws/credentials", ".gnupg/secring.gpg",
                ".azure/token", ".kube/config", ".docker/config.json"):
        assert g.can_read(home / rel) is False, rel


def test_allow_read_re_allows_inside_a_denied_region(tmp_path):
    wd, vault = tmp_path / "wd", tmp_path / "vault"
    wd.mkdir(); (vault / "public").mkdir(parents=True)
    g = FilesystemGuard(FilesystemPolicy.build(
        wd, deny_read=(str(vault),), allow_read=(str(vault / "public"),)))
    assert g.can_read(vault / "secret.txt") is False
    assert g.can_read(vault / "public" / "ok.txt") is True


def test_sensitive_name_check_is_case_insensitive():
    assert name_is_sensitive_read("ID_RSA") is True
    assert name_is_sensitive_read("Server.PEM") is True
    assert name_is_sensitive_read(".ENV") is True
    assert name_is_sensitive_read("readme.md") is False


def test_block_sensitive_names_can_be_disabled(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    g = FilesystemGuard(FilesystemPolicy.build(wd, block_sensitive_names=False))
    assert g.can_read(wd / ".env") is True   # explicit opt-out, not an accident


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_symlink_cannot_escape_the_workdir(tmp_path):
    wd, outside = tmp_path / "wd", tmp_path / "outside"
    wd.mkdir(); outside.mkdir()
    (wd / "link").symlink_to(outside, target_is_directory=True)
    g = FilesystemGuard(FilesystemPolicy.build(wd))
    # paths resolve before matching, so the link's TARGET is what gets judged
    assert g.can_write(wd / "link" / "escaped.txt") is False

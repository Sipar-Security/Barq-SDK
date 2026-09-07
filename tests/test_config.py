from engine.providers import ModelRole
from engine.providers.config import build_router_from_env, load_dotenv


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "real-key")
    envfile = tmp_path / ".env"
    envfile.write_text("DEEPSEEK_API_KEY=file-key\nFOO_BAR=baz\n", encoding="utf-8")
    load_dotenv(envfile)
    import os

    assert os.environ["DEEPSEEK_API_KEY"] == "real-key"  # real env wins
    assert os.environ["FOO_BAR"] == "baz"  # unset key filled from file


def test_load_dotenv_strips_inline_comments_and_quotes(tmp_path, monkeypatch):
    # Section 4E: trailing comments / quotes must not corrupt the value.
    import os

    for k in ("K_PLAIN", "K_COMMENT", "K_QUOTED", "K_QUOTED_COMMENT"):
        monkeypatch.delenv(k, raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text(
        "K_PLAIN=sk-plain\n"
        "K_COMMENT=sk-xxxx # my secret key\n"
        'K_QUOTED="sk-quoted"\n'
        'K_QUOTED_COMMENT="sk-both" # trailing note\n',
        encoding="utf-8",
    )
    load_dotenv(envfile)
    assert os.environ["K_PLAIN"] == "sk-plain"
    assert os.environ["K_COMMENT"] == "sk-xxxx"
    assert os.environ["K_QUOTED"] == "sk-quoted"
    assert os.environ["K_QUOTED_COMMENT"] == "sk-both"


def test_router_from_env_deepseek_only(tmp_path, monkeypatch):
    for k in ("MOONSHOT_API_KEY", "ZHIPU_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    router = build_router_from_env(dotenv=None)
    # every role resolves to deepseek when it's the only provider
    for role in ModelRole:
        assert router.spec(role).provider == "deepseek"
        assert router.spec(role).model == "deepseek-chat"


def test_router_splits_roles_when_multiple_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("ZHIPU_API_KEY", "k2")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    router = build_router_from_env(dotenv=None)
    # triage -> cheap deepseek, vuln-logic -> strong zhipu
    assert router.spec(ModelRole.FAST).provider == "deepseek"
    assert router.spec(ModelRole.SMART).provider == "zhipu"


def test_router_raises_with_no_provider(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "ZHIPU_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    import pytest

    with pytest.raises(RuntimeError):
        build_router_from_env(dotenv=None)

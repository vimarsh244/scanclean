import importlib
import subprocess

from scanclean.cli import build_parser


def test_cli_defaults_match_core_defaults():
    options = build_parser().parse_args(["input.pdf"])

    assert options.inputs == ["input.pdf"]
    assert options.outdir == "cleaned"
    assert options.deskew is True
    assert options.detect is True
    assert options.paper_floor is True
    assert options.bilevel is False


def test_importing_cli_does_not_run_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    import scanclean.cli

    importlib.reload(scanclean.cli)
    assert calls == []

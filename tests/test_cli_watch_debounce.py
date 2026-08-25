"""`graphify watch --debounce N` must reach the watcher.

`watch()` has taken a `debounce` argument since it was written, and
`python -m graphify.watch --help` advertises `--debounce` — but the shipped
`graphify watch` sub-command parsed only `--semantic` / `--backend` /
`--fallback-backend` and exited 2 with "unknown watch option" for the flag its
own help documents. Anyone with a busy tree was stuck on the 3-second default
and had no way to say otherwise.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import graphify.cli as cli
import graphify.watch as watch_mod

PYTHON = sys.executable


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """Run the CLI's watch branch with the real watcher replaced by a recorder."""
    calls: list[dict] = []

    def _fake_watch(path, **kwargs):
        calls.append({"path": path, **kwargs})

    monkeypatch.setattr(watch_mod, "watch", _fake_watch)

    def _run(*flags: str):
        monkeypatch.setattr(sys, "argv", ["graphify", "watch", str(tmp_path), *flags])
        cli.dispatch_command("watch")
        return calls[-1]

    return _run


def test_debounce_space_separated(recorded):
    assert recorded("--debounce", "60")["debounce"] == 60.0


def test_debounce_equals_form(recorded):
    assert recorded("--debounce=12.5")["debounce"] == 12.5


def test_debounce_zero_is_allowed(recorded):
    assert recorded("--debounce", "0")["debounce"] == 0.0


def test_omitting_debounce_leaves_the_default(recorded):
    """Not passing the flag must not pin the value — the watcher's own default
    stands, so it can change in one place."""
    assert "debounce" not in recorded()


def test_debounce_travels_with_the_other_flags(recorded):
    call = recorded("--semantic", "--debounce", "30", "--backend", "openai-cli")
    assert call["debounce"] == 30.0
    assert call["semantic"] is True
    assert call["backend"] == "openai-cli"


def test_debounce_rejects_a_non_number(tmp_path):
    proc = subprocess.run(
        [PYTHON, "-m", "graphify", "watch", str(tmp_path), "--debounce", "soon"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "--debounce expects a number" in proc.stderr


def test_debounce_rejects_a_negative_number(tmp_path):
    proc = subprocess.run(
        [PYTHON, "-m", "graphify", "watch", str(tmp_path), "--debounce", "-5"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "negative" in proc.stderr


def test_unknown_watch_option_is_still_rejected(tmp_path):
    proc = subprocess.run(
        [PYTHON, "-m", "graphify", "watch", str(tmp_path), "--nonsense"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "unknown watch option" in proc.stderr

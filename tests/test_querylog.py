"""Tests for graphify.querylog."""
import json
import os
import pytest
from pathlib import Path

from graphify.querylog import log_query, nodes_from_result


# ---------------------------------------------------------------------------
# nodes_from_result
# ---------------------------------------------------------------------------

def test_nodes_from_result_parses_header():
    result = "Traversal: BFS depth=2 | Start: ['foo'] | 7 nodes found\n\nNODE foo"
    assert nodes_from_result(result) == 7


def test_nodes_from_result_singular():
    assert nodes_from_result("1 node found") == 1


def test_nodes_from_result_missing():
    assert nodes_from_result("no match here") is None


def test_nodes_from_result_empty():
    assert nodes_from_result("") is None


# ---------------------------------------------------------------------------
# log_query — basic write
# ---------------------------------------------------------------------------

def test_log_query_writes_jsonl(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="query", question="what is X", corpus="/some/graph.json",
              result="3 nodes found\nNODE a", duration_ms=12.5, mode="bfs", depth=2)

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "query"
    assert rec["question"] == "what is X"
    assert rec["corpus"] == "/some/graph.json"
    assert rec["nodes_returned"] == 3
    assert rec["result_chars"] > 0
    assert rec["duration_ms"] == pytest.approx(12.5, abs=0.01)
    assert rec["mode"] == "bfs"
    assert "ts" in rec


def test_log_query_appends(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="query", question="q1", corpus="/g.json")
    log_query(kind="query", question="q2", corpus="/g.json")

    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "q1"
    assert json.loads(lines[1])["question"] == "q2"


# ---------------------------------------------------------------------------
# opt-out / opt-in
# ---------------------------------------------------------------------------

def test_disable_env(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_DISABLE", "1")

    log_query(kind="query", question="q", corpus="/g.json")

    assert not log_file.exists()


def test_disable_env_true(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_DISABLE", "true")

    log_query(kind="query", question="q", corpus="/g.json")

    assert not log_file.exists()


def test_responses_not_logged_by_default(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_RESPONSES", raising=False)

    log_query(kind="query", question="q", corpus="/g.json", result="NODE foo")

    rec = json.loads(log_file.read_text())
    assert "response" not in rec


def test_responses_optin(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_RESPONSES", "1")
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="query", question="q", corpus="/g.json", result="NODE foo bar")

    rec = json.loads(log_file.read_text())
    assert rec["response"] == "NODE foo bar"


# ---------------------------------------------------------------------------
# robustness — never raises
# ---------------------------------------------------------------------------

def test_log_never_raises(tmp_path, monkeypatch):
    # Point at a directory — open() for append will fail
    bad_path = tmp_path / "is_a_dir"
    bad_path.mkdir()
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(bad_path))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    # Must not raise
    log_query(kind="query", question="q", corpus="/g.json")


def test_log_creates_parent_dirs(tmp_path, monkeypatch):
    log_file = tmp_path / "deep" / "nested" / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="query", question="q", corpus="/g.json")

    assert log_file.exists()


# ---------------------------------------------------------------------------
# field coverage
# ---------------------------------------------------------------------------

def test_nodes_returned_inferred_from_result(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="query", question="q", corpus="/g.json",
              result="5 nodes found\nNODE a\nNODE b")

    rec = json.loads(log_file.read_text())
    assert rec["nodes_returned"] == 5


def test_explicit_nodes_returned_takes_precedence(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="path", question="A -> B", corpus="/g.json", nodes_returned=3)

    rec = json.loads(log_file.read_text())
    assert rec["nodes_returned"] == 3


def test_kind_mcp_query(tmp_path, monkeypatch):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)

    log_query(kind="mcp_query", question="q", corpus="/g.json")

    rec = json.loads(log_file.read_text())
    assert rec["kind"] == "mcp_query"


# ---------------------------------------------------------------------------
# #1797 — query log is opt-in (default OFF)
# ---------------------------------------------------------------------------

def _clear_log_env(monkeypatch):
    for k in ("GRAPHIFY_QUERY_LOG", "GRAPHIFY_QUERY_LOG_ENABLE", "GRAPHIFY_QUERY_LOG_DISABLE"):
        monkeypatch.delenv(k, raising=False)


def test_query_log_off_by_default(monkeypatch):
    from graphify.querylog import _log_path
    _clear_log_env(monkeypatch)
    assert _log_path() is None


def test_query_log_enabled_by_explicit_flag(monkeypatch):
    from graphify.querylog import _log_path
    _clear_log_env(monkeypatch)
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_ENABLE", "1")
    assert str(_log_path()).endswith("graphify-queries.log")


def test_query_log_enabled_by_explicit_path(monkeypatch, tmp_path):
    from graphify.querylog import _log_path
    _clear_log_env(monkeypatch)
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(tmp_path / "q.log"))
    assert _log_path() == tmp_path / "q.log"


def test_query_log_disable_wins(monkeypatch):
    from graphify.querylog import _log_path
    _clear_log_env(monkeypatch)
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_ENABLE", "1")
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG_DISABLE", "1")
    assert _log_path() is None


def test_log_query_writes_nothing_by_default(monkeypatch, tmp_path):
    """End-to-end: with no opt-in, log_query must not create the default log."""
    _clear_log_env(monkeypatch)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    log_query(kind="query", question="secret internal ticket TICKET-123", corpus=".", result="1 node found")
    assert not (tmp_path / ".cache" / "graphify-queries.log").exists()


# ---------------------------------------------------------------------------
# opt-in rotation — GRAPHIFY_QUERY_LOG_MAX_RECORDS
# ---------------------------------------------------------------------------

def _rot_env(monkeypatch, tmp_path, max_records=None):
    log_file = tmp_path / "q.log"
    monkeypatch.setenv("GRAPHIFY_QUERY_LOG", str(log_file))
    monkeypatch.delenv("GRAPHIFY_QUERY_LOG_DISABLE", raising=False)
    if max_records is None:
        monkeypatch.delenv("GRAPHIFY_QUERY_LOG_MAX_RECORDS", raising=False)
    else:
        monkeypatch.setenv("GRAPHIFY_QUERY_LOG_MAX_RECORDS", str(max_records))
    return log_file, log_file.with_name("q.archive.jsonl")


def test_no_rotation_by_default(tmp_path, monkeypatch):
    log_file, archive = _rot_env(monkeypatch, tmp_path)
    for i in range(10):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    assert len(log_file.read_text().splitlines()) == 10
    assert not archive.exists()


def test_rotation_archives_oldest_keeps_newest(tmp_path, monkeypatch):
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=5)
    for i in range(8):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    live = log_file.read_text().splitlines()
    arch = archive.read_text().splitlines()
    assert len(live) == 5
    assert [json.loads(l)["question"] for l in live] == ["q3", "q4", "q5", "q6", "q7"]
    assert [json.loads(l)["question"] for l in arch] == ["q0", "q1", "q2"]


def test_rotation_never_loses_a_record(tmp_path, monkeypatch):
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=3)
    n = 20
    for i in range(n):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    live = log_file.read_text().splitlines()
    arch = archive.read_text().splitlines()
    assert len(live) + len(arch) == n
    seen = [json.loads(l)["question"] for l in arch] + [json.loads(l)["question"] for l in live]
    assert seen == [f"q{i}" for i in range(n)]


def test_rotation_archive_appends_not_clobbers(tmp_path, monkeypatch):
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=2)
    archive.write_text('{"question": "pre-existing"}\n')
    for i in range(4):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    arch = archive.read_text().splitlines()
    assert json.loads(arch[0])["question"] == "pre-existing"
    assert len(arch) == 3  # pre-existing + q0 + q1


@pytest.mark.parametrize("bad", ["0", "-3", "abc", " "])
def test_rotation_invalid_bound_means_off(tmp_path, monkeypatch, bad):
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=bad)
    for i in range(6):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    assert len(log_file.read_text().splitlines()) == 6
    assert not archive.exists()


def test_rotation_under_bound_untouched(tmp_path, monkeypatch):
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=100)
    for i in range(3):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    assert len(log_file.read_text().splitlines()) == 3
    assert not archive.exists()


# ---------------------------------------------------------------------------
# rotation hardening — locking (P1) and streaming (P2)
# ---------------------------------------------------------------------------

def test_rotation_concurrent_appends_lose_nothing(tmp_path, monkeypatch):
    """P1: a thread appending (and rotating) while the main thread does the
    same — every record must land exactly once across live + archive. Without
    the sidecar flock, an append landing between rotation's read and its
    os.replace vanishes with the replaced file."""
    import threading
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=7)
    n_each = 150
    errors = []

    def worker(tag):
        try:
            for i in range(n_each):
                log_query(kind="query", question=f"{tag}{i}", corpus="/g.json")
        except Exception as exc:  # pragma: no cover — log_query must not raise
            errors.append(exc)

    t = threading.Thread(target=worker, args=("t",))
    t.start()
    worker("m")
    t.join()

    assert not errors
    live = log_file.read_text().splitlines()
    arch = archive.read_text().splitlines() if archive.exists() else []
    questions = [json.loads(line)["question"] for line in arch + live]
    expected = [f"t{i}" for i in range(n_each)] + [f"m{i}" for i in range(n_each)]
    # Zero lost AND zero duplicated (order interleaves across threads).
    assert sorted(questions) == sorted(expected)
    # The last completed call ends with a rotation, so the live log is bounded.
    assert len(live) <= 7
    # Per-thread order is preserved within the archive+live concatenation.
    for tag in ("t", "m"):
        seen = [q for q in questions if q.startswith(tag)]
        assert seen == [f"{tag}{i}" for i in range(n_each)]


def test_rotation_streams_bounded_memory(tmp_path, monkeypatch):
    """P2: rotating an oversized log must not load it whole. tracemalloc peak
    stays far below the file size (the readlines-twice version peaked at
    multiples of it); memory scales with the keep-count, not the log."""
    import tracemalloc
    from graphify.querylog import _rotate

    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=100)
    pad = "x" * 180
    with log_file.open("w", encoding="utf-8") as fh:
        for i in range(40_000):
            fh.write(json.dumps({"question": f"q{i}", "pad": pad}) + "\n")
    size = log_file.stat().st_size
    assert size > 6_000_000  # genuinely oversized — rotation's exact target

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _rotate(log_file, 100)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < size // 4  # streamed, not loaded whole
    assert peak < 2_000_000  # absolute: offset deque + 64 KiB copy buffers

    live = log_file.read_text().splitlines()
    assert len(live) == 100
    assert json.loads(live[0])["question"] == "q39900"
    assert json.loads(live[-1])["question"] == "q39999"
    with archive.open("rb") as fh:
        archived = sum(1 for _ in fh)
    assert archived == 39_900


def test_rotation_without_fcntl_still_rotates(tmp_path, monkeypatch):
    """Degraded (non-POSIX) path: with the lock unavailable, rotation still
    works single-threaded — the pre-lock narrowed-window behavior."""
    import graphify.querylog as ql
    monkeypatch.setattr(ql, "fcntl", None)
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=3)
    for i in range(8):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    live = [json.loads(line)["question"] for line in log_file.read_text().splitlines()]
    arch = [json.loads(line)["question"] for line in archive.read_text().splitlines()]
    assert live == ["q5", "q6", "q7"]
    assert arch == ["q0", "q1", "q2", "q3", "q4"]


def test_rotation_uses_sidecar_lockfile(tmp_path, monkeypatch):
    """The lock anchors on a sidecar (stable inode), not the rotated log."""
    pytest.importorskip("fcntl")
    log_file, archive = _rot_env(monkeypatch, tmp_path, max_records=2)
    for i in range(3):
        log_query(kind="query", question=f"q{i}", corpus="/g.json")
    assert log_file.with_name(log_file.name + ".lock").exists()

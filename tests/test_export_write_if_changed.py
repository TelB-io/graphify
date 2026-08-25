"""An export that produces the same bytes must not rewrite the files.

`to_wiki` and `to_obsidian` regenerate the FULL set of pages on every call, and
a call is triggered by any graph change. On a real vault that is tens of
thousands of small files rewritten byte-for-identical-byte over and over: wasted
disk writes, and a modification event per file for everything watching the
output (Obsidian's indexer, sync clients, inotify pipelines).

These tests pin the behaviour by mtime, because mtime is exactly what the
watchers downstream react to. Each one stamps a known old mtime on the output,
re-runs the export, and asserts the stamp survived — a rewrite of identical
content would move it to "now".
"""
import json
import os
from pathlib import Path

import networkx as nx
import pytest

from graphify.export import to_obsidian, to_canvas
from graphify.paths import write_text_if_changed
from graphify.wiki import to_wiki

_OLD = 1_000_000_000  # a fixed mtime in the past, in seconds


def _stamp_old(paths) -> dict:
    """Set every file's mtime to _OLD and return {path: mtime_ns} for comparison."""
    stamps = {}
    for p in paths:
        os.utime(p, (_OLD, _OLD))
        stamps[p] = p.stat().st_mtime_ns
    return stamps


def _moved(stamps: dict) -> list:
    return [p.name for p, ns in stamps.items() if p.stat().st_mtime_ns != ns]


def _graph():
    G = nx.Graph()
    G.add_node("n1", label="Database", community=0, source_file="app/db.py", type="code")
    G.add_node("n2", label="Server", community=0, source_file="app/srv.py", type="code")
    G.add_node("n3", label="Cache", community=1, source_file="infra/cache.py", type="code")
    G.add_node("n4", label="Queue", community=1, source_file="infra/queue.py", type="code")
    G.add_edge("n1", "n2")
    G.add_edge("n3", "n4")
    G.add_edge("n2", "n3")
    return G, {0: ["n1", "n2"], 1: ["n3", "n4"]}


# ── the primitive ──────────────────────────────────────────────────────────────

def test_write_text_if_changed_skips_identical_content(tmp_path):
    target = tmp_path / "note.md"
    assert write_text_if_changed(target, "hello\n") is True
    stamps = _stamp_old([target])
    assert write_text_if_changed(target, "hello\n") is False
    assert _moved(stamps) == []
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_write_text_if_changed_writes_when_content_differs(tmp_path):
    target = tmp_path / "note.md"
    write_text_if_changed(target, "hello\n")
    stamps = _stamp_old([target])
    assert write_text_if_changed(target, "goodbye\n") is True
    assert _moved(stamps) == ["note.md"]
    assert target.read_text(encoding="utf-8") == "goodbye\n"


def test_write_text_if_changed_creates_missing_parents(tmp_path):
    target = tmp_path / "deep" / "deeper" / "note.md"
    assert write_text_if_changed(target, "x\n") is True
    assert target.read_text(encoding="utf-8") == "x\n"


def test_write_text_if_changed_overwrites_undecodable_file(tmp_path):
    """A file that is not valid UTF-8 cannot be compared — it must be rewritten,
    not silently left in place."""
    target = tmp_path / "note.md"
    target.write_bytes(b"\xff\xfe not utf-8")
    assert write_text_if_changed(target, "clean\n") is True
    assert target.read_text(encoding="utf-8") == "clean\n"


# ── the Obsidian exporter ──────────────────────────────────────────────────────

def test_to_obsidian_rerun_rewrites_nothing(tmp_path, capsys):
    G, communities = _graph()
    out = tmp_path / "vault"
    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    files = sorted(p for p in out.rglob("*") if p.is_file())
    assert files, "export produced no files"
    stamps = _stamp_old(files)
    capsys.readouterr()

    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})

    assert _moved(stamps) == []
    # no file appeared or vanished either
    assert sorted(p for p in out.rglob("*") if p.is_file()) == files
    assert "left untouched" in capsys.readouterr().err


def test_to_obsidian_rerun_still_writes_the_note_that_changed(tmp_path):
    G, communities = _graph()
    out = tmp_path / "vault"
    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    stamps = _stamp_old(sorted(p for p in out.rglob("*") if p.is_file()))

    # Give Database one more neighbour: its note (and the community overview it
    # belongs to) must be rewritten, the notes in the untouched community must not.
    G.add_node("n5", label="Migrations", community=0, source_file="app/mig.py", type="code")
    communities[0].append("n5")
    G.add_edge("n1", "n5")
    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})

    moved = set(_moved(stamps))
    assert "Database.md" in moved
    assert "Cache.md" not in moved and "Queue.md" not in moved
    assert (out / "Migrations.md").exists()


def test_to_obsidian_unchanged_notes_stay_owned_and_are_not_pruned(tmp_path):
    """The manifest is also the prune-exclusion list: a note skipped as unchanged
    must still be recorded as graphify's, or the same run would delete it."""
    G, communities = _graph()
    out = tmp_path / "vault"
    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    to_obsidian(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})

    assert (out / "Database.md").exists()
    manifest = json.loads((out / ".graphify_obsidian_manifest.json").read_text(encoding="utf-8"))
    assert "Database.md" in manifest["files"]
    assert "Cache.md" in manifest["files"]


def test_to_canvas_rerun_does_not_rewrite(tmp_path):
    G, communities = _graph()
    canvas = tmp_path / "graph.canvas"
    to_canvas(G, communities, str(canvas), community_labels={0: "Backend", 1: "Infra"})
    stamps = _stamp_old([canvas])
    to_canvas(G, communities, str(canvas), community_labels={0: "Backend", 1: "Infra"})
    assert _moved(stamps) == []


# ── the wiki exporter ──────────────────────────────────────────────────────────

def test_to_wiki_rerun_rewrites_nothing(tmp_path, capsys):
    G, communities = _graph()
    out = tmp_path / "wiki"
    gods = [{"id": "n2", "label": "Server", "degree": 2}]
    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"},
            god_nodes_data=gods)
    files = sorted(out.glob("*.md"))
    assert files, "wiki export produced no articles"
    stamps = _stamp_old(files)
    capsys.readouterr()

    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"},
            god_nodes_data=gods)

    assert _moved(stamps) == []
    assert sorted(out.glob("*.md")) == files
    assert "left untouched" in capsys.readouterr().err


def test_to_wiki_still_removes_orphans_after_a_relabel(tmp_path):
    """Community labels are LLM-generated and drift between runs; the article for
    the previous name must still disappear now that the wipe is a targeted sweep."""
    G, communities = _graph()
    out = tmp_path / "wiki"
    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    assert (out / "Infra.md").exists()

    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infrastructure"})

    assert not (out / "Infra.md").exists(), "orphaned article survived the re-export"
    assert (out / "Infrastructure.md").exists()
    assert (out / "Backend.md").exists()


def test_to_wiki_rewrites_the_article_that_changed(tmp_path):
    G, communities = _graph()
    out = tmp_path / "wiki"
    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    stamps = _stamp_old(sorted(out.glob("*.md")))

    G.add_node("n5", label="Migrations", community=0, source_file="app/mig.py", type="code")
    communities[0].append("n5")
    G.add_edge("n1", "n5")
    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})

    moved = set(_moved(stamps))
    assert "Backend.md" in moved, "the community that gained a node was not rewritten"
    assert "index.md" in moved, "the index counts nodes/edges — it changed"


def test_to_wiki_never_leaves_a_gap_for_readers(tmp_path):
    """The old code deleted every article before writing any, so a reader between
    the two steps saw an empty wiki. With the sweep at the end, every article that
    survives the run is present on disk for the whole run."""
    G, communities = _graph()
    out = tmp_path / "wiki"
    to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    before = {p.name: p.read_text(encoding="utf-8") for p in out.glob("*.md")}

    seen_during: list[set] = []
    real_write = write_text_if_changed

    import graphify.wiki as wiki_mod

    def _spy(path, text):
        seen_during.append({p.name for p in out.glob("*.md")})
        return real_write(path, text)

    wiki_mod.write_text_if_changed = _spy
    try:
        to_wiki(G, communities, str(out), community_labels={0: "Backend", 1: "Infra"})
    finally:
        wiki_mod.write_text_if_changed = real_write

    assert seen_during, "spy never fired"
    for snapshot in seen_during:
        assert set(before) <= snapshot, "an article vanished mid-export"

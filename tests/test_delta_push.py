"""Unit tests for the delta DB push (only changed repos; pruned nodes removed).

The full push is MERGE-only, so it never removes anything: once
``graphify global add`` prunes a repo's stale nodes out of the global graph, a
full re-push leaves every one of them behind in the database. Measured against
a live FalkorDB on a 75,000-node/40-repo corpus with one repo refreshed
(-1,250 pruned, +1,750 added): a full re-push landed 76,750 nodes / 75,315
edges where a from-scratch load of the same state gives 75,500 / 74,164 - a
permanent surplus of exactly the pruned nodes and their edges. The delta push
landed 75,500 / 74,164, byte-for-byte the from-scratch content (identical
per-repo counts, per-relation counts, and a checksum over every node's
id+community+file_type), while sending 2.6% of the rows.

These tests pin the parts that make that safe:

- the planner: a repo is re-pushed when its manifest hash moved, when the
  database's own count for it disagrees with the manifest (the self-healing
  arm), and dropped when the manifest no longer lists it;
- delete-before-write ordering, and the delete paging idiom - FalkorDB
  documents that LIMIT does not short-circuit eager DELETE, so the LIMIT must
  sit in a WITH that PRECEDES the DELETE or the whole label is wiped;
- row filtering: only changed repos' nodes travel, and an edge travels when
  either endpoint belongs to a changed repo;
- the ledger is written only after the rows land, so an interrupted repo is
  redone rather than marked clean;
- the refusal when there is no global manifest, instead of silently pushing
  nothing.

The fake driver records every query; no server, no network.
"""
from __future__ import annotations

import json
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Fake FalkorDB that can answer the two reads the delta path makes
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, result_set=None, nodes_deleted=0):
        self.result_set = result_set or []
        self.nodes_deleted = nodes_deleted


def _fake_falkordb(recorded: list, repo_counts: dict | None = None,
                   deletable: dict | None = None) -> types.ModuleType:
    """Records queries; answers the per-repo count read and the delete pages."""
    counts = dict(repo_counts or {})
    left = dict(deletable or {})

    class _Graph:
        def query(self, cypher, params=None):
            recorded.append((cypher, params))
            if "RETURN n.repo, count(n)" in cypher:
                return _Result([[k, v] for k, v in counts.items()])
            if "DETACH DELETE" in cypher:
                tag = (params or {}).get("repo")
                gone = left.pop(tag, 0)
                return _Result(nodes_deleted=gone)
            return _Result()

    class FalkorDB:
        def __init__(self, host, port, username=None, password=None):
            pass

        def select_graph(self, name):
            return _Graph()

    mod = types.ModuleType("falkordb")
    mod.FalkorDB = FalkorDB
    return mod


def _corpus(tmp_path, repos: dict[str, int], *, extra_links=()):
    """A global-shaped graph.json + manifest: ids prefixed `<repo>::`."""
    nodes, links = [], []
    for tag, n in repos.items():
        for i in range(n):
            nodes.append({"id": f"{tag}::n{i}", "repo": tag, "label": f"{tag} {i}",
                          "file_type": "python"})
        for i in range(max(n - 1, 0)):
            links.append({"source": f"{tag}::n{i}", "target": f"{tag}::n{i+1}",
                          "relation": "calls"})
    links.extend(extra_links)
    graph = tmp_path / "global-graph.json"
    graph.write_text(json.dumps({"directed": True, "multigraph": False, "graph": {},
                                 "nodes": nodes, "links": links}), encoding="utf-8")
    manifest = tmp_path / "global-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "repos": {
        tag: {"node_count": n, "edge_count": max(n - 1, 0),
              "source_hash": f"h-{tag}-{n}", "source_path": f"/{tag}",
              "added_at": "2026-08-25T00:00:00+00:00"}
        for tag, n in repos.items()}}), encoding="utf-8")
    return graph, manifest


def _writes(recorded):
    return [(c, p) for c, p in recorded if "UNWIND $rows AS row" in c]


def _deletes(recorded):
    return [(c, p) for c, p in recorded if "DETACH DELETE" in c]


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def test_plan_delta_skips_a_repo_whose_hash_and_count_both_match():
    from graphify.exporters.graphdb import plan_delta
    man = {"a": {"source_hash": "h1", "node_count": 5}}
    assert plan_delta(man, {"a": "h1"}, {"a": 5}) == (set(), set())


def test_plan_delta_repushes_a_repo_whose_hash_moved():
    from graphify.exporters.graphdb import plan_delta
    man = {"a": {"source_hash": "h2", "node_count": 5}}
    assert plan_delta(man, {"a": "h1"}, {"a": 5}) == ({"a"}, set())


def test_plan_delta_repushes_when_the_database_disagrees_with_the_manifest():
    """The self-healing arm: an emptied or half-written database repairs itself
    on the next run instead of staying wrong because the ledger says it is clean."""
    from graphify.exporters.graphdb import plan_delta
    man = {"a": {"source_hash": "h1", "node_count": 5}}
    assert plan_delta(man, {"a": "h1"}, {}) == ({"a"}, set())          # never arrived
    assert plan_delta(man, {"a": "h1"}, {"a": 3}) == ({"a"}, set())    # landed short


def test_plan_delta_drops_a_repo_the_manifest_no_longer_lists():
    from graphify.exporters.graphdb import plan_delta
    assert plan_delta({}, {"a": "h1"}, {"a": 5}) == (set(), {"a"})
    # Known to the database but never to this ledger - still dropped.
    assert plan_delta({}, {}, {"a": 5}) == (set(), {"a"})


def test_plan_delta_pushes_a_repo_the_ledger_has_never_seen():
    from graphify.exporters.graphdb import plan_delta
    man = {"a": {"source_hash": "h1", "node_count": 5}}
    assert plan_delta(man, {}, {}) == ({"a"}, set())


# ---------------------------------------------------------------------------
# The delete idiom - the FalkorDB LIMIT trap
# ---------------------------------------------------------------------------

def test_delete_applies_limit_before_the_delete_not_after():
    """FalkorDB's known-limitations doc: LIMIT "does not currently short-circuit
    eager operations like CREATE, SET, or DELETE". A trailing LIMIT on a DELETE
    therefore deletes everything the MATCH found - the whole label, not a page.
    """
    from graphify.exporters.graphdb import _DELETE_REPO_PAGE
    head, _, tail = _DELETE_REPO_PAGE.partition("DETACH DELETE")
    assert "LIMIT" in head, "LIMIT must precede DETACH DELETE"
    assert "LIMIT" not in tail, "a trailing LIMIT would wipe the whole label"
    assert "WITH n LIMIT" in head


def test_delete_is_scoped_to_one_repo_by_parameter():
    from graphify.exporters.graphdb import _DELETE_REPO_PAGE
    assert "n.repo = $repo" in _DELETE_REPO_PAGE
    assert "GraphifyNode" in _DELETE_REPO_PAGE


# ---------------------------------------------------------------------------
# End-to-end shape against the fake driver
# ---------------------------------------------------------------------------

def test_only_changed_repos_travel_and_stale_nodes_are_deleted_first(tmp_path, monkeypatch):
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"keep": 3, "moved": 2})
    # Ledger says both were pushed, but "moved" carries a stale hash.
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "repos": {
        "keep": "h-keep-3", "moved": "OLD"}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "falkordb",
                        _fake_falkordb(recorded, repo_counts={"keep": 3, "moved": 9},
                                       deletable={"moved": 9}))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                                    batch_size=50, manifest_path=manifest,
                                    state_path=ledger)

    assert result["repos_pushed"] == 1
    assert result["nodes_deleted"] == 9
    assert result["skipped"] is False
    # Only the changed repo's rows went out.
    node_rows = [r for c, p in _writes(recorded) if "MERGE (n:" in c for r in p["rows"]]
    assert {r["id"] for r in node_rows} == {"moved::n0", "moved::n1"}
    # And the delete for it preceded every write.
    first_write = next(i for i, (c, _) in enumerate(recorded) if "UNWIND $rows AS row" in c)
    del_idx = [i for i, (c, _) in enumerate(recorded) if "DETACH DELETE" in c]
    assert del_idx and max(del_idx) < first_write


def test_an_edge_travels_when_either_endpoint_belongs_to_a_changed_repo(tmp_path, monkeypatch):
    """A changed repo is DETACH DELETEd, which detaches its cross-repo edges too,
    so those edges must be re-MERGEd even though the far endpoint never moved."""
    recorded: list = []
    cross = [{"source": "moved::n0", "target": "keep::n0", "relation": "references"},
             {"source": "keep::n1", "target": "keep::n2", "relation": "references"}]
    graph, manifest = _corpus(tmp_path, {"keep": 3, "moved": 2}, extra_links=cross)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "repos": {
        "keep": "h-keep-3", "moved": "OLD"}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "falkordb",
                        _fake_falkordb(recorded, repo_counts={"keep": 3, "moved": 2}))
    from graphify.export import delta_push_to_falkordb

    delta_push_to_falkordb(graph, "falkordb://x", graph_name="g", batch_size=50,
                           manifest_path=manifest, state_path=ledger)

    sent = {(r["src"], r["tgt"]) for c, p in _writes(recorded)
            if "MERGE (a)-[r:" in c for r in p["rows"]}
    assert ("moved::n0", "keep::n0") in sent      # crosses into the changed repo
    assert ("keep::n1", "keep::n2") not in sent   # wholly inside an untouched repo


def test_nothing_changed_means_nothing_is_sent(tmp_path, monkeypatch):
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 2, "b": 2})
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "repos": {
        "a": "h-a-2", "b": "h-b-2"}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "falkordb",
                        _fake_falkordb(recorded, repo_counts={"a": 2, "b": 2}))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                                    manifest_path=manifest, state_path=ledger)

    assert result["skipped"] is True
    assert (result["nodes"], result["edges"]) == (0, 0)
    assert _writes(recorded) == []
    assert _deletes(recorded) == []


def test_the_ledger_records_only_what_was_pushed(tmp_path, monkeypatch):
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 2, "b": 2})
    ledger = tmp_path / "ledger.json"
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(recorded))
    from graphify.export import delta_push_to_falkordb

    delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                           manifest_path=manifest, state_path=ledger)

    assert json.loads(ledger.read_text())["repos"] == {"a": "h-a-2", "b": "h-b-2"}


def test_a_crash_before_the_ledger_write_leaves_the_repo_dirty(tmp_path, monkeypatch):
    """The ledger is written last, so an interrupted push is redone next run
    rather than being recorded as clean."""
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 2})
    ledger = tmp_path / "ledger.json"

    boom = _fake_falkordb(recorded)
    real_select = boom.FalkorDB.select_graph

    def _explode(self, name):
        g = real_select(self, name)
        original = g.query

        def q(cypher, params=None):
            if "UNWIND $rows AS row" in cypher:
                raise RuntimeError("connection lost mid-write")
            return original(cypher, params)
        g.query = q
        return g
    boom.FalkorDB.select_graph = _explode
    monkeypatch.setitem(sys.modules, "falkordb", boom)
    from graphify.export import delta_push_to_falkordb

    with pytest.raises(RuntimeError):
        delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                               manifest_path=manifest, state_path=ledger)
    assert not ledger.exists(), "an interrupted push must not be recorded as clean"


def test_delta_refuses_a_graph_with_no_global_manifest(tmp_path, monkeypatch):
    """A single-project graph has no repo tags. Refuse loudly - a delta that
    silently sends nothing is the exact failure this path exists to prevent."""
    recorded: list = []
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"directed": True, "multigraph": False, "graph": {},
                                 "nodes": [{"id": "a", "file_type": "python"}],
                                 "links": []}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(recorded))
    from graphify.export import delta_push_to_falkordb

    with pytest.raises(ValueError, match="global manifest"):
        delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                               manifest_path=tmp_path / "nope.json")
    assert recorded == [], "nothing may be sent when the delta cannot be planned"


def test_ledger_path_is_keyed_by_target(tmp_path):
    """The same graph.json feeding two databases must not let one convince the
    other it is already up to date."""
    from graphify.exporters.graphdb import _push_state_path
    g = tmp_path / "global-graph.json"
    a = _push_state_path(g, "falkordb://host-a:6379", "graphify")
    b = _push_state_path(g, "falkordb://host-b:6379", "graphify")
    c = _push_state_path(g, "falkordb://host-a:6379", "other")
    assert a != b and a != c and b != c
    assert a.parent == g.parent


def test_manifest_defaults_beside_the_graph(tmp_path, monkeypatch):
    """`global-graph.json` implies `global-manifest.json`, the layout
    graphify.global_graph already writes."""
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 2})
    assert manifest.name == "global-manifest.json"
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(recorded))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                                    state_path=tmp_path / "l.json")
    assert result["repos_pushed"] == 1


# ---------------------------------------------------------------------------
# The drop guard - refuse to silently delete a corpus that is merely unfamiliar
# ---------------------------------------------------------------------------

def test_a_mass_drop_is_refused_before_anything_is_deleted(tmp_path, monkeypatch):
    """The real incident this guard exists for: a 40-repo manifest aimed at a
    226-repo graph classified every real repo as removed and deleted 1,211,189
    nodes. Size is the only signal that separates that from a genuine removal.
    """
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"mine": 10})
    live = {f"theirs-{i}": 100 for i in range(20)}      # 2,000 foreign nodes
    live["mine"] = 10
    monkeypatch.setitem(sys.modules, "falkordb",
                        _fake_falkordb(recorded, repo_counts=live))
    from graphify.export import delta_push_to_falkordb

    with pytest.raises(ValueError, match="delta push refused"):
        delta_push_to_falkordb(graph, "falkordb://x", graph_name="graphify",
                               manifest_path=manifest, state_path=tmp_path/"l.json")
    assert _deletes(recorded) == [], "the guard must fire before any delete"
    assert _writes(recorded) == [], "and before any write"
    assert not (tmp_path/"l.json").exists()


def test_the_refusal_names_the_damage_and_the_way_out(tmp_path, monkeypatch):
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"mine": 10})
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(
        recorded, repo_counts={"mine": 10, "a": 500, "b": 500}))
    from graphify.export import delta_push_to_falkordb

    with pytest.raises(ValueError) as ei:
        delta_push_to_falkordb(graph, "falkordb://x", graph_name="graphify",
                               manifest_path=manifest, state_path=tmp_path/"l.json")
    msg = str(ei.value)
    assert "1000 of 1010 nodes" in msg and "99.0%" in msg
    assert "graphify" in msg and str(manifest) in msg
    assert "--allow-drop" in msg


def test_allow_drop_lets_a_genuine_mass_removal_through(tmp_path, monkeypatch):
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"mine": 10})
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(
        recorded, repo_counts={"mine": 10, "a": 500}, deletable={"a": 500}))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="graphify",
                                    manifest_path=manifest, state_path=tmp_path/"l.json",
                                    allow_drop=True)
    assert result["repos_dropped"] == 1
    assert result["nodes_deleted"] == 500


def test_a_small_drop_passes_without_the_flag(tmp_path, monkeypatch):
    """A repo genuinely retired from a large corpus is ordinary business."""
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 100})
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(
        recorded, repo_counts={"a": 100, "gone": 5}, deletable={"gone": 5}))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="graphify",
                                    manifest_path=manifest, state_path=tmp_path/"l.json")
    assert result["repos_dropped"] == 1 and result["nodes_deleted"] == 5


def test_the_guard_does_not_fire_on_an_empty_target(tmp_path, monkeypatch):
    """First push into a fresh graph drops nothing; there is nothing to protect."""
    recorded: list = []
    graph, manifest = _corpus(tmp_path, {"a": 3})
    monkeypatch.setitem(sys.modules, "falkordb", _fake_falkordb(recorded, repo_counts={}))
    from graphify.export import delta_push_to_falkordb

    result = delta_push_to_falkordb(graph, "falkordb://x", graph_name="g",
                                    manifest_path=manifest, state_path=tmp_path/"l.json")
    assert result["repos_pushed"] == 1 and result["repos_dropped"] == 0

"""The global graph is updated one repo-slice at a time, without loading it.

Two properties matter, in this order:

1. **Correctness.** The streamed rewrite must produce what the whole-load path
   produced — same nodes, same links, same attributes, and (when nothing is
   deduplicated onto an existing external node) the same bytes. The oracle
   below, :func:`_reference_global_add`, is the pre-streaming algorithm kept
   verbatim so the comparison is against real behaviour, not a restatement of
   the new code.
2. **Memory.** Peak allocation must track the size of the CHANGE, not the size
   of the store. That is the whole point: on the real 1.888GB global graph the
   whole-load path peaked at ~5.0GB RSS to update 2.2% of the nodes, which is
   what OOM-killed it on a 7.6GB box.
"""
from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest
from networkx.readwrite import json_graph as jg


# ── the oracle: the whole-load algorithm, kept verbatim ──────────────────────

def _reference_global_add(global_path: Path, source_path: Path, repo_tag: str) -> dict:
    """The pre-streaming ``global_add`` merge, operating on explicit paths.

    Deliberately a copy rather than a call: it is the behaviour the streamed
    implementation has to reproduce, so it must not change when that one does.
    """
    from graphify.build import prefix_graph_for_global, prune_repo_from_graph

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if "links" not in data and "edges" in data:
        data = dict(data, links=data["edges"])
    try:
        src_G = jg.node_link_graph(data, edges="links")
    except TypeError:
        src_G = jg.node_link_graph(data)
    prefixed = prefix_graph_for_global(src_G, repo_tag)

    if global_path.exists():
        gdata = json.loads(global_path.read_text(encoding="utf-8"))
        if "links" not in gdata and "edges" in gdata:
            gdata = dict(gdata, links=gdata["edges"])
        try:
            G = jg.node_link_graph(gdata, edges="links")
        except TypeError:
            G = jg.node_link_graph(gdata)
    else:
        G = nx.Graph()
    removed = prune_repo_from_graph(G, repo_tag)

    external_labels = {
        d.get("label", ""): n
        for n, d in G.nodes(data=True)
        if not d.get("source_file") and d.get("label")
    }
    remap = {}
    for node, ndata in prefixed.nodes(data=True):
        if not ndata.get("source_file") and ndata.get("label") in external_labels:
            remap[node] = external_labels[ndata["label"]]

    for node, ndata in prefixed.nodes(data=True):
        if node not in remap:
            G.add_node(node, **ndata)
    for u, v, edata in prefixed.edges(data=True):
        u = remap.get(u, u)
        v = remap.get(v, v)
        if u != v:
            G.add_edge(u, v, **edata)

    added = prefixed.number_of_nodes() - len(remap)
    try:
        out = jg.node_link_data(G, edges="links")
    except TypeError:
        out = jg.node_link_data(G)
    global_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {"nodes_added": added, "nodes_removed": removed}


def _reference_global_remove(global_path: Path, repo_tag: str) -> int:
    from graphify.build import prune_repo_from_graph

    gdata = json.loads(global_path.read_text(encoding="utf-8"))
    if "links" not in gdata and "edges" in gdata:
        gdata = dict(gdata, links=gdata["edges"])
    try:
        G = jg.node_link_graph(gdata, edges="links")
    except TypeError:
        G = jg.node_link_graph(gdata)
    removed = prune_repo_from_graph(G, repo_tag)
    try:
        out = jg.node_link_data(G, edges="links")
    except TypeError:
        out = jg.node_link_data(G)
    global_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return removed


# ── fixtures ─────────────────────────────────────────────────────────────────

def _repo_graph(path: Path, tag: str, n_nodes: int, *, externals: "list[str] | None" = None):
    """A small per-repo graph.json: a chain of source-backed nodes, plus any
    named external (``source_file``-less) nodes hung off the first node."""
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(
            f"{tag}_n{i}",
            label=f"{tag} node {i}",
            source_file=f"src/{tag}/mod{i}.py",
            source_location=f"L{i + 1}",
            node_kind="function",
        )
        if i:
            G.add_edge(f"{tag}_n{i - 1}", f"{tag}_n{i}", relation="calls", weight=1.0)
    for j, label in enumerate(externals or []):
        ext = f"{tag}_ext{j}"
        G.add_node(ext, label=label, source_file="", source_location="")
        G.add_edge(f"{tag}_n0", ext, relation="imports", weight=1.0)
    try:
        data = jg.node_link_data(G, edges="links")
    except TypeError:
        data = jg.node_link_data(G)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def store(tmp_path):
    """A global-graph directory with the module's paths pointed at it."""
    gdir = tmp_path / "globaldir"
    gdir.mkdir()
    gpath = gdir / "global-graph.json"
    with patch("graphify.global_graph._GLOBAL_DIR", gdir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", gpath), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", gdir / "global-manifest.json"):
        yield gpath


def _load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def _canonical(data):
    """(nodes, links) as order-insensitive comparable sets of attributes."""
    nodes = {n["id"]: tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in n.items()))
             for n in data["nodes"]}
    links = {}
    for link in data["links"]:
        key = tuple(sorted((link["source"], link["target"])))
        links[key] = tuple(sorted(
            (k, json.dumps(v, sort_keys=True))
            for k, v in link.items() if k not in ("source", "target")
        ))
    return nodes, links


def _slice_hash(data, repo_tag):
    """A repo's own slice: its nodes and the links wholly inside it, canonical."""
    import hashlib

    own = {n["id"] for n in data["nodes"] if n.get("repo") == repo_tag}
    payload = [json.dumps(n, sort_keys=True) for n in data["nodes"] if n["id"] in own]
    payload += sorted(
        json.dumps(link, sort_keys=True)
        for link in data["links"]
        if link["source"] in own and link["target"] in own
    )
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()


# ── correctness ──────────────────────────────────────────────────────────────

def test_streamed_add_is_byte_identical_to_whole_load(store, tmp_path):
    """With nothing to deduplicate, the streamed file matches the whole-load
    file byte for byte — including NetworkX's key order and json.dump indenting."""
    from graphify.global_graph import _rewrite_global_streamed

    sources = {tag: _repo_graph(tmp_path / f"{tag}.json", tag, 6) for tag in ("repoA", "repoB", "repoC")}

    # Build the same three-repo store twice, one way each.
    ref = tmp_path / "reference-global.json"
    for tag, src in sources.items():
        _reference_global_add(ref, src, tag)
        _add_streamed(store, src, tag)
    assert store.read_text() == ref.read_text()

    # Now update the middle repo — the slice case the refresh service hits.
    changed = _repo_graph(tmp_path / "repoB2.json", "repoB", 9)
    _reference_global_add(ref, changed, "repoB")
    _add_streamed(store, changed, "repoB")
    assert store.read_text() == ref.read_text()


def _add_streamed(global_path: Path, source_path: Path, repo_tag: str):
    """Drive the streamed rewrite the way ``global_add`` does."""
    from graphify.build import prefix_graph_for_global
    from graphify.global_graph import _rewrite_global_streamed

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if "links" not in data and "edges" in data:
        data = dict(data, links=data["edges"])
    try:
        src_G = jg.node_link_graph(data, edges="links")
    except TypeError:
        src_G = jg.node_link_graph(data)
    prefixed = prefix_graph_for_global(src_G, repo_tag)
    try:
        new = jg.node_link_data(prefixed, edges="links")
    except TypeError:
        new = jg.node_link_data(prefixed)
    return _rewrite_global_streamed(repo_tag, new["nodes"], new.get("links") or new.get("edges") or [])


def test_streamed_add_matches_whole_load_with_external_dedup(store, tmp_path):
    """External-library nodes are shared across repos, so updating a repo rewires
    edges onto nodes that belong to OTHER repos. The graphs must still match."""
    ref = tmp_path / "reference-global.json"
    sources = {
        "repoA": _repo_graph(tmp_path / "a.json", "repoA", 5, externals=["Path", "json"]),
        "repoB": _repo_graph(tmp_path / "b.json", "repoB", 5, externals=["Path", "os"]),
    }
    for tag, src in sources.items():
        _reference_global_add(ref, src, tag)
        _add_streamed(store, src, tag)

    changed = _repo_graph(tmp_path / "b2.json", "repoB", 7, externals=["Path", "os", "sys"])
    ref_result = _reference_global_add(ref, changed, "repoB")
    added, removed = _add_streamed(store, changed, "repoB")

    assert (added, removed) == (ref_result["nodes_added"], ref_result["nodes_removed"])
    assert _canonical(_load(store)) == _canonical(_load(ref))


def test_streamed_add_merges_attributes_on_rewired_external_edge(store, tmp_path):
    """A rewired incoming edge can land on a pair that already has a surviving
    edge between two external nodes (both endpoints dedup onto nodes outside
    the repo being replaced). ``add_edge(u, v, **attr)`` updates the existing
    attribute dict rather than replacing it, so an attribute the old edge has
    and the incoming edge doesn't must survive the rewrite."""

    def _bridge_repo(path: Path, tag: str, edge_attrs: dict):
        G = nx.Graph()
        G.add_node(
            f"{tag}_n0", label=f"{tag} node 0", source_file=f"src/{tag}/mod0.py",
            source_location="L1", node_kind="function",
        )
        ext_ids = []
        for j, label in enumerate(("X", "Y")):
            ext = f"{tag}_ext{j}"
            G.add_node(ext, label=label, source_file="", source_location="")
            G.add_edge(f"{tag}_n0", ext, relation="imports", weight=1.0)
            ext_ids.append(ext)
        G.add_edge(ext_ids[0], ext_ids[1], **edge_attrs)
        try:
            data = jg.node_link_data(G, edges="links")
        except TypeError:
            data = jg.node_link_data(G)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    ref = tmp_path / "reference-global.json"

    repoA_src = _bridge_repo(tmp_path / "a.json", "repoA", {"relation": "depends", "weight": 5.0, "provenance": "repoA"})
    _reference_global_add(ref, repoA_src, "repoA")
    _add_streamed(store, repoA_src, "repoA")

    # repoB's X/Y both dedup onto repoA's X/Y, so its X-Y edge rewires onto the
    # exact pair repoA already established — but supplies neither `weight` nor
    # `provenance`.
    repoB_src = _bridge_repo(tmp_path / "b.json", "repoB", {"relation": "depends"})
    _reference_global_add(ref, repoB_src, "repoB")
    _add_streamed(store, repoB_src, "repoB")

    assert _canonical(_load(store)) == _canonical(_load(ref))


def test_streamed_add_leaves_other_repo_slices_untouched(store, tmp_path):
    """The neighbour test: updating one repo must not perturb a single byte of
    any other repo's slice. A streaming bug that corrupts a neighbour shows up
    here and nowhere else."""
    for tag in ("repoA", "repoB", "repoC"):
        _add_streamed(store, _repo_graph(tmp_path / f"{tag}.json", tag, 6, externals=["Path"]), tag)
    before = {tag: _slice_hash(_load(store), tag) for tag in ("repoA", "repoB", "repoC")}

    _add_streamed(store, _repo_graph(tmp_path / "b2.json", "repoB", 11, externals=["Path"]), "repoB")
    after = {tag: _slice_hash(_load(store), tag) for tag in ("repoA", "repoB", "repoC")}

    assert after["repoA"] == before["repoA"]
    assert after["repoC"] == before["repoC"]
    assert after["repoB"] != before["repoB"]
    assert len({n["id"] for n in _load(store)["nodes"] if n.get("repo") == "repoB"}) == 11


def test_streamed_remove_matches_whole_load(store, tmp_path):
    from graphify.global_graph import _rewrite_global_streamed

    ref = tmp_path / "reference-global.json"
    for tag in ("repoA", "repoB"):
        src = _repo_graph(tmp_path / f"{tag}.json", tag, 5, externals=["Path"])
        _reference_global_add(ref, src, tag)
        _add_streamed(store, src, tag)

    ref_removed = _reference_global_remove(ref, "repoA")
    _, removed = _rewrite_global_streamed("repoA")

    assert removed == ref_removed
    assert store.read_text() == ref.read_text()


def test_streamed_add_into_empty_store(store, tmp_path):
    """No global file yet: the whole-load path composed into a fresh nx.Graph(),
    so the header must come out undirected, non-multi, with no graph attrs."""
    _add_streamed(store, _repo_graph(tmp_path / "a.json", "repoA", 3), "repoA")
    data = _load(store)
    assert list(data) == ["directed", "multigraph", "graph", "nodes", "links"]
    assert (data["directed"], data["multigraph"], data["graph"]) == (False, False, {})
    assert len(data["nodes"]) == 3


def test_streamed_rewrite_of_last_repo_leaves_valid_empty_arrays(store, tmp_path):
    """Removing the only repo empties both arrays; json.dump renders those as
    ``[]``, and the file must still parse."""
    from graphify.global_graph import _rewrite_global_streamed

    _add_streamed(store, _repo_graph(tmp_path / "a.json", "repoA", 3), "repoA")
    _rewrite_global_streamed("repoA")
    data = _load(store)
    assert data["nodes"] == [] and data["links"] == []
    assert '"nodes": []' in store.read_text()


def test_global_add_end_to_end_updates_manifest(store, tmp_path):
    """The public entry point still reports and records what it always did."""
    from graphify.global_graph import global_add, global_list

    src = _repo_graph(tmp_path / "a.json", "repoA", 4)
    first = global_add(src, "repoA")
    assert first["skipped"] is False and first["nodes_added"] == 4

    bigger = _repo_graph(tmp_path / "a.json", "repoA", 6)
    second = global_add(bigger, "repoA")
    assert second["nodes_added"] == 6 and second["nodes_removed"] == 4
    assert global_list()["repoA"]["node_count"] == 6
    assert len(_load(store)["nodes"]) == 6


# ── memory ───────────────────────────────────────────────────────────────────

def test_streamed_add_memory_tracks_the_change_not_the_store(store, tmp_path):
    """Peak allocation must stay bounded by the incoming slice while the store
    grows around it. Updating a 40-node repo inside a store of ~24,000 nodes
    must not cost anything like the store's size.

    Written as a ratio against the file rather than an absolute byte count so it
    stays meaningful on any machine: the whole-load path allocated several times
    the file's size, the streamed path allocates a fraction of it.
    """
    big = nx.Graph()
    for repo in range(60):
        tag = f"bulk{repo:03d}"
        for i in range(400):
            big.add_node(
                f"{tag}::{tag}_n{i}",
                label=f"{tag} node {i}",
                source_file=f"src/{tag}/mod{i}.py",
                source_location=f"L{i + 1}",
                node_kind="function",
                repo=tag,
                local_id=f"{tag}_n{i}",
                blurb="x" * 200,  # give each node realistic bulk
            )
            if i:
                big.add_edge(f"{tag}::{tag}_n{i-1}", f"{tag}::{tag}_n{i}", relation="calls", weight=1.0)
    try:
        data = jg.node_link_data(big, edges="links")
    except TypeError:
        data = jg.node_link_data(big)
    store.write_text(json.dumps(data, indent=2), encoding="utf-8")
    del big, data

    store_bytes = store.stat().st_size
    assert store_bytes > 8_000_000, "the synthetic store must be big enough for the ratio to mean something"

    src = _repo_graph(tmp_path / "small.json", "newrepo", 40)

    tracemalloc.start()
    tracemalloc.reset_peak()
    _add_streamed(store, src, "newrepo")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < store_bytes / 4, (
        f"streamed add peaked at {peak/1e6:.1f}MB against a {store_bytes/1e6:.1f}MB store; "
        "memory is tracking the store, not the change"
    )
    # And it really did the work.
    assert len({n["id"] for n in _load(store)["nodes"] if n.get("repo") == "newrepo"}) == 40
    assert len(_load(store)["nodes"]) == 60 * 400 + 40

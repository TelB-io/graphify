from __future__ import annotations
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph as _jg

_GLOBAL_DIR = Path.home() / ".graphify"
_GLOBAL_GRAPH = _GLOBAL_DIR / "global-graph.json"
_GLOBAL_MANIFEST = _GLOBAL_DIR / "global-manifest.json"


def _load_manifest() -> dict:
    if _GLOBAL_MANIFEST.exists():
        try:
            return json.loads(_GLOBAL_MANIFEST.read_text(encoding="utf-8"))
        except Exception as exc:
            # Don't silently wipe the user's manifest on a parse error: that
            # deletes every tracked repo. Back the bad file up and surface the
            # error so the user can recover or report it.
            backup = _GLOBAL_MANIFEST.with_suffix(
                _GLOBAL_MANIFEST.suffix + f".corrupt.{int(datetime.now(timezone.utc).timestamp())}"
            )
            try:
                _GLOBAL_MANIFEST.rename(backup)
                print(
                    f"[graphify global] manifest at {_GLOBAL_MANIFEST} failed to parse ({exc}); "
                    f"moved to {backup} and starting fresh. Restore from the backup if this was "
                    f"unexpected.",
                    file=sys.stderr,
                )
            except Exception as rename_exc:
                print(
                    f"[graphify global] manifest at {_GLOBAL_MANIFEST} failed to parse ({exc}) "
                    f"and could not be backed up ({rename_exc}). Starting fresh.",
                    file=sys.stderr,
                )
    return {"version": 1, "repos": {}}


def _save_manifest(manifest: dict) -> None:
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    from graphify.paths import write_json_atomic
    write_json_atomic(_GLOBAL_MANIFEST, manifest, indent=2)


def _load_global_graph() -> nx.Graph:
    """Materialize the whole global graph as a NetworkX object.

    Costs memory proportional to the ENTIRE global store, so nothing on the
    write path uses it any more — :func:`global_add` and :func:`global_remove`
    stream instead (see :func:`_rewrite_global_streamed`). It survives for
    readers that genuinely need a graph object and for tests.
    """
    if _GLOBAL_GRAPH.exists():
        from graphify.security import check_graph_file_size_cap
        check_graph_file_size_cap(_GLOBAL_GRAPH)
        data = json.loads(_GLOBAL_GRAPH.read_text(encoding="utf-8"))
        if "links" not in data and "edges" in data:
            data = dict(data, links=data["edges"])
        try:
            return _jg.node_link_graph(data, edges="links")
        except TypeError:
            return _jg.node_link_graph(data)
    return nx.Graph()


# --------------------------------------------------------------------------
# Streaming slice rewrite
#
# The global graph is one node-link JSON holding every tracked repo. Updating a
# repo changes only that repo's slice — but the obvious implementation (load
# into NetworkX, prune the repo, merge, save) touches the whole store to change
# a fraction of it. Measured on a real 226-repo, 1.21M-node, 1.888GB global
# graph: refreshing one 26,668-node repo (2.2% of the nodes) peaked at ~5.0GB
# RSS plus ~0.9GB of swap and took 3m49s — 45x more data held in memory than the
# change required, which on a 7.6GB box means OOM kills and, once a memory guard
# is added to stop them, a store that simply stops being refreshed.
#
# Per-repo membership IS addressable in the file: prefix_graph_for_global stamps
# every node with ``repo`` and rewrites its id to ``<tag>::<local_id>``, so a
# repo's slice is exactly "the nodes whose ``repo`` equals the tag, plus the
# links incident to them". That makes the update a streaming edit: read the
# existing file element by element with the maker's bounded-memory node-link
# reader, copy every other repo's nodes and links through untouched, drop the
# target repo's, append the new slice, and atomically replace. Peak memory
# becomes O(the changed repo + the global external-label index), not O(store).
#
# The output is byte-for-byte what json.dump(node_link_data(G), indent=2) would
# have written, so the file stays interchangeable with the whole-load path.
# --------------------------------------------------------------------------

_ELEM_PAD = "    "  # array elements sit two levels in under json.dump(indent=2)


def _dump_element(elem) -> str:
    """One node/link object, rendered exactly as ``json.dump(..., indent=2)`` would
    render it inside a top-level array. Safe against the newline substitution
    because ``json.dumps`` never emits a raw newline inside a string value."""
    return _ELEM_PAD + json.dumps(elem, indent=2).replace("\n", "\n" + _ELEM_PAD)


def _write_node_link_stream(fh, *, directed, multigraph, graph, nodes, links) -> tuple[int, int]:
    """Write a node-link file from two iterators, holding one element at a time.

    Matches ``json.dump(node_link_data(G), indent=2)`` byte for byte, including
    NetworkX's key order (directed, multigraph, graph, nodes, links) and its
    ``[]`` rendering of an empty array. Returns (node count, link count).
    """
    fh.write("{\n")
    fh.write('  "directed": ' + json.dumps(bool(directed)) + ",\n")
    fh.write('  "multigraph": ' + json.dumps(bool(multigraph)) + ",\n")
    fh.write('  "graph": ' + json.dumps(graph, indent=2).replace("\n", "\n  ") + ",\n")
    counts = []
    for key, elements, tail in (("nodes", nodes, ",\n"), ("links", links, "\n")):
        fh.write('  "' + key + '": [')
        count = 0
        for elem in elements:
            fh.write("\n" if count == 0 else ",\n")
            fh.write(_dump_element(elem))
            count += 1
        fh.write("\n  ]" if count else "]")
        fh.write(tail)
        counts.append(count)
    fh.write("}")
    return counts[0], counts[1]


def _rewrite_global_streamed(
    repo_tag: str,
    new_nodes: "list[dict] | None" = None,
    new_links: "list[dict] | None" = None,
) -> tuple[int, int]:
    """Replace ``repo_tag``'s slice of the global graph without loading the file.

    ``new_nodes``/``new_links`` are node-link elements for the incoming slice,
    already prefixed by :func:`prefix_graph_for_global`; pass none to remove the
    repo. Returns (nodes_added, nodes_removed).

    Reproduces the whole-load path's semantics exactly, including the
    external-library dedup: a node with no ``source_file`` whose ``label``
    already exists on a global external node is not re-added — its edges are
    rewired onto the existing node instead (``remap`` below), which is what
    ``G.add_node``/``G.add_edge`` did when both graphs were in memory.
    """
    from graphify.exporters.node_link_stream import scan_node_link, iter_node_link_array
    from graphify.paths import _atomic_replace

    new_nodes = new_nodes or []
    new_links = new_links or []

    # The global file is streamed, never materialized, so the graph-size cap
    # that guards whole-file loads deliberately does NOT apply to it here: the
    # memory this function uses is set by the incoming slice, not by the store.
    scan = scan_node_link(_GLOBAL_GRAPH) if _GLOBAL_GRAPH.exists() else None
    # A missing file means composing into a fresh nx.Graph(): always undirected,
    # non-multi, no graph attrs — matching what the whole-load path produced.
    directed = scan.directed if scan else False
    multigraph = scan.multigraph if scan else False
    graph_attrs = scan.graph if scan else {}

    def _pair(u, v):
        """Canonical key for one edge — orientation-insensitive when undirected,
        so a rewired edge matches the existing one exactly as NetworkX would."""
        if directed or u <= v:
            return (u, v)
        return (v, u)

    # Pass 1 over the existing nodes: which ids this repo owns (they go away,
    # and every link touching them goes with them) and the global external-label
    # index the incoming slice dedups against. Both are bounded by the repo and
    # by the number of external nodes, not by the store.
    removed_ids: set = set()
    external_labels: dict = {}
    external_pos: dict = {}
    if scan is not None and scan.nodes_offset is not None:
        for node in iter_node_link_array(_GLOBAL_GRAPH, scan.nodes_offset):
            if node.get("repo") == repo_tag:
                removed_ids.add(node.get("id"))
                continue
            if not node.get("source_file") and node.get("label"):
                # Last one wins, as the dict comprehension it replaces did.
                external_labels[node["label"]] = node.get("id")
                external_pos.setdefault(node.get("id"), len(external_pos))

    remap = {
        node["id"]: external_labels[node["label"]]
        for node in new_nodes
        if not node.get("source_file") and node.get("label") in external_labels
    }

    def _order_key(node_id):
        # Surviving global nodes keep their place; the incoming slice is
        # appended after all of them, so it always sorts last.
        pos = external_pos.get(node_id)
        return (0, pos) if pos is not None else (1, 0)

    # Rewire the incoming links through ``remap``, drop the self-loops that
    # rewiring can create, and collapse duplicates the way ``add_edge`` did:
    # first occurrence fixes the position, last occurrence wins the attributes.
    merged_links: dict = {}
    for link in new_links:
        u = remap.get(link.get("source"), link.get("source"))
        v = remap.get(link.get("target"), link.get("target"))
        if u == v:
            continue
        if _order_key(v) < _order_key(u):
            u, v = v, u
        merged_links[_pair(u, v)] = {**link, "source": u, "target": v}

    def _nodes_out():
        if scan is not None and scan.nodes_offset is not None:
            for node in iter_node_link_array(_GLOBAL_GRAPH, scan.nodes_offset):
                if node.get("repo") != repo_tag:
                    yield node
        for node in new_nodes:
            if node["id"] not in remap:
                yield node

    def _links_out():
        if scan is not None and scan.edge_array_offset is not None:
            for link in iter_node_link_array(_GLOBAL_GRAPH, scan.edge_array_offset):
                source, target = link.get("source"), link.get("target")
                if source in removed_ids or target in removed_ids:
                    continue
                # A rewired incoming link can land on an edge that already
                # exists between two surviving nodes; ``add_edge`` overwrote its
                # attributes, so the old copy must not be emitted twice.
                if _pair(source, target) in merged_links:
                    continue
                yield link
        yield from merged_links.values()

    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_replace(
        _GLOBAL_GRAPH,
        lambda fh: _write_node_link_stream(
            fh,
            directed=directed,
            multigraph=multigraph,
            graph=graph_attrs,
            nodes=_nodes_out(),
            links=_links_out(),
        ),
        # Rebuilding this file costs minutes and it is the only copy of the
        # cross-repo map, so pay one device flush before the rename.
        fsync=True,
    )
    return len(new_nodes) - len(remap), len(removed_ids)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def global_add(source_path: Path, repo_tag: str) -> dict:
    """Add or update a project graph in the global graph.

    Returns a summary dict with keys: repo_tag, nodes_added, nodes_removed, skipped.
    Skipped=True means the source graph hasn't changed since last add.
    """
    from graphify.build import prefix_graph_for_global

    if not source_path.exists():
        raise FileNotFoundError(f"graph not found: {source_path}")

    manifest = _load_manifest()
    src_hash = _file_hash(source_path)

    existing = manifest["repos"].get(repo_tag, {})
    existing_path = existing.get("source_path", "")
    if existing_path and existing_path != str(source_path.resolve()):
        print(
            f"[graphify global] warning: repo tag '{repo_tag}' previously pointed to "
            f"{existing_path!r}, now updating to {str(source_path.resolve())!r}. "
            f"Use --as <tag> to give it a different name.",
            file=sys.stderr,
        )
    if existing.get("source_hash") == src_hash:
        return {"repo_tag": repo_tag, "nodes_added": 0, "nodes_removed": 0, "skipped": True}

    # Load source graph
    from graphify.security import check_graph_file_size_cap
    check_graph_file_size_cap(source_path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if "links" not in data and "edges" in data:
        data = dict(data, links=data["edges"])
    try:
        src_G = _jg.node_link_graph(data, edges="links")
    except TypeError:
        src_G = _jg.node_link_graph(data)

    # Prefix IDs for cross-project isolation
    prefixed = prefix_graph_for_global(src_G, repo_tag)
    edge_count = prefixed.number_of_edges()

    # Flatten the incoming slice to node-link elements. This is one repo, so it
    # is the only graph-sized thing this function ever holds; the global store
    # it merges into is streamed through element by element instead of loaded.
    try:
        new_data = _jg.node_link_data(prefixed, edges="links")
    except TypeError:
        new_data = _jg.node_link_data(prefixed)
    del data, src_G, prefixed
    new_nodes = new_data.get("nodes") or []
    new_links = new_data.get("links") or new_data.get("edges") or []

    added, removed = _rewrite_global_streamed(repo_tag, new_nodes, new_links)

    manifest["repos"][repo_tag] = {
        "added_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path.resolve()),
        "node_count": added,
        "edge_count": edge_count,
        "source_hash": src_hash,
    }
    _save_manifest(manifest)

    return {"repo_tag": repo_tag, "nodes_added": added, "nodes_removed": removed, "skipped": False}


def global_remove(repo_tag: str) -> int:
    """Remove all nodes for repo_tag from the global graph. Returns count removed."""
    manifest = _load_manifest()
    if repo_tag not in manifest["repos"]:
        raise KeyError(f"repo '{repo_tag}' not in global graph")

    _, removed = _rewrite_global_streamed(repo_tag)

    del manifest["repos"][repo_tag]
    _save_manifest(manifest)
    return removed


def global_list() -> dict:
    """Return the manifest repos dict."""
    return _load_manifest().get("repos", {})


def global_path() -> Path:
    return _GLOBAL_GRAPH

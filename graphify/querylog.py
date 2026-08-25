"""Query logging for graphify — append-only JSONL, fail-silent."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_NODES_RE = re.compile(r"(\d+)\s+nodes?\s+found")


def _log_path() -> Path | None:
    # Opt-in only (#1797). The log records every query/path/explain question and
    # corpus path (and full responses if GRAPHIFY_QUERY_LOG_RESPONSES) in a
    # plaintext file under ~/.cache — outside any repo's .gitignore/retention. A
    # default-on record of proprietary queries contradicts graphify's on-device,
    # no-telemetry posture, so it is OFF unless explicitly enabled:
    #   GRAPHIFY_QUERY_LOG=<path>   log to that path, or
    #   GRAPHIFY_QUERY_LOG_ENABLE=1 log to ~/.cache/graphify-queries.log.
    # GRAPHIFY_QUERY_LOG_DISABLE=1 still forces it off (back-compat, wins).
    if os.environ.get("GRAPHIFY_QUERY_LOG_DISABLE", "").lower() in ("1", "true", "yes"):
        return None
    override = os.environ.get("GRAPHIFY_QUERY_LOG", "").strip()
    if override:
        return Path(override).expanduser()
    if os.environ.get("GRAPHIFY_QUERY_LOG_ENABLE", "").lower() in ("1", "true", "yes"):
        return Path.home() / ".cache" / "graphify-queries.log"
    return None


def _log_responses() -> bool:
    return os.environ.get("GRAPHIFY_QUERY_LOG_RESPONSES", "").lower() in ("1", "true", "yes")


def _max_records() -> int | None:
    # Opt-in rotation bound. Unset/invalid/non-positive = no rotation (today's
    # behavior: the log grows without bound).
    raw = os.environ.get("GRAPHIFY_QUERY_LOG_MAX_RECORDS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _rotate(path: Path, max_records: int) -> None:
    """Keep the newest max_records lines; move the rest to a sibling archive.

    Records are never deleted: the oldest lines beyond the bound are APPENDED
    to <stem>.archive.jsonl next to the log, then the live log is rewritten
    atomically (tempfile in the same directory + os.replace). Lines move raw,
    so archive + live is always byte-identical to what was logged.
    """
    with path.open("rb") as fh:
        lines = fh.readlines()
    cut = len(lines) - max_records
    if cut <= 0:
        return
    archive = path.with_name(path.stem + ".archive.jsonl")
    with archive.open("ab") as fh:
        fh.writelines(lines[:cut])
        fh.flush()
        os.fsync(fh.fileno())
    # Re-read the tail so records appended since the first read survive the
    # rewrite (narrows the lost-append window to the final rename).
    with path.open("rb") as fh:
        keep = fh.readlines()[cut:]
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.writelines(keep)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def nodes_from_result(result: str) -> int | None:
    m = _NODES_RE.search(result or "")
    return int(m.group(1)) if m else None


def log_query(
    *,
    kind: str,
    question: str,
    corpus: str,
    result: str | None = None,
    nodes_returned: int | None = None,
    duration_ms: float | None = None,
    **extra: Any,
) -> None:
    """Append one JSONL record to the query log. Never raises."""
    try:
        path = _log_path()
        if path is None:
            return
        if nodes_returned is None and result is not None:
            nodes_returned = nodes_from_result(result)
        rec: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "question": question,
            "corpus": corpus,
            "nodes_returned": nodes_returned,
        }
        if result is not None:
            rec["result_chars"] = len(result)
        if duration_ms is not None:
            rec["duration_ms"] = round(duration_ms, 3)
        for k, v in extra.items():
            if v is not None:
                rec[k] = v
        if result is not None and _log_responses():
            rec["response"] = result
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        max_records = _max_records()
        if max_records is not None:
            _rotate(path, max_records)
    except Exception:
        pass

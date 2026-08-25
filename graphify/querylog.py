"""Query logging for graphify — append-only JSONL, fail-silent."""
from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX (Windows): degrade to unlocked
    fcntl = None  # type: ignore[assignment]

_NODES_RE = re.compile(r"(\d+)\s+nodes?\s+found")

# Streamed-copy buffer for rotation; also the memory scale of a rotation.
_ROTATE_CHUNK = 64 * 1024


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


@contextlib.contextmanager
def _locked(path: Path, *, exclusive: bool):
    """Advisory flock on a sidecar lockfile (``<log name>.lock``), fail-silent.

    Appenders take the SHARED lock, so they never wait on each other (the
    kernel already serializes append-mode writes); only rotation takes the
    EXCLUSIVE lock. With both sides locking, no append can land between
    rotation's read and its os.replace and vanish with the replaced file.
    The sidecar — not the log itself — is locked because rotation changes the
    log's inode: a lock held on the replaced inode would exclude nobody.

    Fail-silent like the rest of this module: without fcntl (non-POSIX) or if
    the lockfile cannot be opened, yields unlocked and the caller proceeds
    with the pre-lock behavior (rotation's tail re-read still narrows the
    loss window). The lockfile is never unlinked — deleting a lockfile that
    another process may already hold open reopens the race it exists to close.
    """
    fh = None
    if fcntl is not None:
        try:
            fh = open(path.with_name(path.name + ".lock"), "ab")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError:
            if fh is not None:
                with contextlib.suppress(OSError):
                    fh.close()
            fh = None
    try:
        yield
    finally:
        if fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                fh.close()


def _copy_bytes(src, dst, remaining: int | None = None) -> None:
    """Chunked raw copy of ``remaining`` bytes (None = to EOF). Bytes move
    untouched, so what lands is byte-identical to what was read."""
    while remaining is None or remaining > 0:
        want = _ROTATE_CHUNK if remaining is None else min(_ROTATE_CHUNK, remaining)
        buf = src.read(want)
        if not buf:
            return
        dst.write(buf)
        if remaining is not None:
            remaining -= len(buf)


def _rotate(path: Path, max_records: int) -> None:
    """Keep the newest max_records lines; move the rest to a sibling archive.

    Records are never deleted: the oldest lines beyond the bound are APPENDED
    to <stem>.archive.jsonl next to the log, then the live log is rewritten
    atomically (tempfile in the same directory + os.replace). Lines move raw,
    so archive + live is always byte-identical to what was logged.

    Runs under the exclusive sidecar flock (see _locked): appenders hold the
    shared lock around their writes, so no record can slip in between the
    read and the os.replace and be lost. The log is STREAMED, never held in
    memory: one line-iteration pass records the byte offsets of the newest
    max_records line starts (a deque of ints, bounded by the keep-count, not
    the file size), then the head [0, cut) is chunk-copied into the archive
    and the tail [cut, EOF] chunk-copied into the tempfile. The tail copy
    re-reads the file after the archive append, so an unlocked (non-POSIX)
    run keeps the old narrowed-window behavior; archive-first ordering means
    a crash can duplicate a line into the archive but never lose one.
    """
    with _locked(path, exclusive=True):
        count = 0
        pos = 0
        offsets: deque[int] = deque(maxlen=max_records)
        with path.open("rb") as fh:
            for line in fh:
                offsets.append(pos)
                pos += len(line)
                count += 1
        if count <= max_records:
            return
        cut = offsets[0]  # byte offset where the kept tail starts
        archive = path.with_name(path.stem + ".archive.jsonl")
        with path.open("rb") as src, archive.open("ab") as dst:
            _copy_bytes(src, dst, remaining=cut)
            dst.flush()
            os.fsync(dst.fileno())
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with path.open("rb") as src, os.fdopen(fd, "wb") as dst:
                src.seek(cut)
                _copy_bytes(src, dst)  # to current EOF: re-read after the archive append
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
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
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        # Shared lock just for the write: appenders run concurrently with each
        # other, but never overlap a rotation's read+replace window.
        with _locked(path, exclusive=False):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        max_records = _max_records()
        if max_records is not None:
            _rotate(path, max_records)
    except Exception:
        pass

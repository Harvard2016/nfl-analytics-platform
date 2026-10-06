"""Job state in SQLite and private per-job storage. No path here is ever built from user-supplied text."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..shared import config

STATES = ("uploading", "queued", "validating", "extracting", "awaiting_review", "inferring", "rendering", "complete", "failed", "cancelled")
ACTIVE = ("uploading", "validating", "extracting", "inferring", "rendering")
TERMINAL = ("complete", "failed", "cancelled")
RETENTION_HOURS = float(os.environ.get("GRIDIRON_RETENTION_HOURS", "24"))


def now() -> datetime:
    return datetime.now(UTC)


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get("GRIDIRON_JOBS_DIR") or config.ROOT / "data" / "jobs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "jobs.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""create table if not exists jobs (id text primary key, token_hash text not null, module text not null, mode text not null, state text not null,
            stage_note text, progress real, options text not null, input_file text, warnings text not null default '[]', error text, cancel integer not null default 0,
            assets text not null default '{}', created_at text not null, updated_at text not null, expires_at text not null)""")
        self.db.commit()

    # ---- lifecycle
    def create(self, module: str, mode: str, options: dict, suffix: str) -> tuple[str, str, Path]:
        jid, token = secrets.token_hex(12), secrets.token_urlsafe(32)
        d = self.root / jid
        (d / "media").mkdir(parents=True)
        t = now()
        with self.lock:
            self.db.execute("insert into jobs (id, token_hash, module, mode, state, options, input_file, created_at, updated_at, expires_at) values (?,?,?,?,?,?,?,?,?,?)",
                            (jid, hashlib.sha256(token.encode()).hexdigest(), module, mode, "uploading", json.dumps(options), f"input{suffix}", t.isoformat(), t.isoformat(),
                             (t + timedelta(hours=RETENTION_HOURS)).isoformat()))
            self.db.commit()
        return jid, token, d / f"input{suffix}"

    def ready(self, jid: str) -> None:
        """The upload is fully on disk: only now may the worker pick the job up."""
        self.update(jid, state="queued")

    def get(self, jid: str) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute("select * from jobs where id = ?", (jid,)).fetchone()

    def authorized(self, jid: str, token: str | None) -> sqlite3.Row | None:
        """The job row only if the token matches. A wrong token and an unknown id are indistinguishable to the caller."""
        row = self.get(jid)
        if row is None or not token or not secrets.compare_digest(row["token_hash"], hashlib.sha256(token.encode()).hexdigest()):
            return None
        return row

    def update(self, jid: str, **kw) -> None:
        for k in ("warnings", "assets"):
            if k in kw:
                kw[k] = json.dumps(kw[k])
        kw["updated_at"] = now().isoformat()
        with self.lock:
            self.db.execute(f"update jobs set {', '.join(f'{k} = ?' for k in kw)} where id = ?", (*kw.values(), jid))
            self.db.commit()

    def next_queued(self) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute("select * from jobs where state = 'queued' and cancel = 0 order by created_at limit 1").fetchone()

    def cancelled(self, jid: str) -> bool:
        row = self.get(jid)
        return row is None or bool(row["cancel"])

    def dir(self, jid: str) -> Path:
        return self.root / jid

    def delete(self, jid: str) -> None:
        shutil.rmtree(self.root / jid, ignore_errors=True)
        with self.lock:
            self.db.execute("delete from jobs where id = ?", (jid,))
            self.db.commit()

    # ---- housekeeping
    def recover(self) -> int:
        """Jobs that were mid-run when the service stopped cannot be resumed: mark them failed instead of leaving them 'running' forever."""
        with self.lock:
            n = self.db.execute(f"update jobs set state = 'failed', error = 'Interrupted: the service restarted while this job was running. Submit it again.', updated_at = ? "
                                f"where state in ({','.join('?' * len(ACTIVE))})", (now().isoformat(), *ACTIVE)).rowcount
            self.db.commit()
        return n

    def sweep(self) -> int:
        with self.lock:
            old = [r["id"] for r in self.db.execute("select id from jobs where expires_at < ? and state in ('complete','failed','cancelled')", (now().isoformat(),))]
        for jid in old:
            self.delete(jid)
        return len(old)

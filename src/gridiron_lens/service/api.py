"""HTTP API for the upload lab. Loopback by default; CORS allows only the configured local frontends.

    PYTHONPATH=src .venv/bin/python -m gridiron_lens.service.api          # http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..coverage import upload as cov_upload
from ..highlights import bundle as hl_bundle
from . import media
from .store import RETENTION_HOURS, TERMINAL, Store
from .worker import LIMITS, Worker

ORIGINS = [o for o in os.environ.get("GRIDIRON_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3111,http://127.0.0.1:3111").split(",") if o]
MEDIA_TYPES = {".mp4": "video/mp4", ".m4a": "audio/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska", ".wav": "audio/wav", ".mp3": "audio/mpeg"}
SUFFIX_OK = re.compile(r"^\.[a-z0-9]{1,5}$")


def capabilities() -> dict:
    ff = media.available()
    bundle_ok = (hl_bundle.DIR / "manifest.json").exists()
    return {
        "service": "gridiron-lens-inference", "api": "v1", "hosting": "local (loopback)", "retention_hours": RETENTION_HOURS,
        "privacy": "Files stay on the machine running this service, are reachable only with the job's access token, are never used for training and are deleted on request or after the retention period.",
        "modules": {
            "coverage": {"kind": "tracking classification", "ready": cov_upload.MODEL_FILE.exists(), "model": "v2 temporal (seed 42)", "input": cov_upload.DOCS, "limits": LIMITS["coverage"],
                         "modes": {"release-compatible": {"ready": cov_upload.MODEL_FILE.exists()}, "broader": {"ready": cov_upload.MODEL_FILE.exists(), "experimental": True}}},
            "highlights": {"kind": "highlight ranking", "limits": LIMITS["highlights"], "modes": {
                "loudness_baseline": {"ready": ff, "reason": None if ff else "ffmpeg and ffprobe are not installed"},
                "trained_multimodal": {"ready": False, "bundle_present": bundle_ok,
                                       "reason": "The H3 bundle reproduces cached scores, but the CLIP, SlowFast and PANN extractors that produce its inputs are not installed or parity-checked here. Without them the trained model cannot score a new file."},
                "commentary_experimental": {"ready": False, "reason": "No local speech-to-text model is installed."}}},
            "video_coverage": {"kind": "experimental video coverage", "ready": False,
                               "reason": "Player detection, tracking and the review tools are not built. Field calibration has been measured on labelled helmet positions from real sideline and end-zone video, but no clip can be processed end to end."},
        },
    }


def create_app(store: Store | None = None, start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="Gridiron Lens local inference", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type", "Range"],
                       expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"])
    st = store or Store()
    st.recover()
    st.sweep()
    app.state.store = st
    if start_worker:
        app.state.worker = Worker(st)
        app.state.worker.start()

    def auth(jid: str, authorization: str | None, token: str | None):
        tok = token or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        row = st.authorized(jid, tok)
        if row is None:
            raise HTTPException(404, "No such job for this access token.")
        return row

    def public(row) -> dict:
        return {"id": row["id"], "module": row["module"], "mode": row["mode"], "state": row["state"], "stage": row["stage_note"], "progress": row["progress"],
                "progress_note": "Only stages are reported; no percentage is measured." if row["progress"] is None else None,
                "warnings": json.loads(row["warnings"]), "error": row["error"], "result_available": row["state"] == "complete",
                "created_at": row["created_at"], "updated_at": row["updated_at"], "expires_at": row["expires_at"], "assets": sorted(json.loads(row["assets"]))}

    @app.get("/v1/capabilities")
    def caps():
        return capabilities()

    @app.get("/v1/templates/coverage.csv")
    def template():
        return Response(cov_upload.TEMPLATE, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="coverage_upload_template.csv"'})

    @app.post("/v1/jobs", status_code=202)
    def submit(request: Request, module: str = Form(...), mode: str = Form("default"), options: str = Form("{}"), file: UploadFile = File(...)):  # noqa: B008
        caps_ = capabilities()["modules"]
        if module not in ("coverage", "highlights"):
            raise HTTPException(422, "module must be coverage or highlights.")
        mode = {"coverage": "release-compatible", "highlights": "loudness_baseline"}[module] if mode == "default" else mode
        m = caps_[module]["modes"].get(mode)
        if m is None:
            raise HTTPException(422, f"Unknown mode for {module}: {mode}.")
        if not m["ready"]:
            raise HTTPException(409, f"Mode {mode} is not available: {m.get('reason')}")
        try:
            opts = json.loads(options)
            assert isinstance(opts, dict)
        except (ValueError, AssertionError):
            raise HTTPException(422, "options must be a JSON object.") from None
        limit = LIMITS[module]["max_bytes"]
        declared = request.headers.get("content-length")
        if declared and int(declared) > limit + 2**20:
            raise HTTPException(413, f"The file is larger than the {limit // 2**20} MB limit for {module} uploads.")
        suffix = Path(file.filename or "").suffix.lower()
        if module == "coverage" and suffix not in LIMITS["coverage"]["formats"]:
            raise HTTPException(422, "Tracking uploads must be .csv or .json.")
        suffix = suffix if SUFFIX_OK.match(suffix) else ".bin"                # the stored name is generated; only a short sanitized extension is kept
        jid, token, dest = st.create(module, mode, opts | {"mode": mode}, suffix)
        size = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    out.close()
                    st.delete(jid)
                    raise HTTPException(413, f"The file is larger than the {limit // 2**20} MB limit for {module} uploads.")
                out.write(chunk)
        if size == 0:
            st.delete(jid)
            raise HTTPException(422, "The file is empty.")
        return {"id": jid, "access_token": token, "state": "queued", "status_url": f"/v1/jobs/{jid}", "retention_hours": RETENTION_HOURS}

    @app.get("/v1/jobs/{jid}")
    def status(jid: str, authorization: str | None = Header(None), token: str | None = Query(None)):
        return public(auth(jid, authorization, token))

    @app.post("/v1/jobs/{jid}/cancel")
    def cancel(jid: str, authorization: str | None = Header(None), token: str | None = Query(None)):
        row = auth(jid, authorization, token)
        if row["state"] in TERMINAL:
            return public(row) | {"note": "The job had already finished; nothing was cancelled."}
        st.update(jid, cancel=1, **({"state": "cancelled"} if row["state"] == "queued" else {}))
        return public(st.get(jid)) | {"note": "Cancellation requested. A running stage stops at its next check."}

    @app.delete("/v1/jobs/{jid}")
    def delete(jid: str, authorization: str | None = Header(None), token: str | None = Query(None)):
        row = auth(jid, authorization, token)
        if row["state"] not in TERMINAL and row["state"] != "queued":
            st.update(jid, cancel=1)
            return JSONResponse({"deleted": False, "note": "The job is running. Cancellation was requested; delete again once it has stopped."}, status_code=409)
        st.delete(jid)
        return {"deleted": True}

    @app.get("/v1/jobs/{jid}/result")
    def result(jid: str, authorization: str | None = Header(None), token: str | None = Query(None)):
        row = auth(jid, authorization, token)
        if row["state"] != "complete":
            raise HTTPException(409, f"No result: the job is {row['state']}.")
        return JSONResponse(json.loads((st.dir(jid) / "result.json").read_text()))

    @app.get("/v1/jobs/{jid}/media/{asset}")
    def media_file(jid: str, asset: str, range_: str | None = Header(None, alias="Range"), authorization: str | None = Header(None), token: str | None = Query(None)):
        row = auth(jid, authorization, token)
        names = json.loads(row["assets"])
        if asset not in names:                                                 # lookup in the job's own manifest: the URL text is never joined to a path
            raise HTTPException(404, "No such asset for this job.")
        path = st.dir(jid) / (row["input_file"] if asset == "source" else Path("media") / names[asset])
        size = path.stat().st_size
        ctype = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        start, end = 0, size - 1
        if range_ and (m := re.match(r"bytes=(\d*)-(\d*)$", range_.strip())):
            if m.group(1):
                start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            end = min(end, size - 1)

        def body():
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(1 << 16, left))):
                    left -= len(chunk)
                    yield chunk

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "private, no-store"}
        if range_:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(body(), status_code=206 if range_ else 200, media_type=ctype, headers=headers)

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("GRIDIRON_API_HOST", "127.0.0.1"), port=int(os.environ.get("GRIDIRON_API_PORT", "8765")), log_level="warning", access_log=False)

"""Upload API: rejection, ownership isolation, cancellation, restart recovery, deletion and safe media paths.

Media here is a generated test tone and colour bars (ffmpeg lavfi): it checks mechanics only and is not football footage.
"""
import itertools
import shutil
import subprocess
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gridiron_lens.coverage import upload as U
from gridiron_lens.service import media
from gridiron_lens.service.api import create_app
from gridiron_lens.service.store import Store
from tests.test_coverage_upload import _csv, _rows

needs_model = pytest.mark.skipif(not U.MODEL_FILE.exists(), reason="local coverage model not present")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def client(tmp_path):
    app = create_app(Store(tmp_path / "jobs"))
    with TestClient(app) as c:
        c.store = app.state.store
        yield c
    app.state.worker.stop.set()


def _wait(c, jid, tok, states=("complete", "failed", "cancelled"), timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get(f"/v1/jobs/{jid}", headers={"Authorization": f"Bearer {tok}"}).json()
        if s["state"] in states:
            return s
        time.sleep(0.2)
    raise AssertionError(f"job stuck in {s['state']}")


def _submit(c, module, name, data, **form):
    return c.post("/v1/jobs", data={"module": module, **form}, files={"file": (name, data)})


def test_capabilities_hide_paths_and_state_what_is_not_ready(client):
    caps = client.get("/v1/capabilities").json()
    text = str(caps)
    assert "/Users/" not in text and "/home/" not in text
    assert caps["modules"]["highlights"]["modes"]["trained_multimodal"]["ready"] is False
    assert caps["modules"]["video_coverage"]["ready"] is False


def test_input_rejection(client):
    assert _submit(client, "nope", "a.csv", b"x").status_code == 422
    assert _submit(client, "coverage", "a.exe", b"x").status_code == 422
    assert _submit(client, "coverage", "a.csv", b"").status_code == 422
    assert _submit(client, "coverage", "a.csv", b"x" * (3 * 2**20)).status_code == 413
    assert _submit(client, "highlights", "a.mp4", b"x", mode="trained_multimodal").status_code == 409        # never accepts a file for a mode that cannot run
    assert _submit(client, "coverage", "a.csv", b"x", options="[1]").status_code == 422
    assert list(client.store.root.glob("*/input*")) == []                                                   # rejected uploads leave nothing behind


@needs_model
def test_tracking_job_completes_and_bad_tracking_fails_with_reasons(client):
    j = _submit(client, "coverage", "play.csv", _csv(_rows()).encode()).json()
    s = _wait(client, j["id"], j["access_token"])
    assert s["state"] == "complete" and s["progress"] is None
    r = client.get(f"/v1/jobs/{j['id']}/result", headers={"Authorization": f"Bearer {j['access_token']}"}).json()
    assert r["kind"] == "tracking classification" and r["horizons"]["post_1_5s"]["available"] and len(r["playback"]["entities"]) == 7
    assert "accuracy" not in r                                                                               # no label, no accuracy badge
    bad = _submit(client, "coverage", "play.csv", _csv([x | {"x": 999} for x in _rows()]).encode()).json()
    f = _wait(client, bad["id"], bad["access_token"])
    assert f["state"] == "failed" and "yards" in f["error"]


@needs_model
def test_jobs_are_isolated_by_token_and_can_be_deleted(client):
    a = _submit(client, "coverage", "a.csv", _csv(_rows()).encode()).json()
    b = _submit(client, "coverage", "b.csv", _csv(_rows(frames=8)).encode()).json()
    _wait(client, a["id"], a["access_token"]), _wait(client, b["id"], b["access_token"])
    for url in (f"/v1/jobs/{a['id']}", f"/v1/jobs/{a['id']}/result", f"/v1/jobs/{a['id']}/media/source"):
        assert client.get(url, headers={"Authorization": f"Bearer {b['access_token']}"}).status_code == 404   # another job's token
        assert client.get(url).status_code == 404                                                            # no token
    assert client.delete(f"/v1/jobs/{a['id']}", headers={"Authorization": f"Bearer {b['access_token']}"}).status_code == 404
    assert client.delete(f"/v1/jobs/{a['id']}", headers={"Authorization": f"Bearer {a['access_token']}"}).json() == {"deleted": True}
    assert not (client.store.root / a["id"]).exists()
    assert client.get(f"/v1/jobs/{a['id']}", headers={"Authorization": f"Bearer {a['access_token']}"}).status_code == 404
    assert client.get(f"/v1/jobs/{b['id']}", headers={"Authorization": f"Bearer {b['access_token']}"}).status_code == 200


def test_restart_marks_interrupted_jobs_failed_and_retention_sweeps(tmp_path):
    st = Store(tmp_path / "jobs")
    jid, tok, dest = st.create("coverage", "release-compatible", {}, ".csv")
    assert st.next_queued() is None                                                                         # a job whose upload is still being written is never picked up
    dest.write_text("x")
    st.ready(jid)
    assert st.next_queued()["id"] == jid
    st.update(jid, state="inferring")
    app = create_app(Store(tmp_path / "jobs"), start_worker=False)                                           # a new process finds the job mid-run
    with TestClient(app) as c:
        s = c.get(f"/v1/jobs/{jid}", headers={"Authorization": f"Bearer {tok}"}).json()
        assert s["state"] == "failed" and "restarted" in s["error"]
        app.state.store.update(jid, expires_at="2000-01-01T00:00:00+00:00")
        assert app.state.store.sweep() == 1 and not dest.exists()


def test_cancel_before_start_is_reported_as_cancelled(tmp_path):
    app = create_app(Store(tmp_path / "jobs"), start_worker=False)
    with TestClient(app) as c:
        j = _submit(c, "coverage", "a.csv", b"player_id,frame,side,x,y\n").json()
        h = {"Authorization": f"Bearer {j['access_token']}"}
        assert c.post(f"/v1/jobs/{j['id']}/cancel", headers=h).json()["state"] == "cancelled"
        assert c.get(f"/v1/jobs/{j['id']}/result", headers=h).status_code == 409                             # no fake success


@pytest.fixture(scope="module")
def tone(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "synthetic_tone.mp4"
    expr = "aevalsrc='0.02*sin(2*PI*220*t)+0.8*sin(2*PI*880*t)*gt(mod(t\\,15)\\,12)':s=16000:d=46"                # quiet tone with a loud burst every 15 s
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=46", "-f", "lavfi", "-i", expr,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


@needs_ffmpeg
def test_highlight_job_cuts_within_budget_and_serves_ranges_safely(client, tone):
    assert _submit(client, "highlights", "x.mp4", b"not a video at all" * 100).json()["id"]                 # accepted, then fails validation with a reason
    j = _submit(client, "highlights", "clip.mp4", tone.read_bytes(), options='{"reel_seconds": 12, "lead_s": 1, "tail_s": 1}').json()
    h = {"Authorization": f"Bearer {j['access_token']}"}
    s = _wait(client, j["id"], j["access_token"])
    assert s["state"] == "complete", s
    r = client.get(f"/v1/jobs/{j['id']}/result", headers=h).json()
    assert r["mode"].startswith("loudness baseline") and r["evidence_streams_used"] == ["audio loudness"] and r["event_types"].startswith("none")
    reel = r["reel"]
    assert reel["decoder_output_s"] <= 12 and reel["within_budget"]
    for k in ("ffprobe_duration_s", "ffprobe_video_s", "ffprobe_audio_s"):                                   # the duration rule, checked on the exported file itself
        assert 12 - 0.5 <= reel[k] <= 12, (k, reel[k])
    again = media.probe(client.store.dir(j["id"]) / "media" / reel["asset"])
    assert max(again["duration_s"], again["video_duration_s"], again["audio_duration_s"]) <= 12
    assert all(abs(c["ffprobe_duration_s"] - c["requested_s"]) < 0.25 for c in r["candidates"])
    assert any(11 <= c["candidate_moment_s"] % 15 <= 15 or c["candidate_moment_s"] % 15 <= 1 for c in r["candidates"])   # the loud bursts are found
    part = client.get(f"/v1/jobs/{j['id']}/media/{r['reel']['asset']}", headers=h | {"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-range"].startswith("bytes 0-99/")
    assert client.get(f"/v1/jobs/{j['id']}/media/{r['reel']['asset']}?token={j['access_token']}").status_code == 200
    for evil in ("..%2F..%2Fjobs.sqlite", "%2Fetc%2Fpasswd", "input.mp4", "..%5Cx"):
        assert client.get(f"/v1/jobs/{j['id']}/media/{evil}", headers=h).status_code == 404
    assert media.probe(client.store.dir(j["id"]) / "media" / r["reel"]["asset"])["has_video"]


@needs_ffmpeg
def test_overlapping_padding_does_not_play_twice_in_the_reel(client, tone):
    """Found on real footage: two picks 6 s apart with 4 s padding were cut separately and the reel ran 8 s over budget."""
    j = _submit(client, "highlights", "clip.mp4", tone.read_bytes(), options='{"reel_seconds": 30, "lead_s": 5, "tail_s": 5}').json()
    s = _wait(client, j["id"], j["access_token"])
    assert s["state"] == "complete", s
    r = client.get(f"/v1/jobs/{j['id']}/result", headers={"Authorization": f"Bearer {j['access_token']}"}).json()
    spans = sorted((c["clip_start_s"], c["clip_end_s"]) for c in r["candidates"])
    assert all(b1 < a2 for (_, b1), (a2, _) in itertools.pairwise(spans))                                       # rendered clips never overlap
    assert abs(sum(b - a for a, b in spans) - r["reel"]["decoder_output_s"]) < 1e-6
    assert r["reel"]["within_budget"] and max(r["reel"]["ffprobe_duration_s"], r["reel"]["ffprobe_video_s"], r["reel"]["ffprobe_audio_s"]) <= 30   # never over, however many cuts
    assert any(len(c["moments_s"]) > 1 for c in r["candidates"]) or len(spans) >= 2


@needs_ffmpeg
def test_unreadable_media_fails_with_a_reason(client):
    j = _submit(client, "highlights", "x.mp4", b"not a video at all" * 100).json()
    s = _wait(client, j["id"], j["access_token"])
    assert s["state"] == "failed" and "could not be read" in s["error"]

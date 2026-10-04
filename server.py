"""Experiment server: serves the viewer and runs brain+body experiments on request.

    C:/Users/70sou/.venvs/flies/Scripts/python.exe server.py      ->  http://localhost:8765

Experiments run one at a time on the GPU in a worker thread; results are saved
to runs/<id>.json so earlier runs can be reopened.
"""
import json
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
CACHE = RUNS / "_cache"
VIEWER = ROOT / "viewer"
PENDING = RUNS / "_queue"  # configs of queued/running jobs, so a restart resumes them
RUNS.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True); PENDING.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from flybrain import experiment as ex  # noqa: E402

jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()
cache_lock = threading.Lock()


def cached(name: str, build) -> Path:
    path = CACHE / f"{name}.json"
    with cache_lock:
        if not path.exists():
            path.write_text(json.dumps(build(), ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    return path


def worker():
    while True:
        jid = work.get()
        job = jobs[jid]
        job.update(state="running", started=time.time())

        def progress(frac, msg):
            job.update(progress=round(frac, 3), msg=msg)
        try:
            result = ex.run_experiment(job["config"], progress)
            result["id"] = jid
            result["created"] = job["created"]
            (RUNS / f"{jid}.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"),
                                                         default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                              encoding="utf8")
            cfg = job["config"]
            meta = dict(id=jid, created=job["created"], seconds=cfg.get("seconds"),
                        names=[v.get("name") for v in cfg.get("variants", [])],
                        eaten=[len(v["summary"]["eaten"]) for v in result["variants"]])
            (RUNS / f"{jid}.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf8")
            job.update(state="done", progress=1.0, msg="완료")
        except Exception as e:  # report to the UI instead of killing the worker
            traceback.print_exc()
            job.update(state="error", msg=str(e))
        (PENDING / f"{jid}.json").unlink(missing_ok=True)


def enqueue(jid: str, cfg: dict, created: float):
    jobs[jid] = dict(id=jid, state="queued", progress=0.0, msg="대기 중", created=created, config=cfg)
    work.put(jid)


def resume_pending():
    for p in sorted(PENDING.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf8"))
        enqueue(p.stem, d["config"], d["created"])
        print(f"resumed queued run {p.stem}")


def list_runs():
    out = []
    for p in sorted(RUNS.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf8")))
        except Exception:
            continue
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(VIEWER), **kw)

    def log_message(self, fmt, *args):  # keep the console quiet except errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/catalog":
            return self.send_file(cached("catalog", ex.catalog))
        if p == "/api/brainmap":
            return self.send_file(cached("brainmap", ex.brain_map))
        if p == "/api/body":
            return self.send_file(cached("body", ex.body_meshes))
        if p == "/api/runs":
            return self.send_json(list_runs())
        if p.startswith("/api/job/"):
            job = jobs.get(p.rsplit("/", 1)[1])
            if not job:
                return self.send_json({"error": "없는 작업이에요"}, 404)
            return self.send_json({k: v for k, v in job.items() if k != "config"} |
                                  {"queue": list(work.queue).index(p.rsplit("/", 1)[1]) + 1
                                   if job["state"] == "queued" else 0})
        if p.startswith("/runs/"):
            f = RUNS / Path(p).name
            if f.suffix == ".json" and not f.name.endswith(".meta.json") and f.exists():
                return self.send_file(f)
            return self.send_json({"error": "없는 실행 기록이에요"}, 404)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            cfg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            n = len(cfg.get("variants", []))
            if not 1 <= n <= 4:
                raise ValueError("변이(초파리)는 1~4마리까지 가능해요")
            if not 1 <= float(cfg.get("seconds", 0)) <= 30:
                raise ValueError("시간은 1~30초 사이로 정해 주세요")
            for v in cfg["variants"]:
                for m in v.get("manips", []) + (v.get("twin") or {}).get("manips", []):
                    if not ex.resolve(m["target"], m.get("side", "both")):
                        raise ValueError(f"'{m['target']}' ({m.get('side', 'both')}) 에 해당하는 뉴런이 없어요")
            for f in cfg.get("foods", []):
                if isinstance(f, dict) and f.get("taste", "설탕") != "설탕" and not ex.resolve(f["taste"], f.get("side", "both")):
                    raise ValueError(f"먹이 자극 '{f['taste']}' 에 해당하는 뉴런이 없어요")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        created = time.time()
        (PENDING / f"{jid}.json").write_text(json.dumps({"config": cfg, "created": created}, ensure_ascii=False),
                                             encoding="utf8")
        enqueue(jid, cfg, created)
        self.send_json({"id": jid})


if __name__ == "__main__":
    resume_pending()
    threading.Thread(target=worker, daemon=True).start()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"초파리 실험실: http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

"""スマイル商談メモ：ドライブの商談録音を文字起こしして、商談メモアプリに登録する。

GitHub Actions から定期実行する。音声・文字起こしの中身はログに出さない（公開リポジトリのため）。
必要な環境変数：APPS_URL（Apps Script の Web アプリURL）, INGEST_KEY（合言葉）, WORKER_URL, MODEL_DIR, VAD_MODEL（任意）
"""
import base64, json, os, re, subprocess, sys, tempfile, time, urllib.parse, urllib.request

APPS_URL = os.environ["APPS_URL"]
KEY = os.environ["INGEST_KEY"]
WORKER = os.environ.get("WORKER_URL", "https://smilecarz-shodan.smilecarz-jimu.workers.dev").rstrip("/")
MODEL_DIR = os.environ.get("MODEL_DIR", "models/sherpa-onnx-whisper-turbo")
TIME_BUDGET = int(os.environ.get("TIME_BUDGET_SEC", "2700"))  # 1回の実行で使う最大時間
LIST_ONLY = os.environ.get("LIST_ONLY") == "1"
T0 = time.time()


def http(url, data=None, headers=None, timeout=300):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def apps(params):
    q = urllib.parse.urlencode(dict(params, key=KEY))
    out = json.loads(http(APPS_URL + ("&" if "?" in APPS_URL else "?") + q))
    if "error" in out:
        raise RuntimeError("apps: " + out["error"])
    return out


def worker(path, payload=None):
    h = {"x-ingest-key": KEY, "content-type": "application/json", "user-agent": "smile-transcriber"}
    data = json.dumps(payload).encode() if payload is not None else None
    return json.loads(http(WORKER + path, data, h))


def pending_files():
    files = apps({"action": "list"})["files"]
    if not files:
        return []
    status = {}
    for i in range(0, len(files), 50):
        ids = ",".join(f["id"] for f in files[i:i + 50])
        status.update(worker("/api/ingest/status?file_ids=" + urllib.parse.quote(ids))["status"])
    todo = [f for f in files if status.get(f["id"]) not in ("done", "error")]
    todo.sort(key=lambda f: f["created"])
    return todo


def download(f, path):
    off = 0
    with open(path, "wb") as out:
        while True:
            part = apps({"action": "get", "id": f["id"], "offset": off})
            data = base64.b64decode(part["data"]) if part["data"] else b""
            if not data:
                break
            out.write(data)
            off += len(data)
            if off >= int(part["size"]):
                break
    return off


_rec = None


def recognizer():
    global _rec
    if _rec is None:
        import sherpa_onnx
        _rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=f"{MODEL_DIR}/turbo-encoder.int8.onnx",
            decoder=f"{MODEL_DIR}/turbo-decoder.int8.onnx",
            tokens=f"{MODEL_DIR}/turbo-tokens.txt",
            language="ja", task="transcribe", num_threads=os.cpu_count() or 2, tail_paddings=800)
    return _rec


REPEAT = re.compile(r"(.{1,8}?)(?:[、 ]?\1){3,}")
JUNK = ("ご視聴ありがとうございました", "おわり", "お疲れ様でした", "おやすみなさい")


def transcribe(src):
    import numpy as np, soundfile as sf
    wav = src + ".wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                    "-af", "highpass=f=80,dynaudnorm=f=150:g=15:p=0.9", wav], check=True)
    samples, sr = sf.read(wav, dtype="float32")
    rec = recognizer()
    W = 24 * 16000
    lines, prev = [], None
    for i in range(0, len(samples), W):
        a = samples[i:i + W]
        if len(a) < 16000 or float(np.sqrt(np.mean(a ** 2))) < 0.003:
            continue
        s = rec.create_stream()
        s.accept_waveform(16000, a)
        rec.decode_stream(s)
        t = REPEAT.sub(r"\1…", s.result.text.strip())[:500]
        if not t or t == prev or (len(t) < 16 and any(j in t for j in JUNK)):
            continue
        st = i / 16000
        lines.append(f"[{int(st // 60):02d}:{int(st % 60):02d}] {t}")
        prev = t
    return "\n".join(lines), int(len(samples) / 16000)


def main():
    todo = pending_files()
    print(f"new recordings: {len(todo)}")
    if LIST_ONLY or not todo:
        return
    done = 0
    for f in todo:
        if time.time() - T0 > TIME_BUDGET:
            print("time budget reached; the rest will be handled next run")
            break
        with tempfile.TemporaryDirectory() as d:
            ext = os.path.splitext(f["name"])[1] or ".m4a"
            path = os.path.join(d, "audio" + ext)
            try:
                download(f, path)
                text, dur = transcribe(path)
                res = worker("/api/ingest/result", {"file_id": f["id"], "filename": f["name"], "size": f["size"],
                                                     "created": f["created"], "transcript": text, "duration_sec": dur})
                done += 1
                print(f"recording {done}: ok ({dur // 60} min)")
            except Exception as e:  # 中身はログに出さない
                print(f"recording failed: {type(e).__name__}")
                try:
                    worker("/api/ingest/result", {"file_id": f["id"], "filename": f["name"], "size": f["size"],
                                                  "created": f["created"], "error": type(e).__name__})
                except Exception:
                    pass
    print(f"finished: {done}")


if __name__ == "__main__":
    main()

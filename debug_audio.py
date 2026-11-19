"""調査用：1件の録音について、音声の形式と区間ごとの音量・文字数だけをログに出す（中身は出さない）。"""
import json, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf
import transcribe as T

fid = os.environ["DEBUG_ID"]
f = next((x for x in T.apps({"action": "list"})["files"] if x["id"] == fid), None)
if not f:
    sys.exit("file not found in folder")
with tempfile.TemporaryDirectory() as d:
    src = os.path.join(d, "audio.m4a")
    T.download(f, src)
    print("downloaded bytes:", os.path.getsize(src), "listed size:", f["size"])
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,bit_rate:stream=codec_name,sample_rate,channels,duration,nb_frames",
                        "-of", "json", src], capture_output=True, text=True)
    print("ffprobe:", p.stdout.replace("\n", " "), p.stderr[:300])
    for label, af in (("raw", None), ("filtered", "highpass=f=80,dynaudnorm=f=150:g=15:p=0.9")):
        wav = os.path.join(d, label + ".wav")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000"] + (["-af", af] if af else []) + [wav]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(label, "ffmpeg stderr:", r.stderr[:300])
        s, _ = sf.read(wav, dtype="float32")
        print(label, "seconds:", round(len(s) / 16000))
        W = 60 * 16000
        print(label, "rms per minute:", [round(float(np.sqrt(np.mean(s[i:i + W] ** 2))), 4) for i in range(0, len(s), W)])
    s, _ = sf.read(os.path.join(d, "filtered.wav"), dtype="float32")
    rec = T.recognizer()
    W = 24 * 16000
    out = []
    for i in range(0, len(s), W):
        a = s[i:i + W]
        if len(a) < 16000:
            continue
        st = rec.create_stream(); st.accept_waveform(16000, a); rec.decode_stream(st)
        out.append((i // 16000, round(float(np.sqrt(np.mean(a ** 2))), 4), len(st.result.text.strip())))
    print("window(sec, rms, chars):", out)

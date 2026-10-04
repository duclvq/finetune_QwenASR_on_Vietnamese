"""Concurrent realtime-stream load test: N simulated users, each streaming test clips at 1x speed to the gateway.

Per level it reports final/partial latency (p50/p95/max), latency drift (second half vs first half), stream WER,
errors, gateway CPU and GPU utilisation/power sampled every 0.5 s (steady-state window only).
Usage: python realtime/load_test.py --users 8 16 32 --seconds 90 --out results/realtime_load.json
"""
import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time

import jiwer
import numpy as np
import psutil
import websockets

sys.path.insert(0, ".")
from eval_wer import normalize
from train_qwen_asr import DATA, PreparedAudio

GPU = "1"


def gpu_sampler(path):
    return subprocess.Popen(["nvidia-smi", "-i", GPU, "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm",
                             "--format=csv,noheader,nounits", "-lms", "500"], stdout=open(path, "w"))


def gateway_procs():
    """The uvicorn master (cmdline mentions gateway:app) and all its worker children."""
    out = []
    for p in psutil.process_iter(["cmdline"]):
        if "gateway:app" in " ".join(p.info["cmdline"] or []):
            out += [p] + p.children(recursive=True)
    return out


def pctl(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else None


def build_stream(ds, rng, seconds, gap):
    parts, refs, total = [np.zeros(int(0.5 * 16000), np.float32)], [], 0.5
    while total < seconds:
        i = int(rng.integers(len(ds)))
        a = ds[i]["audio"]
        g = np.zeros(int(rng.uniform(*gap) * 16000), np.float32)
        parts += [a, g]
        refs.append(ds[i]["text"])
        total += (len(a) + len(g)) / 16000
    return (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2"), refs


async def user(idx, url, pcm, start_delay, res):
    await asyncio.sleep(start_delay)
    ev = []
    try:
        async with websockets.connect(url, max_size=None) as ws:
            t0 = time.monotonic()

            async def rx():
                async for raw in ws:
                    e = json.loads(raw)
                    e["_t"] = time.monotonic() - t0
                    ev.append(e)

            task = asyncio.create_task(rx())
            for i in range(0, len(pcm), 1600):
                await asyncio.sleep(max(0, t0 + i / 16000 - time.monotonic()))
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[i:i + 1600].tobytes()).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.sleep(8)
            task.cancel()
    except Exception as e:  # connection errors count against the level
        res["errors"].append(repr(e)[:100])
    res["events"][idx] = ev


async def run_level(args, n, ds):
    rng = np.random.default_rng(1000 + n + args.seed_offset)
    streams = [build_stream(ds, rng, args.seconds, (0.6, 2.0)) for _ in range(n)]
    res = {"events": {}, "errors": []}
    gw = gateway_procs()
    for p in gw:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    smp = gpu_sampler(f"outputs/gpu_{n}{args.tag}.csv")
    t_start = time.monotonic()
    await asyncio.gather(*[user(i, args.url, pcm, i * (5.0 / n), res) for i, (pcm, _) in enumerate(streams)])
    wall = time.monotonic() - t_start
    gw_cpu = 0.0
    for p in gw:
        try:
            gw_cpu += p.cpu_percent(None)
        except psutil.Error:
            pass
    smp.terminate()
    rows = []
    for l in open(f"outputs/gpu_{n}{args.tag}.csv", errors="ignore"):
        try:
            v = list(map(float, l.strip().split(",")))
            if len(v) == 4:
                rows.append(v)
        except ValueError:
            pass
    steady = rows[int(len(rows) * 0.15): int(len(rows) * 0.85)]   # drop ramp-up and tail
    g = np.array(steady)
    fin_lat, fp_lat, early, late, hyps, refs = [], [], [], [], [], []
    for i, (pcm, r) in enumerate(streams):
        ev = res["events"].get(i, [])
        starts = {e["item_id"]: e for e in ev if e["type"] == "input_audio_buffer.speech_started"}
        finals = [e for e in ev if e["type"].endswith("completed")]
        firstp = {}
        for e in ev:
            if e["type"].endswith("partial"):
                firstp.setdefault(e["item_id"], e["_t"])
        for e in finals:
            lat = e["_t"] - e["audio_end_ms"] / 1000
            fin_lat.append(lat)
            (early if e["audio_end_ms"] / 1000 < len(pcm) / 32000 else late).append(lat)
        fp_lat += [firstp[i] - starts[i]["audio_start_ms"] / 1000 for i in firstp if i in starts]
        hyps.append(" ".join(e["transcript"] for e in finals))
        refs.append(" ".join(r))
        res["errors"] += [e["message"][:80] for e in ev if e["type"] == "error"]
    wer = 100 * jiwer.wer([normalize(x) for x in refs], [normalize(x) for x in hyps])
    return {"users": n, "wall_s": round(wall), "finals": len(fin_lat), "errors": len(res["errors"]),
            "error_samples": res["errors"][:2], "stream_wer": round(wer, 2),
            "final_latency_s": {"p50": pctl(fin_lat, 50), "p95": pctl(fin_lat, 95), "max": pctl(fin_lat, 100)},
            "drift_p50_first_vs_second_half_s": [pctl(early, 50), pctl(late, 50)],
            "first_partial_p50_s": pctl(fp_lat, 50),
            "gpu_util_pct": {"mean": round(g[:, 0].mean()), "p95": round(np.percentile(g[:, 0], 95))},
            "gpu_power_w": {"mean": round(g[:, 1].mean()), "max": round(g[:, 1].max())},
            "gpu_mem_gb": round(g[:, 2].max() / 1024, 1), "gateway_cpu_cores_busy": round(gw_cpu / 100 / 1.0, 2)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8200/v1/realtime?sample_rate=16000")
    ap.add_argument("--users", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--out", default="results/realtime_load.json")
    ap.add_argument("--seed_offset", type=int, default=0, help="use different clips when running several generator processes")
    ap.add_argument("--tag", default="", help="suffix for the GPU sample file when running several processes")
    args = ap.parse_args()
    ds = PreparedAudio(DATA + "/test", max_samples=300, seed=5)
    out = []
    for n in args.users:
        r = await run_level(args, n, ds)
        print(json.dumps(r), flush=True)
        out.append(r)
        json.dump(out, open(args.out, "w"), indent=1)
        await asyncio.sleep(5)

asyncio.run(main())

"""Stream test-set clips to the realtime gateway at real-time speed and measure latency and WER.

Builds one long stream from N ViMD test clips separated by silence, sends 100 ms frames paced by the wall clock,
and records every event. Usage: python realtime/test_stream.py [--clips 12] [--gap 1.5] [--speed 1.0]
"""
import argparse
import asyncio
import base64
import json
import sys
import time

import jiwer
import numpy as np
import websockets

sys.path.insert(0, ".")
from eval_wer import normalize
from train_qwen_asr import DATA, PreparedAudio


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8200/v1/realtime?sample_rate=16000")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--gap", type=float, default=1.5, help="silence between clips, seconds")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/realtime_stream.json")
    args = ap.parse_args()

    ds = PreparedAudio(DATA + "/test", max_samples=args.clips, seed=args.seed)
    gap = np.zeros(int(args.gap * 16000), np.float32)
    parts, bounds, pos = [gap], [], len(gap)
    for i in range(len(ds)):
        a = ds[i]["audio"]
        parts += [a, gap]
        bounds.append((pos / 16000, (pos + len(a)) / 16000))
        pos += len(a) + len(gap)
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2")
    dur = len(pcm) / 16000
    refs = [m["text"] for m in ds.meta]
    print(f"stream {dur:.0f}s, {len(ds)} clips, pacing x{args.speed}", flush=True)

    events = []
    async with websockets.connect(args.url, max_size=None) as ws:
        t0 = time.monotonic()

        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                ev["_t"] = (time.monotonic() - t0) * args.speed   # wall time mapped to stream time
                events.append(ev)

        rx = asyncio.create_task(receiver())
        frame = 1600  # 100 ms
        for i in range(0, len(pcm), frame):
            await asyncio.sleep(max(0, t0 + (i / 16000) / args.speed - time.monotonic()))
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + frame].tobytes()).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.sleep(6)  # let the last finals arrive
        rx.cancel()

    finals = [e for e in events if e["type"].endswith("transcription.completed")]
    partials = [e for e in events if e["type"].endswith("transcription.partial")]
    starts = {e["item_id"]: e for e in events if e["type"] == "input_audio_buffer.speech_started"}
    # final latency: arrival time minus the stream time at which that utterance's audio ended
    fin_lat = [e["_t"] - e["audio_end_ms"] / 1000 for e in finals]
    first_partial = {}
    for e in partials:
        first_partial.setdefault(e["item_id"], e["_t"])
    fp_lat = [first_partial[i] - starts[i]["audio_start_ms"] / 1000 for i in first_partial if i in starts]
    hyp = " ".join(e["transcript"] for e in finals)
    ref = " ".join(refs)
    wer = 100 * jiwer.wer(normalize(ref), normalize(hyp))
    pct = lambda v, p: round(float(np.percentile(v, p)), 2) if v else None
    res = {"stream_seconds": round(dur, 1), "clips": len(ds), "speech_started": len(starts), "finals": len(finals),
           "partials": len(partials), "stream_wer": round(wer, 2),
           "final_latency_s": {"p50": pct(fin_lat, 50), "p95": pct(fin_lat, 95), "max": pct(fin_lat, 100)},
           "first_partial_latency_s": {"p50": pct(fp_lat, 50), "p95": pct(fp_lat, 95)}}
    # VAD boundary check: does each clip's span overlap exactly one utterance's span?
    spans = [(e["audio_start_ms"] / 1000, e["audio_end_ms"] / 1000) for e in finals]
    res["clips_cut_by_vad"] = sum(sum(1 for a, b in spans if min(b, e) - max(a, s) > 0.2) > 1 for s, e in bounds)
    print(json.dumps(res, indent=2))
    for e in finals[:4]:
        print(f"  [{e['audio_start_ms']/1000:.1f}-{e['audio_end_ms']/1000:.1f}s @ {e['_t']:.1f}] {e['transcript'][:90]}")
    json.dump({"summary": res, "events": events}, open(args.out, "w"), ensure_ascii=False, indent=1)


asyncio.run(main())

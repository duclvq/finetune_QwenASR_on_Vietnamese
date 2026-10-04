"""Realtime transcription gateway: WebSocket in, Silero VAD, Qwen3-ASR (vLLM server) out.

Protocol is a subset of OpenAI's Realtime *transcription* session (wss://host/v1/realtime):
  client -> server: {"type":"input_audio_buffer.append","audio":"<base64 PCM16 mono>"}   (or raw binary PCM16 frames)
                    {"type":"input_audio_buffer.commit"}                                (finalize the current utterance now)
                    {"type":"transcription_session.update","session":{"language":"vi"}}
  server -> client: transcription_session.created
                    input_audio_buffer.speech_started / speech_stopped      (VAD, with audio_*_ms offsets)
                    conversation.item.input_audio_transcription.partial     (NON-standard: full hypothesis so far)
                    conversation.item.input_audio_transcription.completed   (final text of one utterance)
Sample rate defaults to 24 kHz like OpenAI; pass ?sample_rate=16000 to skip resampling.

How it works: audio is resampled to 16 kHz and fed to Silero VAD in 512-sample frames. While speech is active, the
audio of the current utterance is re-transcribed every PARTIAL_EVERY seconds (partials). When VAD sees MIN_SILENCE_MS of
silence, the utterance is transcribed once more and sent as the final. Utterances longer than MAX_SEG seconds are split.

Run: UPSTREAM=http://127.0.0.1:8100/v1/audio/transcriptions uvicorn gateway:app --port 8200
"""
import asyncio
import base64
import io
import json
import math
import os
import re
import time
import uuid

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly
from silero_vad import VADIterator, load_silero_vad

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8100/v1/audio/transcriptions")
MODEL = os.environ.get("MODEL", "qwen3-asr-vi")
PARTIAL_EVERY = float(os.environ.get("PARTIAL_EVERY", "1.0"))   # seconds of new speech between partials
MIN_SILENCE_MS = int(os.environ.get("MIN_SILENCE_MS", "800"))   # silence that ends an utterance
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.2"))
MAX_SEG = float(os.environ.get("MAX_SEG", "25"))                # model was trained on clips <= 30 s
PREROLL = 0.2                                                   # extra audio kept before detected speech start
SR, FRAME = 16000, 512
PREFIX = re.compile(r"^\s*language\s+\S+?<asr_text>")

torch.set_num_threads(1)
app = FastAPI()
http = httpx.AsyncClient(timeout=60)


async def transcribe(audio, language):
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    r = await http.post(UPSTREAM, files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                        data={"model": MODEL, "language": language})
    r.raise_for_status()
    return PREFIX.sub("", r.json()["text"]).strip()


class Session:
    def __init__(self, ws, in_sr):
        self.ws, self.in_sr, self.language = ws, in_sr, "vi"
        self.vad = VADIterator(load_silero_vad(), threshold=VAD_THRESHOLD, sampling_rate=SR,
                               min_silence_duration_ms=MIN_SILENCE_MS, speech_pad_ms=100)
        self.buf = np.zeros(0, np.float32)   # audio since buf_off (absolute sample index)
        self.buf_off = 0
        self.total = 0                        # absolute samples received (16 kHz)
        self.pending = np.zeros(0, np.float32)
        self.speaking, self.seg_start, self.seg_id = False, 0, None
        self.last_partial_total, self.partial_busy = 0, False
        self.finals = asyncio.Queue()
        self.send_lock = asyncio.Lock()

    async def send(self, **ev):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(ev, ensure_ascii=False))

    def ms(self, samples):
        return int(samples * 1000 / SR)

    def to16k(self, pcm):
        x = pcm.astype(np.float32) / 32768.0
        if self.in_sr != SR:
            g = math.gcd(self.in_sr, SR)
            x = resample_poly(x, SR // g, self.in_sr // g).astype(np.float32)
        return x

    def start_segment(self, abs_start):
        self.speaking, self.seg_id = True, "item_" + uuid.uuid4().hex[:12]
        self.seg_start = max(int(abs_start - PREROLL * SR), self.buf_off)
        self.last_partial_total = self.total
        return self.seg_id

    async def finalize(self, a, b):
        """Queue the utterance [a, b) for its final transcription and reset segment state."""
        item = self.seg_id
        audio = self.buf[a - self.buf_off: b - self.buf_off].copy()
        await self.finals.put((item, audio, self.ms(a), self.ms(b)))
        keep = max(b - int(PREROLL * SR), self.buf_off)         # trim, but keep a little for the next pre-roll
        self.buf, self.buf_off = self.buf[keep - self.buf_off:], keep
        self.speaking, self.seg_id = False, None

    async def feed(self, pcm):
        x = self.to16k(pcm)
        self.buf = np.concatenate([self.buf, x])
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            frame_start = self.total
            self.total += FRAME
            ev = self.vad(torch.from_numpy(frame), return_seconds=False)
            if not ev:
                continue
            if "start" in ev and not self.speaking:
                item = self.start_segment(ev["start"])
                await self.send(type="input_audio_buffer.speech_started", item_id=item, audio_start_ms=self.ms(ev["start"]))
            elif "end" in ev and self.speaking:
                end = min(int(ev["end"]), self.total)
                await self.send(type="input_audio_buffer.speech_stopped", item_id=self.seg_id, audio_end_ms=self.ms(end))
                await self.finalize(self.seg_start, end)
        if self.speaking and self.total - self.seg_start > MAX_SEG * SR:   # force-split very long utterances
            await self.finalize(self.seg_start, self.total)
            self.start_segment(self.total)
        if (self.speaking and not self.partial_busy and self.total - self.last_partial_total >= PARTIAL_EVERY * SR
                and self.total - self.seg_start >= 0.5 * SR):
            self.partial_busy, self.last_partial_total = True, self.total
            audio = self.buf[self.seg_start - self.buf_off:].copy()
            asyncio.create_task(self.partial(self.seg_id, audio))

    async def partial(self, item, audio):
        try:
            text = await transcribe(audio, self.language)
            if self.seg_id == item and text:     # drop if the utterance was finalized meanwhile
                await self.send(type="conversation.item.input_audio_transcription.partial", item_id=item, text=text)
        finally:
            self.partial_busy = False

    async def commit(self):
        if self.speaking:
            await self.send(type="input_audio_buffer.speech_stopped", item_id=self.seg_id, audio_end_ms=self.ms(self.total))
            await self.finalize(self.seg_start, self.total)
            self.vad.triggered, self.vad.temp_end = False, 0   # keep the VAD's sample counter (reset_states would zero it)

    async def final_worker(self):
        while True:
            item, audio, a_ms, b_ms = await self.finals.get()
            try:
                text = await transcribe(audio, self.language)
            except Exception as e:
                await self.send(type="error", item_id=item, message=str(e))
                continue
            await self.send(type="conversation.item.input_audio_transcription.completed", item_id=item,
                            transcript=text, audio_start_ms=a_ms, audio_end_ms=b_ms)


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM}


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket, sample_rate: int = 24000, language: str = "vi"):
    await ws.accept()
    s = Session(ws, sample_rate)
    s.language = language
    worker = asyncio.create_task(s.final_worker())
    await s.send(type="transcription_session.created", session={"input_sample_rate": sample_rate, "language": language,
                 "turn_detection": {"type": "silero_vad", "threshold": VAD_THRESHOLD, "silence_duration_ms": MIN_SILENCE_MS}})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await s.feed(np.frombuffer(msg["bytes"], dtype="<i2"))
                continue
            ev = json.loads(msg["text"])
            t = ev.get("type")
            if t == "input_audio_buffer.append":
                await s.feed(np.frombuffer(base64.b64decode(ev["audio"]), dtype="<i2"))
            elif t == "input_audio_buffer.commit":
                await s.commit()
            elif t == "transcription_session.update":
                s.language = ev.get("session", {}).get("language", s.language)
    except WebSocketDisconnect:
        pass
    finally:
        worker.cancel()

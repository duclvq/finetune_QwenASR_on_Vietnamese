# finetune_QwenASR_on_Vietnamese

Fine-tuning [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) for Vietnamese dialect speech with **LoRA, QLoRA and full fine-tuning**, plus weight-only quantization (q8 → q2) and vLLM serving. Everything here was run on a single 24 GB GPU; every training step is logged (`results/*/train_log.jsonl`).

Models on Hugging Face (public):
- LoRA adapter: [`duclvQ/qwen3-asr-1.7b-vimd-lora`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-lora)
- QLoRA adapter: [`duclvQ/qwen3-asr-1.7b-vimd-qlora`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-qlora)
- Full fine-tune (+ `q8/`, `q4/` HQQ variants): [`duclvQ/qwen3-asr-1.7b-vimd-full`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full)

## Data

[ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) (Vietnamese multi-dialect speech), North + Central regions, clips ≤ 30 s, converted to 16 kHz int16 (`data_prep/build_vimd_dataset.py`).

| Split | Clips | Hours |
|---|---|---|
| train | 10,346 | 54.9 |
| valid | 1,321 | 7.0 |
| test | 1,368 | 7.2 (939 speakers, none in train) |

Transcripts are verbatim: numbers spelled out, fillers and repetitions kept. Check the dataset's license before reuse.

## Results (test set, 1,368 clips)

WER/CER after lowercasing and stripping punctuation, `language="Vietnamese"` forced. One seed per method.

| System | WER | CER | WER North | WER Central | Train time | Peak GPU | Trainable |
|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B (zero-shot) | 11.56 | 8.12 | 8.56 | 15.45 | – | – | – |
| LoRA (r=16, lr 1e-4) | 8.72 | 6.29 | 6.59 | 11.47 | 62 min | 13.1 GB | 17.4M (0.85%) |
| QLoRA (NF4, r=16, lr 1e-4) | 9.45 | 6.80 | 7.23 | 12.31 | 75 min | 13.3 GB | 17.4M |
| Full FT (lr 1e-5) | **8.59** | **6.17** | 6.60 | **11.17** | 126 min | 18.1 GB | 2.04B |

Common settings: 2 epochs (1,294 steps), effective batch 16, cosine schedule, bf16, gradient checkpointing, seed 42.
Full FT beats LoRA by only 0.13 WER (within noise) at twice the training time and 5 GB more memory, so LoRA is the better trade-off here. QLoRA is ~0.7 WER behind LoRA and was slower with no memory benefit at this model size.

### Is the gain real recognition or just label format?

Partly format. Analysis on the LoRA predictions (`results/eval_test/`):
- The references spell numbers out; the base model writes digits in 74 clips. Those clips account for 0.65 of the 2.84 WER points gained.
- Filler/repetition handling ("cái", "là", "thì", repeats) accounts for ~0.4 points.
- Substitutions (dialect words, e.g. `chi→tri`, `chèo→trèo`, `các→cả`) fell from 5.72% to 3.93% (63% of the gain).
- With digit clips and fillers/repeats neutralised, WER goes 8.51 → 6.69, so ~64% of the raw gain remains.
- No speaker overlap between train and test; 4 of 1,368 test texts appear in train.

The filler list was hand-picked, so treat "~64%" as roughly 55–70%.

## Quantization (full fine-tuned model)

Weight-only [HQQ](https://github.com/mobiusml/hqq) (no calibration) on the LLM linears; audio encoder stays bf16. 400 random test clips unless noted.

| Variant | WER | CER | GPU after load |
|---|---|---|---|
| bf16 | 8.68 | 6.29 | 3.80 GB |
| q8 | 8.69 | 6.29 | 2.66 GB |
| q6 (RTN simulation, no memory saving) | 8.63 | 6.18 | 3.80 GB |
| q4 | 10.29 | 7.21 | 2.00 GB |
| q3 (group 64) | 100.92 | 96.84 | 1.88 GB |

On a different 100-clip subset (bf16 = 9.98 WER): q4 11.64, q3 group 16 23.85, q2 group 16 104.0, q2 group 64 108.1. In short: q8/q6 are free, q4 costs ~1.6 WER, q3 and q2 collapse without calibration. These are floors for a calibration-free method, not for quantization in general; AWQ/GPTQ/QAT were not tried. HQQ inference in transformers is ~2x slower than bf16.

## Serving with vLLM

`./serve_vllm.sh` serves the full model (OpenAI-compatible, `127.0.0.1:8100`, 96 concurrent sequences (`MAX_SEQS`, was 32), `max-model-len 4096`, `gpu-memory-utilization 0.5`).

- Supported: `POST /v1/audio/transcriptions` (`json`/`text`, `stream=true`), `/v1/chat/completions` with `audio_url`.
- Not supported: `verbose_json`/`srt`/`vtt` (no timestamps), OpenAI Realtime API, Batch API.
- Measured: ~16 req/s (~300x real time) at 32–64 clients; p95 latency 2.7 s at 32, 12 s at 128 (requests queue, none failed). WER through vLLM (8.13 on 300 clips) matched offline inference (8.40).
- **Outputs start with `language Vietnamese<asr_text>`** (the training target format); strip it client-side with `^\s*language\s+\S+?<asr_text>`, otherwise WER is inflated.
- Untested: audio longer than 30 s (training clips were ≤ 30 s; 58 s and 142 s files returned plausible text but were not scored).

## Realtime transcription (proof of concept)

`realtime/gateway.py` is a WebSocket gateway in front of the vLLM server: audio in, Silero VAD, partial and final transcripts out. The vLLM server itself has no realtime endpoint, and Qwen's native streaming (`init_streaming_state`) was not used.

```bash
./serve_vllm.sh                                   # upstream ASR on :8100
uvicorn gateway:app --app-dir realtime --port 8200 --workers 6   # one worker saturates a CPU core at ~30 users
python realtime/test_stream.py --clips 12         # streams test clips at real-time speed
```

- **Protocol:** subset of OpenAI's Realtime *transcription* session at `ws://host:8200/v1/realtime` (`input_audio_buffer.append/commit`, `speech_started/stopped`, `conversation.item.input_audio_transcription.completed`). Default input is 24 kHz PCM16 like OpenAI; `?sample_rate=16000` skips resampling. Partials use a **non-standard** event `...transcription.partial` carrying the full hypothesis so far.
- **VAD:** yes, Silero VAD on 512-sample frames (threshold 0.2, 800 ms of silence ends an utterance, utterances over 25 s are split).
- **Partials:** while speech is active, the audio of the current utterance is re-transcribed every 1 s (re-decoding a growing buffer, not incremental decoding). At end of speech it is transcribed once more as the final.

Measured with 12 test clips (209 s) streamed at 1x speed: 14 finals, stream WER 11.62 vs 9.94 for offline per-clip decoding of the same clips; final transcript arrives p50 1.15 s / p95 1.43 s after the audio of that utterance ends (about 0.8 s of that is the silence wait); first partial p50 1.24 s after speech starts. On a different 30 clips (4x pacing, WER only): 11.07 vs 11.46 offline.

Things that mattered: Silero's default threshold (0.5) missed most of one noisy clip and added about 5 WER points on the 12-clip stream; 0.2 fixed it. Silence length changed segmentation (27 / 17 / 15 utterances at 500 / 800 / 1200 ms) but not WER.

Limitations: only concatenated clean test clips with digital-silence gaps were tested (no microphone, no background noise, no overlapping speakers, one client at a time). The 0.2 threshold was tuned on 12 clips and its false-alarm rate on real noise is untested. 24 kHz resampling is per message and was not tested. Utterances split mid-sentence at long pauses lose some context.

### Capacity (simulated users streaming at 1x, nearly continuous speech)

`python realtime/load_test.py --users 16 32 56 --seconds 60` (raw results in `results/realtime_load/`). vLLM with 96 sequences, 6 gateway workers, RTX PRO 4000 Blackwell 24 GB:

| Users | 4 | 16 | 24 | 32 | 40 | 56 | 80 | 112 | 144 |
|---|---|---|---|---|---|---|---|---|---|
| final latency p50 (s) | 1.21 | 1.57 | 1.73 | 2.20 | 2.56 | 3.24 | 3.93 | 5.07 | 7.40 |
| final latency p95 (s) | 1.53 | 2.17 | 2.56 | 3.28 | 3.72 | 5.25 | 6.36 | 7.88 | 11.42 |

p95 stays under 3 s up to ~28 users, under 5 s up to ~52, and no request failed in the two-generator runs at 112 and 144 users (the single-generator runs had 1 and 2 HTTP 400 errors, cause not investigated). Stream WER stayed at 7.9-9.6% up to 112 users.

- GPU: 136-143 W of a 145 W limit from 16 users on (SM clock 2430 vs 3090 MHz max in one sample at 56 users), so it is power-limited; nvidia-smi "utilization" averages 50-70%, which is not a saturation measure. VRAM 11-14 GB of 24 GB.
- A single gateway process hit 100% of one CPU core at ~30 users (p95 6.5 s at 32, 35 s at 40) while the GPU sat at 27-37%; 6 workers removed that bottleneck. vLLM `max-num-seqs` 32 to 96 cut p95 at 56 users from 7.2 to 5.25 s. Partials every 2 s instead of 1 s cut it further to 4.2 s (`PARTIAL_EVERY=2`), at the cost of a slower first partial.
- Limits: clean test clips with digital-silence gaps only, one run per level, no real microphone or noise; whether vLLM's single engine loop (~1.8 CPU cores at 56 users) is a ceiling was not tested.

## Reproduce

```bash
./setup.sh                      # venv (uv), qwen-asr[vllm]==0.0.6, models; edit env.sh for your GPU
source env.sh
pip install bitsandbytes hqq silero-vad   # QLoRA, quantization, realtime VAD
python data_prep/build_vimd_dataset.py --out data/prepared
export VIMD_DATA=$PWD/data/prepared
./run_all.sh                    # LoRA -> QLoRA -> test eval (base, LoRA, QLoRA)
./run_full.sh                   # full fine-tune -> test eval
python eval_quant.py --configs bf16 q8 q6 q4 q3 q2 --max_samples 400
./serve_vllm.sh
```

| File | Purpose |
|---|---|
| `train_qwen_asr.py` | LoRA / QLoRA / full training (`--mode lora\|qlora\|full`), logs every step to `train_log.jsonl` |
| `eval_wer.py` | Test-set WER/CER for base, adapters and full models (overall + per region) |
| `eval_quant.py`, `load_hqq_asr.py` | Quantization sweep and loader for the HQQ checkpoints |
| `serve_vllm.sh` | vLLM server config |
| `realtime/` | WebSocket realtime gateway (Silero VAD) and streaming test client |
| `results/` | Per-step training logs, run summaries, all test predictions |

Tested with: torch 2.9.1, transformers 4.57.6, peft 0.21.2, bitsandbytes 0.50.2, hqq 0.2.8, vllm 0.14.0, qwen-asr 0.0.6.

## Notes and limitations

- One seed and one hyperparameter setting per method; differences below ~0.5 WER are not reliable.
- QLoRA numbers are for the adapter merged into the bf16 base, not the 4-bit base it trained against.
- Full-FT checkpoints saved weights only (disk was tight), so a resumed run restarts the optimizer.
- A label-masking bug (the processor left-pads batches) was found and fixed before the real runs; the first pilot showed an implausible loss of ~24.
- The upstream training script is in [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning); this repo adds PEFT/QLoRA, per-step logging, evaluation, quantization and serving.

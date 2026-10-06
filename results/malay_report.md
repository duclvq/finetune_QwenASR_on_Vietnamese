# Qwen3-ASR-1.7B LoRA on Malay — experiment report (2026-10-06)

## Setup

| | |
|---|---|
| Base model | Qwen3-ASR-1.7B (bf16) |
| Method | LoRA r=16, alpha=32, attention + MLP of the LLM (17.4M trainable, 0.85%) |
| Train data | 50.0h / 29,966 clips, random subset of `mesolitica/malaya-speech-malay-stt` (1.6M clips, ~2,600h) |
| Hyper-params | 2 epochs, lr 1e-4 cosine, batch 2 x grad_acc 8 (effective 16), 3,746 steps |
| Hardware / time | RTX 5060 Ti 16GB (Windows), 4.1h wall, 5.96GB peak |
| Final eval loss | 0.777 (valid, 200 clips) |

Data build: `data_prep/build_malay_dataset.py` (500h, seeded split: 2,000 valid / 2,000 test held out by row),
then a 50h random subset (`data_prep/subset_malay.py`). Labels are lowercase Malay without punctuation or digits.

## Results

### In-domain test (1,994 clips from the same corpus)

Scored with `score_malay.py` (MalayTextNormalizer from revolab-asr-benchmark + Malay number spelling):

| System | WER | CER | Sub | Del | Ins |
|---|---|---|---|---|---|
| base | 52.10 | 30.55 | 29.21 | 3.03 | 19.85 |
| LoRA 50h | **37.07** | **21.21** | 23.04 | 6.60 | 7.42 |

Split by label quality (VAD speech time vs transcript length, see below):

| System | clean clips (1,860) | suspect clips (134) |
|---|---|---|
| base | 46.8 (del 2.8, ins 15.4) | 273.4 |
| LoRA 50h | **34.9** (del 6.4, ins 6.2) | 126.0 |

### Out-of-domain: Revolab/ASR-Benchmark-Public (820 clips, 12 domains)

Benchmark scoring (MalayTextNormalizer + numbers, best of `text`/`normalized_text` per row):

| Domain | n | base | LoRA 50h |
|---|---|---|---|
| **All** | 820 | **15.01** | 16.14 |
| news | 52 | 23.11 | **12.97** |
| short-inputs | 52 | 31.67 | **17.65** |
| read-speech | 51 | 7.98 | **3.68** |
| commonvoice | 141 | 8.40 | **7.36** |
| fleurs | 153 | **7.34** | 7.38 |
| drama | 52 | 26.91 | 26.80 |
| parliament | 52 | **13.93** | 15.38 |
| animation | 50 | **17.35** | 20.81 |
| podcast | 52 | **16.21** | 22.76 |
| street interview | 48 | **37.35** | 45.38 |
| singing | 51 | **18.72** | 36.76 |
| telephony | 66 | **14.14** | 33.47 |

## Analysis

1. **Big in-domain gain**: WER 52.1 -> 37.1 (46.8 -> 34.9 on clean clips). The model learns the corpus' spoken
   Malay vocabulary and conversational style; most remaining substitutions are short function words in fast
   conversational speech (di/dia, nak/anak, korang/orang, dah/dan).
2. **Truncated labels.** The corpus is semi-supervised (Google STT). Silero VAD shows ~6% of clips whose transcript is
   far too short for the amount of speech (< 6 chars per speech-second vs a median of 14.2), plus ~1% with no speech.
   On those test clips the base model outputs ~3x more words than the reference (it is right, the label is partial).
   The LoRA learned to drop words: deletions doubled (3.0 -> 6.6 in-domain; singing 3.2 -> 20.5,
   street interview 13.4 -> 29.1 on Revolab), with outputs like "Perjalananku terang menyuluh segala" -> "berjalan".
3. **English / code-switch forgetting.** The training data is Malay-only, and the LoRA now maps English speech onto
   Malay words ("that's all" -> "datang", "Okay, that sounds like a deal to me" -> "okey tak suka ke dia tu ni").
   That is why telephony (14 -> 33) and parts of short-inputs/animation/drama regress, and why Revolab overall is
   slightly worse than base despite the in-domain win.
4. Numbers and punctuation are not a factor: normalization changes WER by < 2 points for either system.

## Data improvements (built, training pending)

| Set | Content | Purpose |
|---|---|---|
| `D:/data/malay_50h_clean` (50.0h, 30,225 clips) | drop 1,997 clips with chars/speech-sec < 6 or no speech; top up with 2,256 clips from the 500h pool passing the same check (`data_prep/clean_malay_subset.py`) | fix deletions from truncated labels |
| `D:/data/malay_50h_mix` (63.0h, 34,425 clips) | clean set + Synth-Manglish (2,236 TTS code-switch clips) + cs_pilot (real podcast CS clips) + 5h FLEURS en_us tagged `language English` (`data_prep/build_malay_mix.py`); `cs_test` (221 clips) held out | fix English/code-switch forgetting |

FLEURS ms_my was deliberately not used: Revolab's `fleurs` domain comes from it.

Status: the clean-data run (`outputs/lora-1.7b-malay50h-clean`) was stopped by the host at step 510 because system RAM
ran out (checkpoint-400 kept). The mixed run has not started.

## Next steps

1. Resume `lora-1.7b-malay50h-clean` from checkpoint-400 (~3.5h), then train `lora-1.7b-malay50h-mix` (~4.5h).
2. Evaluate all on test (clean subset), Revolab, and `cs_test`:
   `eval_wer.py --skip_base --runs <run> --data D:/data/malay_50h --lang Malay --out outputs/eval_malay50h`,
   `score_malay.py`, `eval_revolab.py --skip_base --runs <run>`,
   `eval_wer.py --data D:/data/malay_50h_mix --split cs_test --out outputs/eval_cs_test`.
3. If English still regresses, raise the English/CS share or lower the LoRA lr; the real code-switch corpus from
   `D:/cs_mining` (target 50h) is the better long-term source than TTS.

## Reproduce

```bash
export USE_TF=0   # the base conda env has TensorFlow installed, which breaks transformers' Trainer import
python data_prep/build_malay_dataset.py --out D:/data/malay_prepared --max_hours 500
python data_prep/subset_malay.py --src D:/data/malay_prepared --out D:/data/malay_50h --hours 50
python train_qwen_asr.py --mode lora --model_path <Qwen3-ASR-1.7B> --data D:/data/malay_50h --lang Malay \
  --output_dir outputs/lora-1.7b-malay50h --epochs 2 --lr 1e-4 --batch_size 2 --grad_acc 8 --eval_steps 200 --save_steps 200
python eval_wer.py --model_path <Qwen3-ASR-1.7B> --data D:/data/malay_50h --lang Malay --runs lora-1.7b-malay50h --out outputs/eval_malay50h
python score_malay.py --dir outputs/eval_malay50h
python eval_revolab.py --model_path <Qwen3-ASR-1.7B> --runs lora-1.7b-malay50h
```

Per-clip predictions and summaries: `results/malay/`.

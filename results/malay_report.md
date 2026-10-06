# Qwen3-ASR-1.7B LoRA on Malay — experiment report (2026-10-06)

**Bottom line:** a Malay-only LoRA wins in-domain but loses to the base model out of domain (truncated labels +
English forgetting). Cleaning labels helps a little; mixing in code-switched and English speech fixes most of it:
`lora-1.7b-malay50h-mix` keeps the in-domain gain (WER 52.1 -> 37.4) and beats base on the Revolab benchmark
(15.01 -> **13.38**, -11% relative), but still trails base on telephony, singing and podcast.

## Systems

| System | Train data | Hours / clips | Steps |
|---|---|---|---|
| base | — | — | — |
| lora | random 50h of `mesolitica/malaya-speech-malay-stt` | 50.0h / 29,966 | 3,746 |
| lora-clean | same, minus truncated-label clips (VAD), topped up to 50h | 50.0h / 30,225 | 3,778 |
| lora-mix | lora-clean + Synth-Manglish + cs_pilot (code-switch) + 5h FLEURS en_us (`language English`) | 63.0h / 34,425 | 4,304 |

All LoRA runs: r=16, alpha=32 on the LLM's attention + MLP (17.4M trainable, 0.85%), 2 epochs, lr 1e-4 cosine,
effective batch 16 (lora: 2 x 8; clean / mix: 4 x 4), RTX 5060 Ti 16GB on Windows.
Train wall time: lora 4.1h, lora-clean 1.7h, lora-mix 2.3h (the later runs pin the trainer to P-cores, see below).

## Results

### In-domain test (1,994 clips, same corpus)

`score_malay.py` (MalayTextNormalizer + Malay number spelling). "Clean" = the 1,860 clips whose label is not truncated.

| System | WER | Sub | Del | Ins | WER clean clips |
|---|---|---|---|---|---|
| base | 52.10 | 29.21 | 3.03 | 19.85 | 46.76 |
| lora | **37.07** | 23.04 | 6.60 | 7.42 | 34.92 |
| lora-clean | 37.32 | 23.25 | 5.59 | 8.49 | 34.69 |
| lora-mix | 37.35 | 23.29 | 5.35 | 8.71 | **34.63** |

### Revolab/ASR-Benchmark-Public (820 clips, 12 domains, out of domain)

Benchmark scoring (MalayTextNormalizer + numbers, best of `text`/`normalized_text` per row). WER:

| Domain | n | base | lora | lora-clean | lora-mix |
|---|---|---|---|---|---|
| **All** | 820 | 15.01 | 16.14 | 15.76 | **13.38** |
| news | 52 | 23.11 | 12.97 | 13.39 | **11.13** |
| short-inputs | 52 | 31.67 | 17.65 | 18.49 | **13.45** |
| drama | 52 | 26.91 | 26.80 | 26.14 | **21.26** |
| read-speech | 51 | 7.98 | **3.68** | 4.60 | **3.68** |
| commonvoice | 141 | 8.40 | 7.36 | 7.45 | **7.08** |
| fleurs | 153 | 7.34 | 7.38 | 7.47 | **6.93** |
| animation | 50 | 17.35 | 20.81 | 18.70 | **16.89** |
| parliament | 52 | **13.93** | 15.38 | 15.48 | 14.43 |
| street interview | 48 | **37.35** | 45.38 | 42.27 | 37.95 |
| podcast | 52 | **16.21** | 22.76 | 21.73 | 18.78 |
| singing | 51 | **18.72** | 36.76 | 29.45 | 23.06 |
| telephony | 66 | **14.14** | 33.47 | 38.37 | 19.11 |

### Code-switch test (`cs_test`, 221 held-out clips)

| System | All | Synth-Manglish (203, TTS) | cs_pilot (18, real podcast) |
|---|---|---|---|
| base | 24.61 | 27.20 | **10.32** |
| lora | 26.54 | 24.24 | 39.21 |
| lora-clean | 25.59 | 23.76 | 35.65 |
| lora-mix | **8.31** | **7.42** | 13.23 |

The Synth-Manglish part is the same TTS source (and voices) as lora-mix's training data, so 7.4 is optimistic;
the 18 real podcast clips are the honest signal: lora-mix recovers from 39 to 13 but is still behind base (10.3).

## Analysis

1. **In-domain gain is real and stable** (~37 WER vs 52, ~34.6 vs 46.8 on clean clips) and does not depend on the
   data variant. Remaining errors are mostly short function words in fast conversational speech
   (di/dia, nak/anak, korang/orang, dah/dan).
2. **Truncated labels -> deletions.** The corpus is semi-supervised (Google STT); Silero VAD finds ~6% of clips whose
   transcript is far too short for the speech (< 6 chars per speech-second vs a median of 14.2) and ~1% with no
   speech. Training on them taught the model to drop words. Removing them (lora-clean) lowers deletions
   (test 6.6 -> 5.6, cs_test 7.2 -> 5.8, singing 36.8 -> 29.5, street interview 45.4 -> 42.3), but the overall gain is
   small because the test labels are equally noisy.
3. **English / code-switch forgetting was the main out-of-domain problem.** Malay-only LoRAs map English speech onto
   Malay words ("that's all" -> "datang"), wrecking telephony (14 -> 33-38) and real code-switched speech (10 -> 36-39).
   Adding 7.3h code-switch + 5h English (lora-mix) fixes most of it: telephony 38.4 -> 19.1, real CS 35.7 -> 13.2,
   short-inputs 18.5 -> 13.5, and the Revolab total drops below base.
4. Numbers and punctuation are not a factor: normalization moves WER by < 2 points for every system.

## Training-infrastructure findings (Windows, RTX 5060 Ti 16GB, i5-14400F)

- The trainer is single-thread CPU-bound at small batch. Windows schedules the hidden background process on E-cores,
  halving speed; pinning to P-cores (`CPU_AFFINITY=0-11`) took throughput from 4.4 to ~8.7 samples/s (GPU 30% -> 92%).
- CUDA "sysmem fallback" lets the caching allocator silently spill past VRAM into shared system RAM (9GB seen at
  batch 8) instead of freeing cache; a per-process cap (`CUDA_MEM_FRACTION=0.8`) prevents it.
- Trainer restores the checkpoint's `train_batch_size` on resume, silently ignoring a new `--batch_size`;
  `train_qwen_asr.py` now refuses such a resume.
- HF eval loss is a mean of per-batch means, so it shifts with eval batch size; compare runs by WER, not eval loss.

## Next steps

1. Real code-switched speech is the lever: the `D:/cs_mining` pipeline (target 50h of podcast CS) should replace the
   TTS Synth-Manglish data; the real-CS gap to base (13.2 vs 10.3) is the main thing left.
2. Telephony (8kHz-like) and singing still trail base: add narrowband / music augmentation, or more such data.
3. Try a larger English/CS share or a lower LoRA lr to trade a little in-domain WER for robustness.

## Reproduce

```bash
export USE_TF=0 CPU_AFFINITY=0-11 CUDA_MEM_FRACTION=0.8   # TF in the base env breaks transformers' Trainer import
python data_prep/build_malay_dataset.py --out D:/data/malay_prepared --max_hours 500
python data_prep/subset_malay.py --src D:/data/malay_prepared --out D:/data/malay_50h --hours 50
python data_prep/vad_stats.py D:/data/malay_50h test train
python data_prep/clean_malay_subset.py --src D:/data/malay_50h --pool D:/data/malay_prepared --out D:/data/malay_50h_clean
python data_prep/build_malay_mix.py --base D:/data/malay_50h_clean --out D:/data/malay_50h_mix
python train_qwen_asr.py --mode lora --model_path <Qwen3-ASR-1.7B> --data D:/data/malay_50h_mix --lang Malay \
  --output_dir outputs/lora-1.7b-malay50h-mix --epochs 2 --lr 1e-4 --batch_size 4 --grad_acc 4 --eval_steps 200 --save_steps 200
python eval_wer.py --model_path <Qwen3-ASR-1.7B> --data D:/data/malay_50h --lang Malay --runs lora-1.7b-malay50h-mix --out outputs/eval_malay50h
python score_malay.py --dir outputs/eval_malay50h
python eval_revolab.py --model_path <Qwen3-ASR-1.7B> --runs lora-1.7b-malay50h-mix
python eval_wer.py --model_path <Qwen3-ASR-1.7B> --data D:/data/malay_50h_mix --split cs_test --lang Malay --runs lora-1.7b-malay50h-mix --out outputs/eval_cs_test
python tb_sync.py & tensorboard --logdir outputs/tb   # live curves from train_log.jsonl
```

Per-clip predictions, scores and train logs for all systems: `results/malay/`.

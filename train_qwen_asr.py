"""LoRA / QLoRA fine-tune Qwen3-ASR on prepared ViMD audio (int16 memmap + meta.jsonl).

Logs EVERY optimizer step to <output_dir>/train_log.jsonl (loss, lr, grad_norm,
step time, samples/s, GPU memory) for the final report, plus eval loss on the
valid split every --eval_steps. Resumes from the latest checkpoint if one exists.

Usage: python train_qwen_asr.py --mode lora|qlora|full --output_dir outputs/lora-1.7b
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import BitsAndBytesConfig, GenerationConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from qwen_asr import Qwen3ASRModel

DATA = os.environ.get("VIMD_DATA", os.path.expanduser("~/training/vimd-whisper/data/prepared"))  # prepared ViMD dir (see data_prep/)
LANG = "Vietnamese"  # Qwen3-ASR language tag; override with --lang (e.g. Malay)
LORA_TARGETS = r".*thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


class PreparedAudio(Dataset):
    def __init__(self, split_dir, max_samples=None, seed=0):
        with open(os.path.join(split_dir, "meta.jsonl"), encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]
        if max_samples and max_samples < len(self.meta):
            idx = np.random.default_rng(seed).choice(len(self.meta), max_samples, replace=False)
            self.meta = [self.meta[i] for i in sorted(idx)]
        self.path = os.path.join(split_dir, "audio.bin")
        self.audio = None  # lazy memmap, one per dataloader worker

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        if self.audio is None:
            self.audio = np.memmap(self.path, dtype=np.int16, mode="r")
        m = self.meta[i]
        pcm = self.audio[m["offset"]: m["offset"] + m["length"]]
        return {"audio": pcm.astype(np.float32) / 32768.0, "text": m["text"], "lang": m.get("lang")}


class Collator:
    def __init__(self, processor, lang=LANG):
        self.processor = processor
        self.lang = lang  # default tag; a clip's meta "lang" (e.g. English in a mixed set) overrides it
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": None}]}]
        self.prefix = processor.apply_chat_template([msgs], add_generation_prompt=True, tokenize=False)[0]
        self.eos = processor.tokenizer.eos_token or ""

    def __call__(self, batch):
        audios = [b["audio"] for b in batch]
        prefixes = [self.prefix] * len(batch)
        default = getattr(self, "lang", None) or self.lang_prefix[len("language "):-len("<asr_text>")]  # old pickles
        full = [self.prefix + f"language {b.get('lang') or default}<asr_text>" + b["text"] + self.eos for b in batch]
        full_in = self.processor(text=full, audio=audios, return_tensors="pt", padding=True, truncation=False)
        pre_in = self.processor(text=prefixes, audio=audios, return_tensors="pt", padding=True, truncation=False)
        labels = full_in["input_ids"].clone()
        T = labels.size(1)
        pre_lens = pre_in["attention_mask"].sum(dim=1).tolist()
        full_lens = full_in["attention_mask"].sum(dim=1).tolist()
        for i, (pl, fl) in enumerate(zip(pre_lens, full_lens)):
            start = T - fl if full_in["attention_mask"][i, 0] == 0 else 0  # processor may left-pad
            labels[i, : start + pl] = -100
        pad = self.processor.tokenizer.pad_token_id
        if pad is not None:
            labels[labels == pad] = -100
        full_in["labels"] = labels
        return full_in


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=dtype)
        return inputs


class StepLogger(TrainerCallback):
    """Appends one JSON line per logged step (logging_steps=1 -> every step)."""

    def __init__(self, path):
        self.path = path
        self.t0 = time.time()
        self.last = self.t0
        self.fh = open(path, "a", buffering=1)

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        now = time.time()
        row = {"step": state.global_step, "epoch": state.epoch, "elapsed_s": round(now - self.t0, 1),
               "step_time_s": round(now - self.last, 3),
               "gpu_mem_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
               "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **logs}
        self.last = now
        self.fh.write(json.dumps(row) + "\n")
        if state.global_step % 10 == 0 or "eval_loss" in logs:
            print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "qlora", "full"], required=True)
    ap.add_argument("--model_path", default="models/Qwen3-ASR-1.7B")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default=DATA, help="prepared data dir with train/ and valid/")
    ap.add_argument("--lang", default=LANG)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=2)

    kw = dict(dtype=torch.bfloat16, device_map={"": 0})
    if args.mode == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["audio_tower", "lm_head"])  # quantize only the LLM; keep encoder/head bf16
    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, **kw)
    model, processor = wrapper.model, wrapper.processor

    # route Trainer's outer forward to thinker (same patch as the official SFT script)
    cls = model.__class__
    if not getattr(cls, "_forward_patched", False):
        def forward(self, input_ids=None, attention_mask=None, input_features=None,
                    feature_attention_mask=None, labels=None, **kwargs):
            return self.thinker.forward(input_ids=input_ids, attention_mask=attention_mask,
                                        input_features=input_features,
                                        feature_attention_mask=feature_attention_mask, labels=labels, **kwargs)
        cls.forward, cls._forward_patched = forward, True
    # the wrapper doesn't implement get_input_embeddings, which PEFT/gradient checkpointing need
    cls.get_input_embeddings = lambda self: self.thinker.get_input_embeddings()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False

    if args.mode == "qlora":  # also casts non-quantized params (norms etc.) to fp32 for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.mode != "full":
        model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                                 target_modules=LORA_TARGETS, bias="none"))
    # mode == "full": every parameter (audio tower + LLM) trains, bf16 weights like the official script
    trainable, total = (sum(p.numel() for p in model.parameters() if p.requires_grad),
                        sum(p.numel() for p in model.parameters()))
    print(f"[{args.mode}] trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)", flush=True)

    train_ds = PreparedAudio(os.path.join(args.data, "train"), max_samples=args.max_train_samples)
    eval_ds = PreparedAudio(os.path.join(args.data, "valid"), max_samples=args.max_eval_samples)
    print(f"train {len(train_ds)} clips, eval {len(eval_ds)} clips", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0, max_grad_norm=1.0,
        optim="adamw_torch" if args.mode == "lora" else "paged_adamw_8bit",
        bf16=True, logging_steps=1, logging_first_step=True,
        eval_strategy="steps", eval_steps=args.eval_steps, save_steps=args.save_steps, save_total_limit=1 if args.mode == "full" else 2,
        save_only_model=args.mode == "full",  # disk is tight: no optimizer state in full-FT checkpoints
        dataloader_num_workers=args.workers, remove_unused_columns=False, report_to="none", seed=42)
    trainer = CastFloatInputsTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                                     data_collator=Collator(processor, args.lang),
                                     callbacks=[StepLogger(os.path.join(args.output_dir, "train_log.jsonl"))])
    ckpt = get_last_checkpoint(args.output_dir)
    if ckpt:
        print(f"[resume] {ckpt}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=ckpt)
    final_eval = trainer.evaluate()
    if args.mode == "full":
        final = os.path.join(args.output_dir, "final")
        trainer.save_model(final)
        processor.save_pretrained(final)
        for name in ("generation_config.json", "preprocessor_config.json", "chat_template.json"):
            src = os.path.join(args.model_path, name)
            if os.path.exists(src):
                shutil.copy2(src, final)
    else:
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    json.dump({"mode": args.mode, "trainable_params": trainable, "total_params": total,
               "train_wall_s": round(time.time() - t0, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "final_eval": final_eval, "state": trainer.state.log_history[-1]},
              open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    print("[done]", final_eval, flush=True)


if __name__ == "__main__":
    main()

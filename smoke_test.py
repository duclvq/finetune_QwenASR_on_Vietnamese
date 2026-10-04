"""Smoke test Qwen3-ASR-1.7B on both backends. Usage: python smoke_test.py [transformers|vllm]"""
import sys
import time

import torch
from qwen_asr import Qwen3ASRModel

MODEL = "models/Qwen3-ASR-1.7B"
AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "transformers"
    if backend == "vllm":
        model = Qwen3ASRModel.LLM(model=MODEL, gpu_memory_utilization=0.6, max_inference_batch_size=8, max_new_tokens=256, max_model_len=4096)
    else:
        model = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0",
                                              max_inference_batch_size=8, max_new_tokens=256)
    for i in range(2):
        t0 = time.perf_counter()
        res = model.transcribe(audio=AUDIO, language=None)
        print(f"[{backend}] run {i}: {time.perf_counter() - t0:.2f}s | {res[0].language} | {res[0].text}", flush=True)

#!/usr/bin/env bash
# Serve the full fine-tuned Qwen3-ASR (Vietnamese/ViMD) with vLLM on GPU 1 (OpenAI-compatible API).
#   ./serve_vllm.sh                       # local weights, 127.0.0.1:8100
#   HOST=0.0.0.0 API_KEY=secret ./serve_vllm.sh
# Endpoints: POST /v1/audio/transcriptions (OpenAI-style, multipart file) and /v1/chat/completions (audio_url).
cd "$(dirname "$0")" && source env.sh     # CUDA_VISIBLE_DEVICES=1, PCI bus order, venv
MODEL=${MODEL:-outputs/full-1.7b/final}                  # or duclvQ/qwen3-asr-1.7b-vimd-full
HOST=${HOST:-127.0.0.1}                                  # loopback by default: no auth unless API_KEY is set
PORT=${PORT:-8100}                                       # 8000/8080/8084/8041 are taken on this box
exec qwen-asr-serve "$MODEL" \
  --served-model-name qwen3-asr-vi \
  --host "$HOST" --port "$PORT" \
  ${API_KEY:+--api-key "$API_KEY"} \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"audio":1}'

#!/usr/bin/env bash
# Set up the Qwen3-ASR assessment workspace on aidev: code, venv, model weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
UV=~/.local/bin/uv
cd "$ROOT"

echo "== repo"
[ -d .git ] || git init -q
cat > .gitignore <<'EOF'
.venv/
models/
data/
outputs/
third_party/
*.log
__pycache__/
EOF
mkdir -p third_party models data outputs
[ -d third_party/Qwen3-ASR ] || git clone -q https://github.com/QwenLM/Qwen3-ASR.git third_party/Qwen3-ASR
git -C third_party/Qwen3-ASR log -1 --format="Qwen3-ASR commit %h (%cd)"

echo "== venv"
[ -x .venv/bin/python ] || "$UV" venv -q --python 3.12 .venv
# --no-cache: the shared disk is nearly full, so don't keep a second copy of the wheels.
UV_NO_CACHE=1 "$UV" pip install --python .venv/bin/python "qwen-asr[vllm]==0.0.6" datasets jiwer peft
.venv/bin/python - <<'EOF'
import torch, vllm, transformers, qwen_asr
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__, "| cuda available", torch.cuda.is_available())
EOF

echo "== models"
for m in Qwen3-ASR-1.7B Qwen3-ASR-0.6B Qwen3-ForcedAligner-0.6B; do
  [ -f "models/$m/config.json" ] || .venv/bin/hf download "Qwen/$m" --local-dir "models/$m" >/dev/null
  du -sh "models/$m"
done
du -sh .venv
df -h ~ | tail -1
echo "== SETUP DONE"

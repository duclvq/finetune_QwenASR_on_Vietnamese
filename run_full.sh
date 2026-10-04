#!/usr/bin/env bash
# Full fine-tune queue: train -> test eval of the final model (merged into existing eval summary).
cd "$(dirname "$0")" && source env.sh
[ -f outputs/full-1.7b/summary.json ] || python train_qwen_asr.py --mode full --output_dir outputs/full-1.7b \
  --epochs 2 --lr 1e-5 --batch_size 8 --grad_acc 2 --eval_steps 200 --save_steps 200 >> outputs/full-1.7b.stdout.log 2>&1
[ -d outputs/full-1.7b/final ] && python eval_wer.py --skip_base --runs --full full-1.7b >> outputs/eval_full.stdout.log 2>&1
echo FULL_QUEUE_DONE >> outputs/queue_full.log

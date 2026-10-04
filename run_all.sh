#!/usr/bin/env bash
# Sequential queue: LoRA -> QLoRA -> eval (base, lora, qlora on test). Resumable: rerun to continue.
cd "$(dirname "$0")" && source env.sh
COMMON="--epochs 2 --lr 1e-4 --batch_size 8 --grad_acc 2 --eval_steps 100 --save_steps 100"
for m in lora qlora; do
  [ -f outputs/$m-1.7b/summary.json ] && continue
  python train_qwen_asr.py --mode $m --output_dir outputs/$m-1.7b $COMMON >> outputs/$m-1.7b.stdout.log 2>&1
done
[ -f outputs/eval_test/summary.json ] || python eval_wer.py >> outputs/eval.stdout.log 2>&1
echo QUEUE_DONE >> outputs/queue.log

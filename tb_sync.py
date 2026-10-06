"""Mirror every outputs/<run>/train_log.jsonl into TensorBoard event files, live.

Runs alongside training without touching it (works for finished, running and resumed runs):
  python tb_sync.py [--outputs outputs] [--logdir outputs/tb] [--interval 30]
  tensorboard --logdir outputs/tb
`train/loss` is divided by the run's grad_acc (Trainer logs the sum over accumulation steps), so runs with different
batch / grad_acc splits are comparable. Steps re-logged after a resume simply overwrite the same points.
"""
import argparse
import glob
import json
import os
import time

from torch.utils.tensorboard import SummaryWriter

SKIP = {"step", "epoch"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--logdir", default="outputs/tb")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--once", action="store_true", help="sync what exists and exit")
    args = ap.parse_args()
    writers, offsets, grad_acc = {}, {}, {}
    while True:
        for path in glob.glob(os.path.join(args.outputs, "*", "train_log.jsonl")):
            run = os.path.basename(os.path.dirname(path))
            if run not in writers:
                writers[run] = SummaryWriter(os.path.join(args.logdir, run))
                offsets[run] = 0
                a = os.path.join(os.path.dirname(path), "args.json")
                grad_acc[run] = json.load(open(a)).get("grad_acc", 1) if os.path.exists(a) else 1
            if os.path.getsize(path) < offsets[run]:  # run dir was recreated: start over
                offsets[run] = 0
            with open(path, encoding="utf-8") as f:
                f.seek(offsets[run])
                for line in iter(f.readline, ""):
                    if not line.endswith("\n"):  # partial line still being written
                        break
                    offsets[run] = f.tell()
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step, w = row["step"], writers[run]
                    for k, v in row.items():
                        if k in SKIP or not isinstance(v, (int, float)):
                            continue
                        if k.startswith("eval_"):
                            w.add_scalar(f"eval/{k[5:]}", v, step)
                        elif k == "loss":
                            w.add_scalar("train/loss", v / grad_acc[run], step)
                        elif k.startswith("gpu_") or k in ("step_time_s", "elapsed_s"):
                            w.add_scalar(f"system/{k}", v, step)
                        else:
                            w.add_scalar(f"train/{k}", v, step)
                    w.add_scalar("train/epoch", row.get("epoch", 0), step)
            writers[run].flush()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

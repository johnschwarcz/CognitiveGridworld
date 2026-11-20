"""Pre-train the MNIST checkpoints that notebooks 05-07 can load.

    python pretrain_checkpoints.py            # classifier + unconditional + conditional U-Net

On a GPU this takes a few minutes. On a CPU it takes about an hour; the
checkpoints shipped with the tutorial were made exactly this way (CPU, bf16
autocast). Notebooks 05 and 06 show the same training loop step by step.
"""
import argparse
import torch
import dmtools as dm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="training steps per U-Net")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (faster on recent CPUs)")
    ap.add_argument("--only", choices=["classifier", "uncond", "cond"], default=None)
    args = ap.parse_args()
    dev = dm.get_device(verbose=True)
    dm.seed_everything(0)
    if args.only in (None, "classifier"):
        dm.load_or_train_classifier(device=dev, force=True)
    steps = args.steps or (20000 if dev.type == "cuda" else 12000)
    for cond in (False, True):
        if args.only not in (None, "cond" if cond else "uncond"):
            continue
        dm.load_or_train_mnist_model(conditional=cond, steps=steps, device=dev, force=True,
                                     batch_size=args.batch_size, amp=args.amp or None)

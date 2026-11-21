"""Pre-train the MNIST checkpoints that notebooks 05-09 load.

    python pretrain_checkpoints.py                 # classifier + unconditional + conditional U-Net
    python pretrain_checkpoints.py --only cond     # just one of them

Recipe: dm.MNIST_RECIPE U-Net (1.2M parameters), linear schedule with T=1000,
eps-prediction, EMA 0.999, 12 000 steps at batch 64 (the conditional model drops
its label 10% of the time for classifier-free guidance). On a GPU this takes a
few minutes per model; on a 4-core CPU with bfloat16 autocast (--amp) about an
hour per model. The checkpoints in checkpoints/ were made exactly this way.
Notebook 05 writes out the same training loop step by step.
"""
import argparse
import dmtools as dm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000, help="training steps per U-Net")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (default: on for CUDA GPUs that support it)")
    ap.add_argument("--only", choices=["classifier", "uncond", "cond"], default=None)
    args = ap.parse_args()
    dev = dm.get_device(verbose=True)
    if args.only in (None, "classifier"):
        dm.seed_everything(0)
        dm.load_or_train_classifier(device=dev, force=True)
    for cond, seed in ((False, 1), (True, 0)):
        if args.only not in (None, "cond" if cond else "uncond"):
            continue
        dm.seed_everything(seed)
        dm.load_or_train_mnist_model(conditional=cond, steps=args.steps, device=dev, force=True,
                                     batch_size=args.batch_size, amp=True if args.amp else None)

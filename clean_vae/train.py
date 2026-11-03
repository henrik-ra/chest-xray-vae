import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from .config import get_device, parse_args, plain_args
from .dataset import load_cache, prepare
from .losses import VGGPerceptual, negative_elbo
from .model import build_model

PROGRESS_EPOCHS = (1, 2, 5, 10, 20, 40)   # epochs at which fixed-noise samples are stored


@torch.no_grad()
def evaluate_perceptual(model, loader, device, perceptual):
    model.eval()
    total, n = 0.0, 0
    for (xb,) in loader:
        x = xb.to(device).float().div_(255)
        total += perceptual(torch.sigmoid(model(x)[0]), x).item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def evaluate_elbo(model, loader, device, likelihood="bernoulli", sigma=0.1):
    model.eval()
    rec_sum = kl_sum = 0.0
    n = 0
    for (xb,) in loader:
        x = xb.to(device).float().div_(255)
        # same single-sample ELBO estimate as in training, so both curves are comparable
        # (decoding mu instead of a sample z would not be an ELBO any more)
        logits, mu, logvar = model(x)
        rec, kl = negative_elbo(logits, x, mu, logvar, likelihood, sigma)
        rec_sum += rec.item() * len(x)
        kl_sum += kl.item() * len(x)
        n += len(x)
    return rec_sum / n, kl_sum / n


def random_shift(x, pad=4):
    """Shift every image by up to pad pixels. Whole pixels only, because interpolating (zoom, rotation)
    would blur the training targets. No flips: they put the heart on the wrong side."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")[:, 0]
    dy, dx = torch.randint(0, 2 * pad + 1, (2, n), device=x.device)
    rows = torch.arange(h, device=x.device) + dy[:, None]
    cols = torch.arange(w, device=x.device) + dx[:, None]
    return x[torch.arange(n, device=x.device)[:, None, None], rows[:, :, None], cols[:, None, :]].unsqueeze(1)


def train(args):
    device = get_device()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train_x, val_x, test_x = load_cache(args)
    print(f"[train] {len(train_x)} train / {len(val_x)} validation / {len(test_x)} test images, "
          f"{args.image_size}px, device={device}")

    loader = DataLoader(TensorDataset(train_x), args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x), args.batch_size)   # small batches keep the perceptual net in memory
    model = build_model(plain_args(args)).to(device)
    print(f"[train] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    perceptual = VGGPerceptual().to(device) if args.perceptual else None

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader), eta_min=args.lr * 1e-2)
    warmup_steps = args.kl_warmup * len(loader)
    z_fixed = torch.randn(8, args.latent_dim, device=device)
    history, progress, best, step = [], [], float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, sums = time.time(), torch.zeros(3, dtype=torch.float64)
        for (xb,) in loader:
            x = xb.to(device).float().div_(255)
            if args.augment:
                x = random_shift(x)
            beta = args.beta * min(1.0, step / warmup_steps) if warmup_steps else args.beta  # linear KL warm-up
            logits, mu, logvar = model(x)
            rec, kl = negative_elbo(logits, x, mu, logvar, args.likelihood, args.sigma)
            loss = rec + beta * kl
            if perceptual:
                loss = loss + args.perceptual * perceptual(torch.sigmoid(logits), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            step += 1
            sums += torch.tensor([loss.item(), rec.item(), kl.item()])
        tr_loss, tr_rec, tr_kl = (sums / len(loader)).tolist()
        val_rec, val_kl = evaluate_elbo(model, val_loader, device, args.likelihood, args.sigma)

        row = dict(epoch=epoch, beta=beta, lr=sched.get_last_lr()[0], time=time.time() - t0,
                   train_loss=tr_loss, train_rec=tr_rec, train_kl=tr_kl, train_nelbo=tr_rec + tr_kl,
                   validation_rec=val_rec, validation_kl=val_kl,
                   validation_nelbo=val_rec + val_kl)
        score = row["validation_nelbo"]     # checkpoint selection uses the loss that is trained
        if perceptual:
            row["validation_perceptual"] = evaluate_perceptual(model, val_loader, device, perceptual)
            score += args.perceptual * row["validation_perceptual"]
        history.append(row)
        print(f"[train] epoch {epoch:3d}/{args.epochs}  beta={beta:.2f}  "
              f"train -ELBO={row['train_nelbo']:8.1f} (rec {tr_rec:8.1f}, KL {tr_kl:6.1f})  "
              f"validation -ELBO={row['validation_nelbo']:8.1f} (rec {val_rec:8.1f}, KL {val_kl:6.1f})  "
              f"{row['time']:.0f}s", flush=True)

        if epoch in PROGRESS_EPOCHS or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                progress.append(torch.sigmoid(model.decode(z_fixed)).cpu())
        ckpt = {"model": model.state_dict(), "args": plain_args(args), "epoch": epoch,
                "validation_nelbo": row["validation_nelbo"]}
        save_checkpoint(ckpt, out / "vae_last.pt")
        if score < best:
            best = score
            save_checkpoint(ckpt, out / "vae_best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    save_image(torch.cat(progress), out / "progress.png", nrow=8, padding=2, pad_value=1.0)
    plot_history(history, out)
    print(f"[train] done, best validation {'loss' if perceptual else '-ELBO'} = {best:.1f}")


def save_checkpoint(obj, path, retries=5):
    # write to a temp file and rename; retry because synced folders (OneDrive) sometimes
    # lock the file for a moment, and a 1-hour run should not die because of that
    tmp = path.with_suffix(".tmp")
    for _ in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError) as e:
            print(f"[train] could not write {path.name} ({e}), retrying")
            time.sleep(2)
    print(f"[train] WARNING: {path.name} was not saved")


def plot_history(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    # 5.5in = NeurIPS text width, so the figure is included at 100% and the fonts stay readable
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.5))
    panels = [("nelbo", r"$-$ELBO (nats / image)"), ("rec", "reconstruction NLL"),
              ("kl", r"KL$(q(z|x)\,\|\,p(z))$")]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(ep, [h[f"train_{key}"] for h in history], color="#2a78d6", lw=1.4, label="train")
        ax.plot(ep, [h[f"validation_{key}"] for h in history], color="#eb6834", lw=1.4, label="validation")
        if key == "kl":
            ax.set_yscale("log")
        elif len(history) > 10:   # the first epochs dominate the range, so zoom in on epoch >= 5
            vals = [h[f"{s}_{key}"] for h in history[4:] for s in ("train", "validation")]
            ax.set_ylim(min(vals) - 0.1 * (max(vals) - min(vals)), max(vals) + 0.1 * (max(vals) - min(vals)))
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(which="both", labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "loss_curves.pdf")
    fig.savefig(out / "loss_curves.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    prepare(args)
    train(args)


if __name__ == "__main__":
    main()

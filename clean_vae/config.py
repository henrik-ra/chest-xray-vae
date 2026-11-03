import argparse
from pathlib import Path

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plain_args(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def parse_args():
    # the defaults are the final model of the report
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=None, help="extracted chest_xray folder (skips kagglehub)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/final"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--preprocess", choices=["resize", "crop", "eq", "crop_eq"], default="resize",
                   help="squeeze to a square, center crop, histogram equalization, or crop + equalization")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--res-blocks", type=int, default=2, help="residual blocks per resolution (0 = plain DCGAN)")
    p.add_argument("--dec-res-blocks", type=int, default=None, help="residual blocks per decoder stage (default: --res-blocks)")
    p.add_argument("--upsample", choices=["transpose", "conv", "shuffle"], default="conv", help="decoder upsampling")
    p.add_argument("--norm", choices=["group", "batch"], default="group")
    p.add_argument("--dec-channels", type=int, default=1, help="width multiplier of the decoder")
    p.add_argument("--perceptual", type=float, default=1440.0, help="weight of the VGG perceptual loss (0 = off)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random shifts of up to 4 pixels during training")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--likelihood", choices=["bernoulli", "gaussian"], default="gaussian")
    p.add_argument("--sigma", type=float, default=0.25, help="std of the Gaussian likelihood (pixels in [0, 1])")
    p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term after warm-up")
    p.add_argument("--kl-warmup", type=int, default=10, help="epochs of linear KL warm-up (0 = off)")
    p.add_argument("--clip", type=float, default=10.0, help="gradient-norm clipping")
    p.add_argument("--checkpoint", default="vae_best.pt")
    p.add_argument("--n-eval", type=int, default=2000, help="#images for IS/FID against the training images")
    p.add_argument("--eval-repeats", type=int, default=1, help="#sample draws for IS/FID mean +- std")
    p.add_argument("--skip-reference", action=argparse.BooleanOptionalAction, default=True,
                   help="quick mode, only pool3 FID and IS of the generated images")
    p.add_argument("--validation-only", action="store_true",
                   help="only metrics against the validation images (model selection)")
    p.add_argument("--gmm-components", type=int, default=10,
                   help="also sample z from a GMM fitted to the training posteriors (0 = off)")
    p.add_argument("--runs", nargs="+", default=[], help="evaluate: compare these run directories instead")
    p.add_argument("--labels", nargs="+", default=None, help="evaluate: names of the runs in the figures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args()

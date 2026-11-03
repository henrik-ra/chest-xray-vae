"""Validation-only comparison of the training setups of the report. The test images are not used."""

import argparse
import subprocess
import sys
from pathlib import Path

# each setup adds to the one before; the last one is the final model (the defaults of config.py)
SETUPS = {
    "pixel_sigma0.1": ["--epochs", "60", "--sigma", "0.1", "--res-blocks", "0", "--perceptual", "0", "--no-augment",
                       "--upsample", "transpose"],
    "sigma0.25": ["--epochs", "60", "--res-blocks", "0", "--perceptual", "0", "--no-augment", "--upsample", "transpose"],
    "res_blocks": ["--epochs", "60", "--perceptual", "0", "--no-augment", "--upsample", "transpose"],
    "perceptual480_shifts": ["--epochs", "60", "--perceptual", "480", "--upsample", "transpose"],
    "epochs160": ["--perceptual", "480", "--upsample", "transpose"],
    "perceptual1440": ["--upsample", "transpose"],
    "final": [],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-root", type=Path, default=Path("runs/experiments"))
    args = p.parse_args()
    data = ["--data-dir", str(args.data_dir)] + (["--data-root", str(args.data_root)] if args.data_root else [])
    for name, extra in SETUPS.items():
        out = ["--output-dir", str(args.output_root / name)]
        subprocess.run([sys.executable, "-m", "clean_vae.train", *extra, *data, *out], check=True)
        subprocess.run([sys.executable, "-m", "clean_vae.evaluate", *extra, *data, *out, "--validation-only"],
                       check=True)


if __name__ == "__main__":
    main()

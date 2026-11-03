import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

CLASSES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]


def cache_path(args):
    pre = getattr(args, "preprocess", "resize")
    return args.data_dir / f"chest_xray_{args.image_size}{'' if pre == 'resize' else '_' + pre}.pt"


def find_split_root(root):
    """The Kaggle archive contains chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
    (sometimes nested one level deeper); return the directory holding the splits."""
    cands = [p.parent for p in Path(root).rglob("train") if (p / "NORMAL").is_dir()]
    if not cands:
        raise FileNotFoundError(f"no train/NORMAL folder below {root}")
    return min(cands, key=lambda p: len(p.parts))


def load_gray(path, size, preprocess="resize"):
    img = Image.open(path)
    # The X-rays are ~1500x1200 JPEGs. draft() lets libjpeg decode directly at a
    # reduced scale (>= 2x the target), which is several times faster than a full
    # decode; the final antialiased resize then brings it to size x size.
    img.draft("L", (2 * size, 2 * size))
    img = img.convert("L")
    if "crop" in preprocess:     # the images are ~1.4x wider than high: center crop instead of squeezing
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.LANCZOS)
    if "eq" in preprocess:       # histogram equalization, makes the ribs stand out more
        img = ImageOps.equalize(img)
    return np.asarray(img, dtype=np.uint8)


def prepare(args):
    cache = cache_path(args)
    if cache.exists():
        print(f"[prepare] using cached tensors {cache}")
        return
    if args.data_root is None:
        import kagglehub
        args.data_root = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    root = find_split_root(args.data_root)
    print(f"[prepare] dataset at {root}")

    out = {"image_size": args.image_size, "classes": CLASSES}
    with Pool(os.cpu_count()) as pool:
        for split in SPLITS:
            paths, labels = [], []
            for y, cls in enumerate(CLASSES):
                files = sorted(p for p in (root / split / cls).iterdir()
                               if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
                paths += files
                labels += [y] * len(files)
            t0 = time.time()
            imgs = pool.map(partial(load_gray, size=args.image_size, preprocess=args.preprocess), paths, chunksize=16)
            out[split] = {"x": torch.from_numpy(np.stack(imgs)).unsqueeze(1), "y": torch.tensor(labels)}
            print(f"[prepare] {split}: {len(paths)} images ({labels.count(0)} normal / "
                  f"{labels.count(1)} pneumonia) in {time.time() - t0:.1f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)


def load_cache(args):
    d = torch.load(cache_path(args), weights_only=False)
    x = d["train"]["x"]
    # The official val split has only 16 images, far too few to pick a checkpoint. So 10% of the
    # training images (fixed split) are held out for validation as well. The test split is never
    # used for any training decision, only as a real-vs-real reference for FID.
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(1234))
    n_val = len(x) // 10
    val_x = torch.cat([x[perm[:n_val]], d["val"]["x"]])
    return x[perm[n_val:]], val_x, d["test"]["x"]

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from .config import get_device, parse_args
from .dataset import load_cache, prepare
from .model import build_model

# Comparison figures use the SAME test images and the SAME latent noise in every run, so figures
# of different models (Bernoulli vs Gaussian likelihood, different beta, ...) can be compared 1:1.
FIXED_TEST = (17, 205, 430)                     # indices into the 624 test images
ZOOM = ((0.22, 0.60), (0.52, 0.90))             # (rows, cols) as fractions: one lung field with ribs


@torch.no_grad()
def make_figures(model, train_x, test_x, device, out):
    model.eval()
    grid = dict(nrow=8, padding=2, pad_value=1.0)
    save_image(model.sample(32, device), out / "samples.png", **grid)
    save_image(train_x[torch.randperm(len(train_x))[:64]].float() / 255, out / "real_samples.png", **grid)

    x = test_x[torch.randperm(len(test_x))[:8]].to(device).float() / 255
    mu, _ = model.encode(x)
    rec = torch.sigmoid(model.decode(mu))
    # rows alternate: 4 originals, their 4 reconstructions, 4 originals, 4 reconstructions
    save_image(torch.cat([x[:4], rec[:4], x[4:], rec[4:]]), out / "reconstructions.png",
               nrow=4, padding=2, pad_value=1.0)

    # linear interpolation between the posterior means of two held-out images
    w = torch.linspace(0, 1, 8, device=device).unsqueeze(1)
    rows = [torch.sigmoid(model.decode((1 - w) * mu[2 * i] + w * mu[2 * i + 1])) for i in range(4)]
    save_image(torch.cat(rows), out / "interpolations.png", **grid)


def _zoom(img):
    h, w = img.shape[-2:]
    (r0, r1), (c0, c1) = ZOOM
    return img[..., int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def _show(ax, img, title=None, **kw):
    ax.imshow(img, **({"cmap": "gray", "vmin": 0, "vmax": 1} | kw))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@torch.no_grad()
def reconstruct(model, x):
    return torch.sigmoid(model.decode(model.encode(x)[0]))


@torch.no_grad()
def make_compare_figures(model, train_x, test_x, device, out, gmm_sample=None):
    """recon_compare.png: original | reconstruction | |error| | zoom original | zoom reconstruction
    samples_compare.png: one row each of real images, prior samples and GMM samples"""
    plt = _plt()
    model.eval()
    x = test_x[list(FIXED_TEST)].to(device).float() / 255
    rec = reconstruct(model, x)
    x, rec = x.cpu()[:, 0], rec.cpu()[:, 0]
    cols = ["original", "reconstruction", "|error|, 0 to 0.3", "zoom: original", "zoom: reconstruction"]
    fig, axes = plt.subplots(len(x), 5, figsize=(5.5, 1.13 * len(x)))
    for i in range(len(x)):
        imgs = [x[i], rec[i], (x[i] - rec[i]).abs(), _zoom(x[i]), _zoom(rec[i])]
        for j, img in enumerate(imgs):
            kw = {"cmap": "Blues", "vmin": 0, "vmax": 0.3} if j == 2 else {}
            _show(axes[i, j], img, cols[j] if i == 0 else None, **kw)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.3)
    fig.savefig(out / "recon_compare.png", dpi=220)
    plt.close(fig)

    g = torch.Generator().manual_seed(0)                    # same z for every run
    z = torch.randn(8, model.latent_dim, generator=g).to(device)
    rows = [("real", train_x[torch.randperm(len(train_x), generator=g)[:8]].float() / 255),
            ("prior\nN(0, I)", torch.sigmoid(model.decode(z)).cpu())]
    if gmm_sample is not None:
        torch.manual_seed(0)
        rows.append(("GMM\nlatents", gmm_sample(8).cpu()))
    fig, axes = plt.subplots(len(rows), 8, figsize=(5.5, 0.72 * len(rows)))
    for i, (label, imgs) in enumerate(rows):
        for j in range(8):
            _show(axes[i, j], imgs[j, 0])
        axes[i, 0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.15)
    fig.savefig(out / "samples_compare.png", dpi=220)
    plt.close(fig)


def summary(args):
    """Compare several finished runs (each with metrics.json + vae_best.pt):
    runs_recon.png  same test images reconstructed by every model, with zoomed crops
    runs_fid.pdf    FID on the test images of reconstructions, prior samples and GMM samples"""
    plt = _plt()
    device = get_device()
    runs = [Path(r) for r in args.runs]
    labels = args.labels or [r.name for r in runs]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _, _, test_x = load_cache(args)
    x = test_x[list(FIXED_TEST)].to(device).float() / 255

    recs = []
    for r in runs:
        ck = torch.load(r / "vae_best.pt", map_location=device, weights_only=False)
        a = ck["args"]
        m = build_model(a).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        recs.append(reconstruct(m, x).cpu()[:, 0])
    x = x.cpu()[:, 0]
    ncol = 1 + len(runs)
    fig, axes = plt.subplots(2 * len(x), ncol, figsize=(5.5, 2 * len(x) * 5.5 / ncol * 0.98))
    for i in range(len(x)):
        for j, img in enumerate([x[i]] + [r[i] for r in recs]):
            _show(axes[2 * i, j], img, (["original"] + labels)[j] if i == 0 else None)
            _show(axes[2 * i + 1, j], _zoom(img))
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    fig.savefig(out / "runs_recon.png", dpi=220)
    plt.close(fig)

    series = [("reconstructions", "recon", "#2a78d6"), ("prior samples", "prior", "#eb6834"),
              ("GMM samples", "gmm", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    ref = None
    for k, (name, key, color) in enumerate(series):
        xs, ys = [], []
        for i, r in enumerate(runs):
            t = json.loads((r / "metrics.json").read_text()).get("test", {})
            if key in t:
                xs.append(i + (k - 1) * 0.18)
                ys.append(t[key]["generated_vs_train"]["fid"])
            ref = ref or t.get("real_vs_test")
        ax.plot(xs, ys, "o", color=color, ms=6, label=name, markeredgecolor="white", markeredgewidth=1)
    if ref:
        ax.axhline(ref, color="#888888", lw=1, ls="--")
        ax.text(len(runs) - 0.5, ref, "real training vs test images", fontsize=6.5, color="#555555",
                ha="right", va="bottom")
    ax.set_xticks(range(len(runs)), labels, fontsize=7)
    ax.set_ylabel("FID vs. test images", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "runs_fid.pdf")
    fig.savefig(out / "runs_fid.png", dpi=200)
    plt.close(fig)
    print(f"[summary] wrote {out / 'runs_recon.png'} and {out / 'runs_fid.pdf'}")


@torch.no_grad()
def fit_latent_gmm(model, train_x, k, device):
    """Ex-post density estimation (Ghosh et al., 2020, "From Variational to Deterministic
    Autoencoders"): the aggregate posterior q(z) of a trained VAE does not match N(0, I), so
    prior samples often land in regions the decoder never saw. Fitting a GMM to the posterior
    means of the training images and sampling z from it uses the same decoder, only a better
    latent distribution."""
    from sklearn.mixture import GaussianMixture
    model.eval()
    mus = torch.cat([model.encode(train_x[i:i + 256].to(device).float() / 255)[0].cpu()
                     for i in range(0, len(train_x), 256)])
    gmm = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0).fit(mus.numpy())
    w = torch.tensor(gmm.weights_, dtype=torch.float32)
    means = torch.tensor(gmm.means_, dtype=torch.float32)
    chol = torch.linalg.cholesky(torch.tensor(gmm.covariances_, dtype=torch.float32))

    # sample with torch (not gmm.sample, which reseeds with random_state=0 on every call)
    @torch.no_grad()
    def sample(n):
        comp = torch.multinomial(w, n, replacement=True)
        z = means[comp] + (chol[comp] @ torch.randn(n, means.shape[1], 1)).squeeze(-1)
        return torch.sigmoid(model.decode(z.to(device)))
    return sample


def inception_metrics(fake_fn, train_x, test_x, args, device, reference=True, n=None, repeats=None):
    # fake_fn(real_batch) -> generated batch of the same size (prior samples, GMM samples or
    # reconstructions of real_batch); train_x is the real reference set the fakes are compared to
    """IS and FID with pytorch-ignite, same recipe as the ignite GAN-evaluation blog
    post. ignite's default FID uses the 1000-d softmax output of torchvision's
    Inception-v3 as 'features', which gives tiny numbers that are not comparable
    with the literature, so the standard 2048-d pool3 FID of pytorch-fid is
    computed as well (feature-extractor wrapper from the ignite FID docs)."""
    from ignite.engine import Engine, Events
    from ignite.metrics import FID, InceptionScore
    from pytorch_fid.inception import InceptionV3

    def interpolate(batch):
        # PIL bilinear resize to the 299x299 Inception input as in the blog; the
        # single gray channel is replicated to RGB.
        arr = []
        for img in batch.cpu():
            pil = transforms.ToPILImage()(img).convert("RGB").resize((299, 299), Image.BILINEAR)
            arr.append(transforms.ToTensor()(pil))
        return torch.stack(arr).to(device)

    class WrapperInceptionV3(nn.Module):
        def __init__(self, fid_incv3):
            super().__init__()
            self.fid_incv3 = fid_incv3

        @torch.no_grad()
        def forward(self, x):
            return self.fid_incv3(x)[0][:, :, 0, 0]

    pool3 = WrapperInceptionV3(InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)).eval()

    def evaluator(step_fn, names):
        ev = Engine(step_fn)
        if "fid" in names:          # pytorch-fid comparable
            FID(num_features=2048, feature_extractor=pool3, device=device).attach(ev, "fid")
        if "fid_softmax" in names:  # ignite default
            FID(device=device).attach(ev, "fid_softmax")
        if "is" in names:
            InceptionScore(device=device, output_transform=lambda o: o[0]).attach(ev, "is")
        if "is_real" in names:
            InceptionScore(device=device, output_transform=lambda o: o[1]).attach(ev, "is_real")
        ev.add_event_handler(Events.ITERATION_COMPLETED(every=10),
                             lambda e: print(f"    batch {e.state.iteration}/{e.state.epoch_length}", flush=True))
        return ev

    def generated_vs_real(engine, batch):
        real = batch[0].to(device).float().div(255)
        fake = fake_fn(real)
        return interpolate(fake), interpolate(real)

    def real_vs_real(engine, batch):
        a, b = batch
        return interpolate(a.float().div(255)), interpolate(b.float().div(255))

    bs, results = 32, {}
    n = min(n or args.n_eval or len(train_x), len(train_x))
    repeats = repeats or args.eval_repeats
    t0 = time.time()
    quick = args.skip_reference or not reference

    # IS / FID depend on which samples are drawn, so the main measurement can be repeated
    # with new samples (and a new reference subset if n < #train) to get a mean +- std.
    # Only the first draw computes the extra numbers (ignite default FID, IS of real data).
    draws = []
    for r in range(repeats):
        ref = train_x[torch.randperm(len(train_x))[:n]]
        if r == 0:
            first_ref = ref
        names = ("fid", "is") if (r > 0 or quick) else ("fid", "fid_softmax", "is", "is_real")
        print(f"[evaluate] draw {r + 1}/{repeats}: {n} generated vs {n} real images")
        draws.append(evaluator(generated_vs_real, names).run(DataLoader(TensorDataset(ref), bs), max_epochs=1).metrics)
    results["generated_vs_train"] = dict(n=n, **draws[0])
    if len(draws) > 1:
        for key in ("fid", "is"):
            v = np.array([d[key] for d in draws])
            results["generated_vs_train"].update({f"{key}_draws": v.tolist(), f"{key}_mean": float(v.mean()),
                                                  f"{key}_std": float(v.std(ddof=1))})

    # reference numbers at the size of the held-out set: how far is *real* held-out
    # data from the training set under the same metric, and how does the generator
    # do at that same n (FID is biased upwards for small n)?
    if quick:
        results["eval_time_s"] = time.time() - t0
        return results
    k = min(len(test_x), n)
    sub = first_ref[:k]
    print(f"[evaluate] {k} generated vs {k} training images")
    m = evaluator(generated_vs_real, ("fid", "fid_softmax")).run(DataLoader(TensorDataset(sub), bs), max_epochs=1).metrics
    results["generated_vs_train_small"] = dict(n=k, **m)
    print(f"[evaluate] {k} held-out vs {k} training images (real vs real)")
    m = evaluator(real_vs_real, ("fid", "fid_softmax")).run(DataLoader(TensorDataset(test_x[:k], sub), bs),
                                                            max_epochs=1).metrics
    results["test_vs_train"] = dict(n=k, **m)
    results["eval_time_s"] = time.time() - t0
    return results


@torch.no_grad()
def test_metrics(model, train_x, test_x, args, device, prior, gmm_sample=None):
    """Everything on the 624 test images: reconstruction MSE (posterior mean, pixels in [0, 1]),
    IS/FID of the reconstructions against the originals, and IS/FID of prior (and GMM) samples
    against the same test images, so reconstruction and generation are measured on the same
    footing. real_vs_test (real training images vs test images, same n) is the floor for FID."""
    model.eval()
    sq_err = 0.0
    for (xb,) in DataLoader(TensorDataset(test_x), 128):
        x = xb.to(device).float() / 255
        sq_err += F.mse_loss(torch.sigmoid(model.decode(model.encode(x)[0])), x, reduction="sum").item()
    recon = lambda real: torch.sigmoid(model.decode(model.encode(real)[0]))  # noqa: E731
    n = len(test_x)
    print("[evaluate] reconstructions of the test images")
    out = {"recon_mse": sq_err / test_x.numel(),
           "recon": inception_metrics(recon, test_x, test_x, args, device, reference=False, n=n, repeats=1)}
    print("[evaluate] prior samples against the test images")
    out["prior"] = inception_metrics(prior, test_x, test_x, args, device, reference=False, n=n)
    if gmm_sample is not None:
        print("[evaluate] GMM samples against the test images")
        out["gmm"] = inception_metrics(lambda real: gmm_sample(len(real)), test_x, test_x, args, device,
                                       reference=False, n=n)
    print("[evaluate] real training images against the test images (reference)")
    real = lambda r: train_x[torch.randint(len(train_x), (len(r),))].to(device).float() / 255  # noqa: E731
    out["real_vs_test"] = inception_metrics(real, test_x, test_x, args, device, reference=False, n=n,
                                            repeats=1)["generated_vs_train"]["fid"]
    return out


def evaluate(args):
    device = get_device()
    out = args.output_dir
    ckpt = torch.load(out / args.checkpoint, map_location=device)
    a = ckpt["args"]
    model = build_model(a).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_nelbo = ckpt.get("validation_nelbo", ckpt.get("heldout_nelbo"))   # older checkpoints
    print(f"[evaluate] loaded {args.checkpoint} (epoch {ckpt['epoch']}, validation -ELBO {val_nelbo:.1f})")

    train_x, val_x, test_x = load_cache(args)
    if args.validation_only:
        # model selection: same metrics as test_metrics, but against the validation images, so the
        # test images are not touched until the final model is chosen
        prior = lambda real: model.sample(len(real), device)  # noqa: E731
        gmm_sample = fit_latent_gmm(model, train_x, args.gmm_components, device) if args.gmm_components else None
        results = test_metrics(model, train_x, val_x, args, device, prior, gmm_sample)
        results["checkpoint"] = dict(file=args.checkpoint, epoch=ckpt["epoch"], validation_nelbo=val_nelbo)
        (out / "validation_metrics.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return
    make_figures(model, train_x, test_x, device, out)
    print("[evaluate] latent z ~ N(0, I)")
    prior = lambda real: model.sample(len(real), device)  # noqa: E731
    results = {"prior": inception_metrics(prior, train_x, test_x, args, device)}
    gmm_sample = None
    if args.gmm_components:
        print(f"[evaluate] latent z ~ GMM({args.gmm_components}) fitted to the training posteriors")
        gmm_sample = fit_latent_gmm(model, train_x, args.gmm_components, device)
        save_image(gmm_sample(32), out / "samples_gmm.png", nrow=8, padding=2, pad_value=1.0)
        results["gmm"] = inception_metrics(lambda real: gmm_sample(len(real)), train_x, test_x, args, device,
                                           reference=False)
    make_compare_figures(model, train_x, test_x, device, out, gmm_sample)
    results["test"] = test_metrics(model, train_x, test_x, args, device, prior, gmm_sample)
    results["checkpoint"] = dict(file=args.checkpoint, epoch=ckpt["epoch"], validation_nelbo=val_nelbo)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    prepare(args)
    if args.runs:
        summary(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

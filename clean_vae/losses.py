import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def negative_elbo(logits, x, mu, logvar, likelihood="bernoulli", sigma=0.1):
    """Both terms in nats per image, averaged over the batch: reconstruction NLL and
    KL(q(z|x) || N(0, I)). The decoder mean is sigmoid(logits) in [0, 1] for both likelihoods.

    bernoulli: BCE on the gray values (the usual MNIST-style surrogate, not a normalized density
               for continuous pixels, and it barely penalizes small errors -> blurry images)
    gaussian:  N(x | sigmoid(logits), sigma^2 I) with fixed sigma, a proper density for continuous
               pixels. A small sigma makes fine detail (ribs, edges) worth encoding in z."""
    if likelihood == "gaussian":
        err = (x - torch.sigmoid(logits)) / sigma
        rec = (0.5 * err.pow(2) + math.log(sigma) + 0.5 * math.log(2 * math.pi)).flatten(1).sum(1).mean()
    else:
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
    return rec, kl


class VGGPerceptual(nn.Module):
    """Perceptual distance as in LPIPS, without the learned weights: squared difference of the
    channel-normalized VGG16 features (relu1_2, relu2_2, relu3_3), per image. A pixel loss
    averages over everything that is plausible (blur), this one penalizes missing edges."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        self.vgg.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, x):
        h, out = (x.expand(-1, 3, -1, -1) - self.mean) / self.std, []
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in (3, 8, 15):
                out.append(F.normalize(h, dim=1))
        return out

    def forward(self, a, b):
        with torch.no_grad():
            target = self.features(b)
        return sum((f - t).pow(2).sum(1).mean((1, 2)) for f, t in zip(self.features(a), target)).mean()

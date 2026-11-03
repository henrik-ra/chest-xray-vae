import math

import torch
import torch.nn as nn


def make_norm(kind, c):
    return nn.BatchNorm2d(c) if kind == "batch" else nn.GroupNorm(8, c)


class ResBlock(nn.Module):
    def __init__(self, c, norm="group"):
        super().__init__()
        self.net = nn.Sequential(make_norm(norm, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1),
                                 make_norm(norm, c), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.net(x)


class ConvVAE(nn.Module):
    """DCGAN-style encoder / decoder. Stride-2 convs take image_size -> 4x4, the
    4x4xC feature map is mapped linearly to (mu, log sigma^2) and back.

    GroupNorm instead of BatchNorm so that the ELBO of a sample does not depend on
    its mini-batch and train / eval behave identically (matters when sampling
    from the prior). The decoder returns *logits*; the Bernoulli likelihood is
    evaluated with BCE-with-logits, which is numerically safer than sigmoid + BCE.
    res_blocks > 0 adds residual blocks per resolution and a conv output layer at full
    resolution, so the last layer is not a single transposed conv any more.
    """

    def __init__(self, image_size=128, latent_dim=128, base_channels=32, res_blocks=0, dec_res_blocks=None,
                 upsample="transpose", norm="group", dec_channels=1):
        super().__init__()
        n_down = int(math.log2(image_size)) - 2                 # 64px -> 4 blocks, 128px -> 5
        chans = [base_channels * 2 ** i for i in range(n_down)]  # 32, 64, 128, 256, 512
        self.latent_dim = latent_dim
        self.feat_shape = (chans[-1], 4, 4)
        feat_dim = chans[-1] * 16

        enc, c_in = [], 1
        for c in chans:
            enc += [nn.Conv2d(c_in, c, 4, 2, 1), make_norm(norm, c), nn.LeakyReLU(0.2, inplace=True)]
            enc += [ResBlock(c, norm) for _ in range(res_blocks)]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # dec_channels widens the decoder, dec_res_blocks makes it deeper than the encoder
        dec_res = res_blocks if dec_res_blocks is None else dec_res_blocks
        dec_chans = [c * dec_channels for c in chans]
        self.feat_shape = (dec_chans[-1], 4, 4)
        self.fc_dec = nn.Linear(latent_dim, dec_chans[-1] * 16)

        def up(c_in, c_out):
            if upsample == "conv":   # nearest-neighbour upsampling + conv instead of a transposed conv
                return [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c_in, c_out, 3, 1, 1)]
            if upsample == "shuffle":   # sub-pixel convolution: conv to 4x the channels, then rearrange
                return [nn.Conv2d(c_in, 4 * c_out, 3, 1, 1), nn.PixelShuffle(2)]
            return [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)]

        dec = [ResBlock(dec_chans[-1], norm) for _ in range(dec_res)] if upsample == "conv" else []
        for c_in, c_out in zip(dec_chans[::-1], dec_chans[::-1][1:]):
            dec += up(c_in, c_out) + [make_norm(norm, c_out), nn.ReLU(inplace=True)]
            dec += [ResBlock(c_out, norm) for _ in range(dec_res)]
        if dec_res or upsample == "conv":
            dec += up(dec_chans[0], dec_chans[0]) + [make_norm(norm, dec_chans[0]), nn.ReLU(inplace=True)]
            dec += [ResBlock(dec_chans[0], norm) for _ in range(dec_res)]
            dec += [nn.Conv2d(dec_chans[0], 1, 3, 1, 1)]
        else:
            dec += [nn.ConvTranspose2d(dec_chans[0], 1, 4, 2, 1)]   # logits, no activation
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), 20.0 * torch.tanh(self.fc_logvar(h) / 20.0)   # soft limit, keeps exp(logvar) finite early in training

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, *self.feat_shape))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterisation
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        return torch.sigmoid(self.decode(torch.randn(n, self.latent_dim, device=device)))


def build_model(a):
    """Model from the arguments of a run; older checkpoints do not store the newer ones."""
    return ConvVAE(a["image_size"], a["latent_dim"], a["base_channels"], a.get("res_blocks", 0),
                   a.get("dec_res_blocks"), a.get("upsample", "transpose"), a.get("norm", "group"),
                   a.get("dec_channels", 1))

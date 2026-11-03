# ECE 285 Homework 1: VAE for chest X-ray generation

Convolutional VAE trained on the Kaggle chest X-ray pneumonia dataset, evaluated with
Inception Score and FID (pytorch-ignite, with the standard pool3 FID via pytorch-fid).

```
pip install -r requirements.txt
python -m clean_vae.train --data-root <path/to/chest_xray>
python -m clean_vae.evaluate
python -m clean_vae.experiments --data-root <path/to/chest_xray>
```

Without `--data-root`, the dataset is downloaded with kagglehub (Kaggle access needed). The first
runs also download the VGG16 and Inception-v3 weights. A GPU is recommended; training took about
15 s per epoch on an NVIDIA A30.

The default arguments are the final model of the report: grayscale images resized to 128 x 128,
Gaussian likelihood with sigma 0.25, two residual blocks per resolution, a decoder that upsamples
with nearest neighbour + 3x3 convolution, perceptual loss with weight 1440, random shifts, 160 epochs, batch size 32, Adam with learning rate 5e-4 and cosine
decay, beta 1 with a 10-epoch warm-up, seed 0.

## Files

| File | Content |
|---|---|
| `clean_vae/config.py` | command-line arguments of `train` and `evaluate` |
| `clean_vae/dataset.py` | data preparation, cached tensors, train/validation/test split |
| `clean_vae/model.py` | convolutional VAE with optional residual blocks and decoder variants |
| `clean_vae/losses.py` | negative ELBO and VGG perceptual loss |
| `clean_vae/train.py` | training, loss curves, checkpoints |
| `clean_vae/evaluate.py` | figures, Inception Score, FID, GMM sampling |
| `clean_vae/experiments.py` | the setups of Table 3, compared on the validation images |

## Options

- **Preprocessing** (`--preprocess`): `resize` (default, squeezes the images to a square), `crop`
  (central square crop), `eq` (histogram equalization) or `crop_eq`. Every choice has its own cache.
- **Perceptual loss** (`--perceptual`): weight of the VGG16 feature distance (relu1_2, relu2_2,
  relu3_3), added to the negative ELBO. 0 turns it off.
- **Likelihood** (`--likelihood gaussian --sigma`): Gaussian decoder with fixed sigma; `bernoulli`
  is also available.
- **Model** (`--res-blocks`, `--latent-dim`, `--base-channels`).
- **Decoder** (`--upsample conv|transpose|shuffle`, `--norm group|batch`, `--dec-channels`,
  `--dec-res-blocks`): upsampling layer, normalization, width and depth of the decoder.
- **Augmentation** (`--augment` / `--no-augment`): random shifts of up to 4 whole pixels.

Use the same `--preprocess` and `--data-dir` for `train` and `evaluate`. `--validation-only`
evaluates on the validation images, so the test images are not touched during model selection.

## Outputs

`--output-dir` (default `runs/final`) receives `history.json`, `loss_curves.pdf`, `samples.png`,
`samples_gmm.png`, `reconstructions.png`, `interpolations.png`, `recon_compare.png`,
`samples_compare.png`, `metrics.json` and the checkpoints `vae_best.pt` and `vae_last.pt`.

The splits are 4,695 training, 537 validation (10% of the training images plus the 16 official
validation images) and 624 test images. Samples come from the N(0, I) prior and from a
10-component Gaussian mixture fitted to the training posterior means (`--gmm-components`).

The tables of the report use several sample draws per model; `evaluate` computes one draw.

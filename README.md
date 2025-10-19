# SatGAN: Satellite Image Stitching with GANs
## Overview

SatGAN uses Generative Adversarial Networks (GANs) to automatically stitch satellite imagery into panoramas.
This approach outperforms traditional methods like SIFT-based feature matching in speed and efficiency.

## Features

Deep Learning: Custom PyTorch GANs with UNet-style generators and spectral-normalized discriminators

Computer Vision: Baseline SIFT + homography stitching for comparison

Pipeline: Data preprocessing → model training → evaluation → visualization

Evaluation: MSE, PSNR, and timing benchmarks

Engineering: Modular code, checkpointing, and resumption support

## Technical Details

### Model

Generator: UNet-style encoder–decoder

Discriminator: CNN with spectral normalization

### Data

Source: USGS satellite imagery (Philadelphia)

512×512 tiles, train/test split, padding & masking augmentation

### Training

Mixed precision (PyTorch AMP)

Gradient accumulation

Custom dataset loaders

## Results

GAN stitching significantly faster than SIFT-based methods

Improved PSNR and visual consistency across epochs

Detailed loss and timing analysis

## Skills Demonstrated

GAN architecture design & training

Model evaluation and benchmarking

Software engineering for ML pipelines

Image preprocessing and augmentation

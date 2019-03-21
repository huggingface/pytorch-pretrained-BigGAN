# PyTorch pretrained BigGAN
An op-for-op PyTorch reimplementation of DeepMind's BigGAN model with pre-trained models

## Introduction

This repository contains an op-for-op PyTorch reimplementation of DeepMind's BigGAN that was released with the paper [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://openreview.net/forum?id=B1xsqj09Fm) by Andrew Brocky, Jeff Donahuey and Karen Simonyan.
This PyTorch implementation of BigGAN is provided with the [pretrained 128, 256 and 512 pixels resolution models  from DeepMind](https://tfhub.dev/deepmind/biggan-deep-128/1) and the command-line interface used to convert these models from the TensorFlow Hub models.

## Installation

This repo was tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 0.4.1/1.0.0

PyTorch pretrained BigGAN can be installed by pip as follows:
```bash
pip install pytorch-pretrained-biggan
```

If you simply want to play with the GAN this should be enough.

If you want to use the conversion scripts and the imagenet utilities, additional requirements are needed, in particular TensorFlow and NLTK. To install all the requirements please use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Models

This repository provide direct and simple access to the pretrained "deep" versions of BigGAN for 128, 256 and 512 pixels resolutions as described in the [associated publication](https://openreview.net/forum?id=B1xsqj09Fm).
Here are some details on the models:

- `BigGAN-deep-128` is a 50.4M parameters model, the model dump weights 210Mb,
- `BigGAN-deep-256` is a 50.4M parameters model, the model dump weights 210Mb
- `BigGAN-deep-512` is a 50.4M parameters model, the model dump weights 210Mb

All models comprise pre-computed batch norm statistics for 51 truncation values between 0 and 1.

## Usage

Here is a quick-start example using `BigGAN` with a pre-trained model.
See the [doc section](#doc) below for all the details on these classes.

```python
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_name, truncated_noise_sample, save_as_images

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')

# Prepare a dogball input
truncation = 0.4
class_vector = (one_hot_from_name('tennis ball') + one_hot_from_name('chihuahua')) / 2
noise_vector = truncated_noise_sample(truncation=truncation)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

# Generate a dog ball
dogball = model(noise_vector, class_vector, truncation)

# Print results
```

## Doc

## Conversion script

A script that can be used to convert models from TensorFlow Hub is provided in [./scripts/convert_tf_hub_models.sh](./scripts/convert_tf_hub_models.sh).

The script can be used directly as:
```bash
./scripts/convert_tf_hub_models.sh
```

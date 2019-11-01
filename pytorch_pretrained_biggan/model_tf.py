# coding: utf-8
""" BigGAN TF 2.0 model.
    From "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
    By Andrew Brocky, Jeff Donahuey and Karen Simonyan.
    https://openreview.net/forum?id=B1xsqj09Fm

    TF 2.0 version implemented from the computational graph of the TF Hub module for BigGAN.
    Some part of the code are adapted from https://github.com/brain-research/self-attention-gan

    This version only comprises the generator (since the discriminator's weights are not released).
    This version only comprises the "deep" version of BigGAN (see publication).
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import logging
import math

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BigGANConfig
from .file_utils import cached_path
from .utils import truncated_noise_sample, save_as_images, one_hot_from_names

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'biggan-deep-128': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
    'biggan-deep-256': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
    'biggan-deep-512': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
}

WEIGHTS_NAME = 'pytorch_model.bin'


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    From https://medium.com/@FloydHsiu0618/spectral-normalization-implementation-of-tensorflow-2-0-keras-api-d9060d26de77
    """

    def __init__(self, layer, eps=1e-12, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.eps = eps

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        _u = tf.identity(self.u)
        _v = tf.matmul(_u, tf.transpose(w_reshaped))
        _v = _v / tf.maximum(tf.reduce_sum(_v**2)**0.5, self.eps)
        _u = tf.matmul(_v, w_reshaped)
        _u = _u / tf.maximum(tf.reduce_sum(_u**2)**0.5, self.eps)

        self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        self.layer.kernel = self.w / sigma

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


def snconv2d(eps=1e-12, **kwargs):
    return SpectralNormalization(tf.keras.layers.Conv2D(**kwargs), eps=eps)

def snlinear(eps=1e-12, **kwargs):
    return SpectralNormalization(tf.keras.layers.Dense(**kwargs), eps=eps)

def sn_embedding(eps=1e-12, **kwargs):
    return SpectralNormalization(tf.keras.layers.Embedding(**kwargs), eps=eps)


class SelfAttn(tf.keras.layers.Layer):
    """ Self attention Layer"""
    def __init__(self, in_channels, eps=1e-12, **kwargs):
        super(SelfAttn, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps, name='snconv1x1_theta')
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps, name='snconv1x1_phi')
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps, name='snconv1x1_g')
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps, name='snconv1x1_o_conv')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

    def build(self, input_shape):
        self.gamma = self.add_variable(
            shape=(1,),
            initializer=tf.keras.initializers.zeros(),
            name='gamma',
            trainable=True,
            dtype=tf.float32)

        super(SelfAttn, self).build()

    def call(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = tf.reshape(theta, (-1, ch//8, h*w))
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = tf.reshape(phi, (-1, ch//8, h*w//4))
        # Attn map
        attn = tf.matmul(theta, phi, transpose_b=True)  # equivalent to torch bmm(theta.permute(0, 2, 1), phi)
        attn = tf.nn.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = tf.reshape(g, (-1, ch//2, h*w//4))
        # Attn_g - o_conv
        attn_g = tf.matmul(attn, g)  # equivalent to torch bmm(g, attn.permute(0, 2, 1))
        attn_g = tf.reshape(attn_g, (-1, ch//2, h, w))
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma*attn_g
        return out


class BigGANBatchNorm(tf.keras.layers.Layer):
    """ This is a batch norm module that can handle conditional input and can be provided with pre-computed
        activation means and variances for various truncation parameters.

        We computate batch_norm our-self without updating running means and variances.
        If you want to train this model you should add running means and variance computation logic.
    """
    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True, **kwargs):
        super(BigGANBatchNorm, self).__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional
        self.n_stats = n_stats
        self.num_features = num_features

        # We use pre-computed statistics for n_stats values of truncation between 0 and 1
        self.step_size = 1.0 / (n_stats - 1)

        if self.conditional:
            assert condition_vector_dim is not None
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False,
                                  eps=eps, name='scale')
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False,
                                   eps=eps, name='offset')

    def build(self, input_shape):
        self.running_means = self.add_variable(
            shape=(self.n_stats, self.num_features),
            initializer=tf.keras.initializers.zeros(),
            name='running_means',
            trainable=False,
            dtype=tf.float32)
        self.running_vars = self.add_variable(
            shape=(self.n_stats, self.num_features),
            initializer=tf.keras.initializers.ones(),
            name='running_vars',
            trainable=False,
            dtype=tf.float32)

        if not self.conditional:
            self.weight = self.add_variable(
                shape=(self.num_features,),
                initializer=tf.keras.initializers.ones(),
                name='weight',
                trainable=False,
                dtype=tf.float32)
            self.bias = self.add_variable(
                shape=(self.num_features,),
                initializer=tf.keras.initializers.zeros(),
                name='bias',
                trainable=False,
                dtype=tf.float32)

        super(BigGANBatchNorm, self).build()

    def call(self, x, truncation, condition_vector=None):
        # Retreive pre-computed statistics associated to this truncation
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)
        if coef != 0.0:  # Interpolate
            running_mean = self.running_means[start_idx] * coef + self.running_means[start_idx + 1] * (1 - coef)
            running_var = self.running_vars[start_idx] * coef + self.running_vars[start_idx + 1] * (1 - coef)
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]

        if self.conditional:
            running_mean = running_mean[None, :, None, None]  # .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            running_var = running_var[None, :, None, None]  # .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector)[..., None, None]  # .unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector)[..., None, None]  # .unsqueeze(-1).unsqueeze(-1)

            # out = (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias
        else:
            weight = self.weight
            bias = self.bias

        out = tf.nn.batch_normalization(
                x, running_mean, running_var, self.weight, self.bias, self.eps)

        return out


class GenBlock(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12, **kwargs):
        super(GenBlock, self).__init__(**kwargs)
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps,
                                    conditional=True, name='bn_0')
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1,
                               eps=eps, name='conv_0')

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps,
                                    conditional=True, name='bn_1')
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1,
                               eps=eps, name='conv_1')

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps,
                                    conditional=True, name='bn_2')
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1,
                               eps=eps, name='conv_2')

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps,
                                    conditional=True, name='bn_3')
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1,
                               eps=eps, name='conv_3')

    def call(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = tf.nn.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = tf.nn.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = tf.nn.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = tf.nn.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        out = x + x0
        return out

class Generator(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2

        self.gen_z = snlinear(in_features=condition_vector_dim,
                              out_features=4 * 4 * 16 * ch, eps=config.eps, name='gen_z')

        self.layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                self.layers.append(SelfAttn(ch*layer[1], eps=config.eps))
            self.layers.append(GenBlock(ch*layer[1],
                                   ch*layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=config.n_stats,
                                   eps=config.eps,
                                   name='layers_{}'.format(i)))

        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)

    def call(self, cond_vector, truncation):
        z = self.gen_z(cond_vector)

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)

        z = self.bn(z, truncation)
        z = tf.nn.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = tf.math.tanh(z)
        return z

class BigGAN(tf.keras.layers.Layer):
    """BigGAN Generator."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        config = BigGANConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            model_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)

        try:
            resolved_model_file = cached_path(model_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Wrong model name, should be a valid path to a folder containing "
                         "a {} file or a model name in {}".format(
                         WEIGHTS_NAME, PRETRAINED_MODEL_ARCHIVE_MAP.keys()))
            raise

        logger.info("loading model {} from cache at {}".format(pretrained_model_name_or_path, resolved_model_file))

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)

        # build the network with dummy inputs
        truncation = tf.constant(0.4)
        noise = tf.constant(truncated_noise_sample(batch_size=2, truncation=truncation))
        label = tf.constant(one_hot_from_names('diver', batch_size=2))

        ret = model(noise, label, truncation)

        assert os.path.isfile(resolved_model_file), "Error retrieving file {}".format(resolved_model_file)
        model.load_weights(resolved_model_file, by_name=True)

        ret = model(noise, label, truncation)  # Make sure restore ops are run

        return model

    def __init__(self, config, **kwargs):
        super(BigGAN, self).__init__(**kwargs)
        self.config = config
        self.embeddings = tf.keras.layers.Dense(config.num_classes, config.z_dim, bias=False)
        self.generator = Generator(config)

    def call(self, z, class_label, truncation):
        tf.debugging.assert_greater(0, truncation)
        tf.debugging.assert_less_equal(truncation, 1)

        embed = self.embeddings(class_label)
        cond_vector = tf.concat((z, embed), axis=1)

        z = self.generator(cond_vector, truncation)
        return z


if __name__ == "__main__":
    import PIL
    from .utils import truncated_noise_sample, save_as_images, one_hot_from_names
    from .convert_tf_to_pytorch import load_tf_weights_in_biggan

    load_cache = False
    cache_path = './saved_model.pt'
    config = BigGANConfig()
    model = BigGAN(config)
    if not load_cache:
        model = load_tf_weights_in_biggan(model, config, './models/model_128/', './models/model_128/batchnorms_stats.bin')
        torch.save(model.state_dict(), cache_path)
    else:
        model.load_state_dict(torch.load(cache_path))

    model.eval()

    truncation = 0.4
    noise = truncated_noise_sample(batch_size=2, truncation=truncation)
    label = one_hot_from_names('diver', batch_size=2)

    # Tests
    # noise = np.zeros((1, 128))
    # label = [983]

    noise = torch.tensor(noise, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float)
    with torch.no_grad():
        outputs = model(noise, label, truncation)
    print(outputs.shape)

    save_as_images(outputs)

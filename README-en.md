<!-- -*- coding: utf-8 -*- -->
# hed-tool

A tool for thinnig images using Holistically-Nested Edge Detection(HED, https://arxiv.org/abs/1504.06375).

## requirements

* TensorFlow
* Tensorpack
* GNU Octave

## Installation

The tool requires [GNU Octave](https://www.gnu.org/software/octave/)と[Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/).

There is some ansible playbook files for Debian/Ubuntu. On the other environments, you can refer [the Dockerfile in pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow/blob/master/docker/Dockerfile).

### HED pretrained model

Get [HED_pretrained_bsds.npz](http://models.tensorpack.com/HED/HED_pretrained_bsds.npz) file from [Tensorpack pretrained model](http://models.tensorpack.com/HED/) page.

# How to run

The following is an example.

```
python hed-tool.py --load HED_pretrained_bsds.npz -o output.png input.png
```

# License

Apache License 2.0

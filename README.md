<!-- -*- coding: utf-8 -*- -->
# hed-tool

Holistically-Nested Edge Detection(HED, https://arxiv.org/abs/1504.06375)を用いて画像の細線化を行うツールです。

(Tensorpack)[https://github.com/ppwwyyxx/tensorpack]のHED実装に基づいてツール化したものです。
HEDの出力をそのまま用いるのではなく、その結果をさらに繊細化する処理を追加しています。実装は(pix2pix-tensorflowのprocess.py)[https://github.com/affinelayer/pix2pix-tensorflow/blob/master/tools/process.py)をもとにしています。

## 必要なもの

* TensorFlow
* Tensorpack
* Octave

## インストール方法



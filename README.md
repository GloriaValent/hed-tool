<!-- -*- coding: utf-8 -*- -->
# hed-tool

Holistically-Nested Edge Detection(HED, https://arxiv.org/abs/1504.06375)を用いて画像の細線化を行うツールです。

[Tensorpack](https://github.com/ppwwyyxx/tensorpack)のHED実装に基づいてツール化したものです。
HEDの出力をそのまま用いるのではなく、その結果をさらに繊細化する処理を追加しています。実装は[pix2pix-tensorflowのprocess.py](https://github.com/affinelayer/pix2pix-tensorflow/blob/master/tools/process.py)をもとにしています。

## 必要なもの

* TensorFlow
* Tensorpack
* GNU Octave

## インストール方法

[GNU Octave](https://www.gnu.org/software/octave/)と[Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/)他が必要です。

Debian/Ubuntu向けにansible playbookを用意してあります。それ以外の環境については、[Qiitaの解説記事](https://qiita.com/knok/items/c3bbff0597b3158ec31c#piotr-toolbox%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB]あるいは[pix2pix-tensorflowのDockerfile](https://github.com/affinelayer/pix2pix-tensorflow/blob/master/docker/Dockerfile)を参考にしてください。

### HED事前訓練モデル

[Tensorpack pretrained model](http://models.tensorpack.com/HED/)のページから[HED_pretrained_bsds.npz](http://models.tensorpack.com/HED/HED_pretrained_bsds.npz)をダウンロードしてください。

# 実行

以下のように実行してください。

```
python hed-tool.py --load HED_pretrained_bsds.npz -o output.png input.png
```

## オプション解説

* --load 訓練済みモデル(npzファイル)の指定
* --output / -o 出力ファイル名
* --dir 入力ディレクトリ(ディレクトリ上にあるファイルをまとめて処理する)
* --skip-postprocess / -s Octaveによる細線化処理を行わない

# ライセンス

tensorpackのライセンスに従いApache License 2.0とします。


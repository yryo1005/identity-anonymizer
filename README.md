# identity-anonymizer

[GHOST](https://github.com/ai-forever/ghost)による顔交換と，差し替え可能な匿名化モデル
(`Anonymizer`)を組み合わせて，動画・画像中の人物の顔を，属性(性別・年齢等)や表情を保持したまま
架空の人物へ匿名化するパイプラインである．

現在の `Anonymizer` には，Variational Autoencoder (VAE) による実装(`VAEAnonymizer`，提案手法)
と，性別・年齢・人種の属性を入力とするNNによる実装(`AttributeNNAnonymizer`，従来手法2)の
2種類がある．出力は常にArcFaceの512次元顔ベクトルに固定されるが，入力の形式はモデルごとに
自由に設計できる(新しいモデルの実装方法は [document.md](document.md) を参照)．

## Colabで試す

ローカルでのセットアップなしに，Google Colab上で匿名化パイプラインを試すことができる．
[demo.ipynb](demo.ipynb) を開くと，Python環境のセットアップから画像・動画1本ずつの匿名化まで
一通り実行できる(GPUランタイムが必要．メニューの `ランタイム > ランタイムのタイプを変更` で
GPUを選択すること)．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/demo.ipynb)

## リポジトリ構成

```text
identity-anonymizer/
├── third_party/ghost/        # GHOST本体(git submodule, upstreamをそのまま参照)
├── patches/                  # GHOSTへの最小限の変更(パッチファイルとして管理)
├── scripts/                  # セットアップ・重みダウンロード用スクリプト
├── weights/                  # 学習済み重み(Git管理外，スクリプトで取得)
├── src/identity_anonymizer/  # 本プロジェクト独自のPythonパッケージ
│   ├── anonymizers/          # 差し替え可能な匿名化モデルのインターフェースと実装(VAE等)
│   ├── faceswap/             # GHOSTのモデルロードと，匿名化パイプライン本体
│   ├── data/                 # UTKFaceからの顔ベクトル抽出
│   └── evaluation/           # 年齢/性別/コサイン類似度の並列評価
├── notebooks/                # データセット構築・学習・推論・評価用のnotebook
├── sample_images/, sample_videos/  # デモ用のサンプルデータ
├── tests/                    # pytestによる単体テスト(GPU不要な純粋ロジックが対象)
├── demo.ipynb                # Colab上で環境構築から匿名化までを一通り試せるデモnotebook
├── document.md                # Anonymizerサブクラスの実装ガイド(拡張方法)
└── README.md                  # 本ファイル(セットアップ・実行方法)
```

## セットアップ

### 1. Python環境

`mxnet-cu112` / `insightface==0.2.1` / `torch`(CUDAビルド) の組み合わせはビルド済みwheelへの
依存が強く，`uv`等でのゼロからの再現が難しいため，`conda`で環境を構築する．

```bash
conda env create -f environment.yml
conda activate identity-anonymizer
```

Jupyterでこの環境を使う場合は，カーネルとして登録しておく．

```bash
python -m ipykernel install --user --name identity-anonymizer --display-name "Python (identity-anonymizer)"
```

### 2. GHOST submoduleの取得・パッチ適用・重みのダウンロード

以下を一括で行う場合:

```bash
bash scripts/setup.sh
```

個別に行う場合:

```bash
# GHOST本体の取得
git submodule update --init third_party/ghost

# GHOSTへの最小限の修正を適用(内容は patches/0001-*.patch を参照)
git apply --directory third_party/ghost patches/0001-silence-inner-progress-bars.patch

# GHOST/ArcFace由来の重みを取得
bash scripts/download_ghost_weights.sh

# 本リポジトリ独自の重み(DEXの年齢/性別推定モデル，学習済みVAE)を取得
bash scripts/download_release_weights.sh
```

`scripts/download_release_weights.sh` は，[本リポジトリのGitHub Release
(v1.0.0)](https://github.com/yryo1005/identity-anonymizer/releases/tag/v1.0.0)から
以下の重みを取得する．別のRelease先を使う場合は，`RELEASE_OWNER` / `RELEASE_REPO` /
`RELEASE_TAG` 環境変数で上書きすること．

| ファイル | 配置先 |
| :--- | :--- |
| `age_sd.pth` | `weights/dex/age_sd.pth` |
| `gender_sd.pth` | `weights/dex/gender_sd.pth` |
| `vae_512_128.pt` | `weights/anonymizers/vae/vae_512_128.pt` |

### 3. パッケージのインストール

```bash
pip install -e .
```

## notebookの使い方

`notebooks/` 配下のnotebookは番号順に依存している．

| notebook | 役割 | Colab |
| :--- | :--- | :--- |
| `01_make_dataset.ipynb` | UTKFaceの顔画像からArcFace顔ベクトルを抽出し，`data/processed/` に保存する | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/01_make_dataset.ipynb) |
| `02_train_vae_anonymizer.ipynb` | `VAEAnonymizer` を学習する | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/02_train_vae_anonymizer.ipynb) |
| `03_inference_image.ipynb` | 画像1枚に対する匿名化のデモ(ノイズレベルを変えた比較を含む) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/03_inference_image.ipynb) |
| `04_inference_video.ipynb` | 動画1本に対する匿名化のデモ．動画全体で同一の匿名化後の顔ベクトルを使用し，フレーム間で人物の見た目が一貫することを確認する | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/04_inference_video.ipynb) |
| `05_evaluate.ipynb` | UTKFaceに対する年齢/性別/コサイン類似度の定量評価を並列実行する | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/05_evaluate.ipynb) |
| `06_train_attribute_nn_anonymizer.ipynb` | `AttributeNNAnonymizer`(従来手法2，属性入力型)を学習し，`VAEAnonymizer` との差し替えをデモする | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/identity-anonymizer/blob/master/notebooks/06_train_attribute_nn_anonymizer.ipynb) |

いずれのnotebookも，Colab上で開く場合は先に [demo.ipynb](demo.ipynb) と同様の環境構築
(Python 3.9 + CUDA対応ライブラリのインストール，重みのダウンロード)が必要である．

`04_inference_video.ipynb` で対象とする動画は，`sample_videos/` の中身を差し替える，または
notebook内のパスの指定を変更することで，任意の動画に変更できる．

`data/utkface_sample/` にはUTKFaceからの少数サンプル(500枚)が同梱されており，
`data/processed/` には，そのサンプルとは別に予め全件抽出した顔ベクトル(`.npy`)が同梱されている
ため，UTKFace全件をダウンロードしなくても一通りnotebookを動かすことができる．
UTKFace全件で学習・評価する場合は，[UTKFaceの配布ページ](https://susanqq.github.io/UTKFace/)
から取得し，`data/UTKFace/` に配置したうえで `01_make_dataset.ipynb` を実行すること．

## 匿名化モデルの差し替え

GHOSTは常にArcFaceの512次元顔ベクトルを「匿名化後の顔ベクトル」として要求するため，
`Anonymizer` の**出力**は形状 `(B, 512)` に固定されている(`output_dim`)．一方，**入力**は
モデルごとに異なってよい(`VAEAnonymizer` はArcFaceの顔ベクトルを，`AttributeNNAnonymizer`
は性別・年齢・人種の属性を入力とする)．

```python
from identity_anonymizer.faceswap import load_ghost_models, FaceAnonymizerPipeline
from identity_anonymizer.anonymizers import get_anonymizer

models = load_ghost_models()
anonymizer = get_anonymizer("attribute_nn", weight_path="weights/anonymizers/attribute_nn/attribute_nn.pt")
pipeline = FaceAnonymizerPipeline(models, anonymizer)  # "vae" から差し替えるだけでよい

# 入力がArcFace顔ベクトルではないモデルは，anonymizer_input で入力を明示する
face_image, _ = pipeline.anonymize_image(
    "sample_images/beckham.jpg", anonymizer_input={"age": 20, "gender": 0, "race": 0},
)
```

新しい匿名化モデルの実装方法(`validate_input` / `_anonymize` の実装，登録方法，
`AttributeNNAnonymizer` を例にした詳しい解説)は [document.md](document.md) を参照すること．

## 評価の並列実行

`identity_anonymizer.evaluation.run_evaluation_parallel` は，`multiprocessing`の
`ProcessPoolExecutor`(spawnコンテキスト)を用いて画像リストを複数プロセスへ分割し評価する．
プロセス数は，指定しない場合 `determine_worker_count` が「1プロセスで実際に1回処理した際の
DRAM/VRAM使用量」を計測したうえで，空きメモリに収まる範囲で自動的に決定する．

GPU上では複数プロセスが同一のGPUを共有するため，計算資源の競合によりプロセス数を増やしても
スループットが線形には向上しない場合がある．`determine_worker_count` はメモリ超過による
クラッシュを防ぐための上限を与えるものであり，実際の処理時間は `notebooks/05_evaluate.ipynb`
で計測すること．

`spawn` コンテキストを使用するため，`run_evaluation_parallel` および `determine_worker_count`
を呼び出すスクリプトは `if __name__ == "__main__":` の内側から呼び出す必要がある(notebookから
呼び出す場合はこの制約はない)．

## GHOSTへの変更点

GHOST本体(`third_party/ghost`)への変更は，`patches/0001-silence-inner-progress-bars.patch`
の1件のみである．動画処理中の内側のループで進捗バーが二重に表示されるのを防ぐための，
`utils/inference/video_processing.py` 内3箇所での局所的な `tqdm` の無効化にとどまる．
それ以外のGHOSTのコードは変更していない．

## ライセンス

本リポジトリは Apache License 2.0 の下で公開する．GHOST・ArcFace・DEX等の第三者ソフトウェア・
モデルの利用状況については [NOTICE](NOTICE) を参照すること．

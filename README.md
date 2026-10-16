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
├── programs/demo/            # ブラウザで動画を撮影・アップロードして匿名化するデモ(Webサーバー)
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

## ブラウザで使う匿名化デモと，外部からのアクセス (Cloudflare Tunnel)

`programs/demo/` は，ブラウザで撮影した動画やローカルの動画ファイルを受け取り，動画に映る全員の顔を
匿名化して返すWebアプリケーションである．処理はサーバー(GPUのあるこのPC)で行い，利用者の端末は
ブラウザで撮影・表示するだけである．ここでは，サーバーを起動し，Cloudflare Tunnel で外部からアクセス
できるURLを取得するまでの手順をまとめる．デモの構成は [document.md](document.md) の9章を参照すること．

ブラウザは，`localhost` 以外の `http://` で開いたページにはカメラを使わせない．Cloudflare Tunnel は
`https://` のURLを発行するため，他のPCやスマートフォンからでもカメラで撮影できる．

### 1. デモ用の環境を用意する(初回のみ)

貼り戻しにランドマーク検出器(mxnet)が必要なため，既存の環境 `identity-anonymizer` を複製し，
CPU版の mxnet とWebサーバー用のライブラリを追加した環境 `identity-anonymizer-mxcpu` を用いる．
既存の環境は変更しない．

```bash
conda create -y -n identity-anonymizer-mxcpu --clone identity-anonymizer
ENV_PIP=~/miniconda3/envs/identity-anonymizer-mxcpu/bin/pip
$ENV_PIP uninstall -y mxnet-cu112
$ENV_PIP install --no-deps mxnet==1.9.1 dill numpy==1.23.5   # mxnet 1.9.1 は numpy>=1.24 で動かない
$ENV_PIP install -r programs/demo/requirements_demo.txt     # fastapi, uvicorn, python-multipart

# 動画の変換に ffmpeg / ffprobe を用いる
sudo apt-get install -y ffmpeg
```

### 2. デモのサーバーを起動する

Cloudflare の無料プランでは，1回のアップロードの上限が 100MB である．デモの既定の上限(500MB)の
ままだと，大きな動画が Cloudflare の段階で拒否され，画面にわかりにくいエラーが表示される．
トンネル経由で使う場合は，`programs/demo/demo_settings.py` の上限を 100MB 未満に下げておく．

```python
# programs/demo/demo_settings.py
MAX_UPLOAD_BYTES = 95 * 1024 * 1024
```

サーバーは `127.0.0.1` で待ち受けたまま起動する．`cloudflared` は同じPCから `localhost` へ接続するため，
`0.0.0.0` で待ち受ける必要はない．

```bash
cd identity-anonymizer
source programs/demo/env.sh
$PYTHON programs/demo/app.py --host 127.0.0.1 --port 8000
```

別の端末で，モデルの読み込み(約10秒)が終わったことを確認する．`"ready":true` と表示されればよい．

```bash
curl -s http://127.0.0.1:8000/api/status
# {"ready":true,"error":null,"limits":{...}}
```

### 3. cloudflared をインストールする(初回のみ)

Ubuntu(WSL2 を含む)では，公式の `.deb` パッケージを入れる．WSL2 の場合は，WSL2 の中にインストールする．

```bash
curl -L -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

外部URLの取得方法は2通りある．短時間だけ試す場合は 4-A，継続して公開する場合は 4-B を用いる．

### 4-A. すぐに試す: クイックトンネル(アカウント・ドメイン不要)

Cloudflare のアカウントが無くても，次のコマンドだけで `https://<ランダムな文字列>.trycloudflare.com`
のURLが発行される．

```bash
cloudflared tunnel --url http://localhost:8000
```

ログの中に表示される `https://....trycloudflare.com` が外部URLである．ログが流れて見つけにくい場合は，
次のようにURLだけを取り出せる．

```bash
cloudflared tunnel --url http://localhost:8000 2>&1 | tee /tmp/cloudflared.log | grep -o --line-buffered 'https://[a-z0-9-]*\.trycloudflare\.com'
```

**クイックトンネルには認証を付けられない．URLを知っている人は誰でもデモを使え，他の人がアップロード
した動画も見られる．** その場で数分試す用途に限り，終わったら `Ctrl+C` で必ず止めること．
URLは起動のたびに変わる．また，`~/.cloudflared/config.yml` が存在するとクイックトンネルは起動しない
ため，4-B の設定ファイルは別の名前で作る．

### 4-B. 継続して公開する: 名前付きトンネル + Cloudflare Access

前提として，Cloudflare のアカウントと，DNS を Cloudflare で管理しているドメインが必要である．
以下では，公開するホスト名を `demo.example.com`，トンネルの名前を `anonymizer-demo` とする．

**(1) Cloudflare にログインする．** 表示されたURLをブラウザ(WSL2 の場合は Windows のブラウザ)で開き，
使うドメインを選んで許可する．`~/.cloudflared/cert.pem` が作られる．

```bash
cloudflared tunnel login
```

**(2) トンネルを作り，ホスト名を割り当てる．** `create` の出力に表示されるトンネルのUUIDを控えておく．
認証情報 `~/.cloudflared/<UUID>.json` が作られる．`route dns` で `demo.example.com` の DNS レコード
(CNAME)が自動で作られる．

```bash
cloudflared tunnel create anonymizer-demo
cloudflared tunnel route dns anonymizer-demo demo.example.com
cloudflared tunnel list
```

**(3) 設定ファイルを作る．** クイックトンネルと干渉しないよう，`config.yml` ではなく別の名前にする．
`<UUID>` と `<ユーザー名>` は自分の値に置き換える．

```yaml
# ~/.cloudflared/anonymizer-demo.yml
tunnel: <UUID>
credentials-file: /home/<ユーザー名>/.cloudflared/<UUID>.json
ingress:
  - hostname: demo.example.com
    service: http://localhost:8000
  - service: http_status:404
```

```bash
cloudflared tunnel --config ~/.cloudflared/anonymizer-demo.yml ingress validate
```

**(4) トンネルを起動する前に，Cloudflare Access で利用者を制限する．** デモには認証が無いため，
この設定をせずに公開してはならない．Cloudflare の Zero Trust ダッシュボード
(<https://one.dash.cloudflare.com>)で次のように設定する(無料プランで利用できる．画面の名称は変更される
ことがある)．

1. 「Access」→「Applications」→「Add an application」で「Self-hosted」を選ぶ．
2. アプリケーションのドメインに `demo.example.com` を指定する．
3. ポリシーを追加し，Action を「Allow」，Include を「Emails」として，利用を許可するメールアドレスを列挙する．
4. ログイン方法に「One-time PIN」を選ぶ(許可したメールアドレスに届くコードでログインする)．

**(5) トンネルを起動する．** 外部URLは `https://demo.example.com` である．

```bash
cloudflared tunnel --config ~/.cloudflared/anonymizer-demo.yml run anonymizer-demo
```

サーバーと同じく，トンネルも起動している間だけ有効である．2つの端末(または `tmux`)で
サーバーとトンネルを並べて起動しておく．

### 5. 動作を確認する

外部URLをブラウザで開き，「準備完了」と表示されることを確認する．4-B の場合は，最初に Cloudflare
Access のログイン画面が表示され，許可したメールアドレスでログインした後にデモの画面が表示される．
`https://` で開いているため，他のPCやスマートフォンからでも「カメラを起動」で撮影できる．

クイックトンネルの場合は，コマンドラインからも確認できる(4-B では Access のログイン画面へ
転送されるため，この確認はブラウザで行う)．

```bash
curl -s https://<ランダムな文字列>.trycloudflare.com/api/status
```

### 6. 停止と後片付け

サーバーとトンネルは，それぞれの端末で `Ctrl+C` を押して止める．名前付きトンネルを使わなくなった
場合は，トンネルを削除し，`route dns` で作られた `demo.example.com` の DNS レコードを Cloudflare の
ダッシュボードから削除する．

```bash
cloudflared tunnel delete anonymizer-demo
```

### 公開する際の注意

- **利用者の制限**: デモには認証が無い．継続して公開する場合は，必ず Cloudflare Access で利用者を
  制限すること．
- **顔の映った動画の扱い**: 通信は Cloudflare の中継点で一度復号される．第三者の顔を扱う場合は，
  本人の同意の範囲と所属機関の規程を確認すること．処理した動画は `outputs/demo/jobs/` に直近20件が残る．
- **認証情報の管理**: `~/.cloudflared/cert.pem` と `~/.cloudflared/<UUID>.json` はトンネルを操作できる
  認証情報である．リポジトリに含めたり，他人と共有したりしないこと．
- **PCの稼働**: このPCがスリープ・シャットダウンすると外部から使えなくなる．
- **同時利用**: GPUは1つであるため，複数人が同時に使うと順番待ちになる(待ち件数は画面に表示される)．

## GHOSTへの変更点

GHOST本体(`third_party/ghost`)への変更は，`patches/0001-silence-inner-progress-bars.patch`
の1件のみである．動画処理中の内側のループで進捗バーが二重に表示されるのを防ぐための，
`utils/inference/video_processing.py` 内3箇所での局所的な `tqdm` の無効化にとどまる．
それ以外のGHOSTのコードは変更していない．

## ライセンス

本リポジトリは Apache License 2.0 の下で公開する．GHOST・ArcFace・DEX等の第三者ソフトウェア・
モデルの利用状況については [NOTICE](NOTICE) を参照すること．

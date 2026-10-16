"""
ex002_partial_prior_sampling の全スクリプトで共有する設定・パス・小さな補助関数をまとめたモジュール．

既存の `src/`，`third_party/` は変更せず，`sys.path` へ追加して読み込むだけにする．
ArcFaceの顔ベクトルは ex001_vae_latent_diagnosis で作成したキャッシュを再利用する．
"""

import json
import os
import sys

# cuBLASを決定論的に動作させるための設定(CUDAの初期化より前に設定する必要がある)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EXPERIMENT_NAME = "ex002_partial_prior_sampling"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DEV_KIT_SRC_DIR = os.path.join(REPO_ROOT, ".ai", "ai-dev-kit", "src")

for _path in (SRC_DIR, DEV_KIT_SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- 入力 ---
UTKFACE_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "UTKFace", "images")
VAE_WEIGHT_PATH = os.path.join(REPO_ROOT, "weights", "anonymizers", "vae", "vae_512_128.pt")
ARCFACE_CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "ex001_vae_latent_diagnosis", "arcface_utkface")

# --- 出力 ---
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", EXPERIMENT_NAME, "labels_groups_split")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", EXPERIMENT_NAME)

# --- 実験条件 (order_002) ---
#: 1枚の画像・1つの設定あたりのサンプル数 K
NUM_SAMPLES = 10
#: サンプリングのノイズとM2rの次元の選択に用いる乱数シード
SEEDS = [0, 1, 2]
#: データ分割と分類器の学習に用いる乱数シード (全ての実験で固定)
SPLIT_SEED = 0
#: 評価用データの割合
EVAL_FRACTION = 0.2
#: 重複とみなすArcFace空間でのコサイン類似度の閾値
DUPLICATE_COSINE = 0.9
#: 年代ビンの下限 (0-9, 10-19, ..., 70+)
AGE_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70]
#: 実験Aで分類精度を比べる次元数
TOP_K_FOR_PROBE = [8, 16, 32, 64, 128]

#: 各方式で掃引するパラメータ
M0_ALPHAS = [0.0, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
M1_TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
M2_KS = [0, 8, 16, 32, 64, 96, 128]
M3_KS = [16, 32, 64]
M3_GAMMAS = [0.0, 0.25, 0.5, 0.75, 0.9]

#: 同じ類似度での比較表に用いる cos(v_tilde, v) の目標値
TARGET_COSINES = [0.6, 0.5, 0.4, 0.3]
#: 要約統計量として出力する分位点
QUANTILES = [0.05, 0.95]


def condition_dir(method: str, hyperparameter: str, seed: int) -> str:
    """
    `outputs/{実験名}/{手法}/{ハイパーパラメータ}/{seed}/` 形式の保存先ディレクトリを作成して返す．

    引数:
        method (str): 実験の種類 (例: "expB_sampling")．
        hyperparameter (str): ハイパーパラメータを表す文字列 (例: "K10")．
        seed (int): 乱数シード．
    戻り値:
        target_dir (str): 作成済みの保存先ディレクトリのパス．
    """
    target_dir = os.path.join(OUTPUT_DIR, method, hyperparameter, str(seed))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def save_json(obj: dict, path: str) -> None:
    """
    辞書をJSONファイルとして保存する．

    引数:
        obj (dict): 保存する辞書 (JSONへ変換可能な値のみを含むこと)．
        path (str): 保存先のパス．
    戻り値:
        なし．
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """
    JSONファイルを辞書として読み込む．

    引数:
        path (str): 読み込むJSONファイルのパス．
    戻り値:
        obj (dict): 読み込んだ辞書．
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finished(target_dir: str) -> bool:
    """
    その条件の実験が正常終了済みかどうかを，完了マーカー `meta.json` の有無で判定する．

    引数:
        target_dir (str): 条件ごとの保存先ディレクトリ．
    戻り値:
        finished (bool): 正常終了済みであれば True．
    """
    return os.path.exists(os.path.join(target_dir, "meta.json"))
